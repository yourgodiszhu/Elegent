import os
from pathlib import Path

# from Demos.win32console_demo import window_size

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
import cv2
import time
import numpy as np
import concurrent.futures
from typing import Tuple, List
from multiprocessing import Pool, cpu_count
from functools import partial
import re
def cv_imread_unicode(path,gray=True):
    stream = np.fromfile(path, dtype=np.uint8)
    if gray:
        return cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
def cv_imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    result, encoded_img = cv2.imencode(ext, img)
    if result:
        encoded_img.tofile(path)
    else:
        raise Exception(f"cv2.imencode failed: {path}")
def convert_patch_name(filename, base_y=1648, base_x=1648):
    """
    filename: 原始文件名，例如 patch_y0_x1648.png
    base_y, base_x: y、x的基准，用于计算倍数
    返回新的文件名，例如 1_0_1.png
    """
    match = re.search(r'patch_y(\d+)_x(\d+)\.', filename)
    if match:
        y_val = int(match.group(2))
        x_val = int(match.group(1))
        new_name = f"1_{y_val // base_y}_{x_val // base_x}.png"
        return new_name
    else:
        return None

class ImageProcessor:
    def __init__(self, window_size, overlap):
        self.window_size = window_size
        self.overlap = overlap
        
    def split_and_save(self, image_path, output_dir):
        """处理单张图片：读取、切分、保存"""
        try:
            image = universal_imread(image_path)
            if image is None:
                print(f"警告：无法读取图片 {image_path}")
                return

            os.makedirs(output_dir, exist_ok=True)
            crop_center_path = output_dir.replace('crop_moving_img', 'crop_ending_img')
            # print(crop_center_path)
            os.makedirs(crop_center_path, exist_ok=True)
            h, w = image.shape[:2]
            stride_h = self.window_size[0] - self.overlap[0]
            stride_w = self.window_size[1] - self.overlap[1]
            
            for y in range(0, h, stride_h):
                for x in range(0, w, stride_w):
                    end_y = min(y + self.window_size[0], h)
                    end_x = min(x + self.window_size[1], w)
                    patch = image[y:end_y, x:end_x]
                    
                    if patch.shape[:2] != self.window_size:
                        padded_patch = np.zeros((*self.window_size, *image.shape[2:]), dtype=image.dtype)
                        padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                        patch = padded_patch
                    
                    output_path = os.path.join(output_dir, f'1_{x//stride_h}_{y//stride_h}.png')

                    p = Path(output_dir)
                    new_path = p.parents[2]  # 去掉后三个目录
                    current_path=p.parents[1]
                    first=os.listdir(new_path)[0]
                    # # print(output_path)
                    # # print(new_path)
                    # # print(first)
                    # # print(current_path)
                    # # print(first)
                    # # print(os.path.basename(current_path)==first)
                    # if os.path.basename(current_path)==first:
                    #     h1, w1 = patch.shape[:2]
                    #     size = 1848
                    #     y_start = max(0, (h1 - size) // 2)
                    #     x_start = max(0, (w1 - size) // 2)
                    #     center_patch = patch[y_start:y_start + size, x_start:x_start + size]
                    #     file_name = convert_patch_name(f'patch_y{y}_x{x}.png')
                    #
                    #     cv_imwrite_unicode(os.path.join(crop_center_path,file_name),center_patch)
                    #     cv_imwrite_unicode(output_path, patch)

                    # else:
                    cv_imwrite_unicode(output_path, patch)
                    # print(patch.shape)
            
            print(f"完成：{os.path.basename(image_path)} → {output_dir}")
        except Exception as e:
            print(f"处理 {image_path} 出错: {str(e)}")

def process_channel(channel_dir, processor):
    """处理单个channel目录下的图片（线程级任务）"""
    for filename in os.listdir(channel_dir):
        if filename.endswith('_moving.png'):
            image_path = os.path.join(channel_dir, filename)
            print(image_path)
            output_dir = os.path.join(channel_dir, 'crop_moving_img')
            processor.split_and_save(image_path, output_dir)

def process_single_category(cls_all_path,window_size, over_lab, thread_per_process=4):
    """处理一个cls_all分类（进程级任务）"""
    print(f"\n开始处理分类 [{os.path.basename(cls_all_path)}]".center(50, '-'))
    t_start = time.time()
    processor = ImageProcessor(window_size=window_size,overlap=over_lab)
    
    # 收集所有channel目录
    channel_dirs = []
    for channel in os.listdir(cls_all_path):
        channel_dir = os.path.join(cls_all_path, channel)
        if os.path.isdir(channel_dir):
            channel_dirs.append(channel_dir)
    
    # 使用线程池处理各个channel
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_per_process) as executor:
        futures = [executor.submit(process_channel, channel_dir, processor) 
                  for channel_dir in channel_dirs]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取结果（如有异常会在这里抛出）
    
    print(f"分类 [{os.path.basename(cls_all_path)}] 处理完成 | 耗时: {time.time()-t_start:.2f}秒".center(50, '-'))

def universal_imread(file_path):
    """通用图像读取函数（支持中文路径）"""
    # OpenCV优先模式
    try:
        return cv_imread_unicode(file_path)  # 上面的实现
    except:
        # 回退到PIL模式
        try:
            return np.array(Image.open(file_path))
        except:
            raise RuntimeError(f"所有图像读取方式均失败: {file_path}")
def main_proc(params, process_workers=3, thread_per_process=3):
    """主处理函数：进程池+线程池"""
    # 获取所有需要处理的分类路径
    cls_dir = params['data_dir']
    over_lab=params['over_lab']
    window_size=params['window_size']
    all_tasks = []
    # for cls in os.listdir(cls_dir):
    #     path1 = os.path.join(cls_dir, cls)
    for cls_all in os.listdir(cls_dir):
        path2 = os.path.join(cls_dir, cls_all)
        all_tasks.append(path2)
    
    if not all_tasks:
        print("没有找到需要处理的分类!")
        return
    
    # 配置工作进程数
    if process_workers is None:
        process_workers = min(cpu_count(), len(all_tasks))
    
    print(f"\n共发现 {len(all_tasks)} 个分类 | 使用 {process_workers} 进程 x {thread_per_process} 线程".center(60, '='))
    
    # 进程池处理
    with Pool(processes=process_workers) as pool:
        # 使用partial固定线程数参数
        pool.map(partial(process_single_category, window_size=window_size, over_lab= over_lab,thread_per_process=thread_per_process), all_tasks)

# if __name__ == "__main__":
#     cls_dir = 'HC356_Gray_date/HC356'
#
#     if not os.path.exists(cls_dir):
#         raise FileNotFoundError(f"目录 {cls_dir} 不存在")
#
#     start_time = time.time()
#     print("开始处理所有图片...".center(50, '*'))
    
    # 参数说明：
    # process_workers - 进程数（默认根据CPU核心数自动设置）
    # thread_per_process - 每个进程的线程数（建议2-4）
    # main_proc(cls_dir, process_workers=4, thread_per_process=4)
    
    # print(f"\n所有任务完成! 总耗时: {time.time()-start_time:.2f}秒".center(50, '*'))
