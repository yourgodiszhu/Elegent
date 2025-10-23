import gc

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 取消图片大小限制

import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
# os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
# os.add_dll_directory(r'D:\servicebio_deblurred\c12_p311\bin')
import cv2
import numpy as np
import os
import cv2
import numpy as np
import tifffile
from tqdm import tqdm
def cv2_imread_chinese(file_path):
    with open(file_path, "rb") as f:
        img_buff = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(img_buff, cv2.IMREAD_COLOR)
    return img

class JPGtoSVSConverter:
    def __init__(self, input_jpg_path, output_svs_path):
        """
        初始化转换器
        :param input_jpg_path: 输入的大尺寸JPG图片路径
        :param output_svs_path: 输出的SVS文件路径
        """
        self.input_path = input_jpg_path
        self.output_path = output_svs_path
        self.image = None
        self.tile_size = 512  # SVS文件的分块大小

    def load_image(self):
        """加载JPG图片"""
        print(f"正在加载图片: {self.input_path}")
        self.image = cv2_imread_chinese(self.input_path)
        self.image=cv2.imread(self.input_path)
        if self.image is None:
            raise ValueError("无法加载图片，请检查路径是否正确")
        # 转换为RGB格式（OpenCV默认是BGR）
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print(f"图片加载成功，尺寸: {self.image.shape}")

    def process_large_image(self, img, resize_factor=0.5, block_size=512):
        """
        分块处理大图像，避免内存不足
        :param img: 输入图像
        :param resize_factor: 缩放因子
        :param block_size: 处理块大小
        :return: 处理后的图像
        """
        height, width = img.shape[:2]
        resized_height = int(height * resize_factor)
        resized_width = int(width * resize_factor)

        resized_img = np.zeros((resized_height, resized_width, 3), dtype=img.dtype)

        # 按块处理图像
        for y in tqdm(range(0, height, block_size), desc="处理图像块"):
            for x in range(0, width, block_size):
                block = img[y:y + block_size, x:x + block_size]
                if block.shape[0] == 0 or block.shape[1] == 0:
                    continue

                # 缩放块
                block_resized = cv2.resize(block, None, fx=resize_factor, fy=resize_factor,
                                           interpolation=cv2.INTER_LANCZOS4)

                # 计算新位置
                new_y = int(y * resize_factor)
                new_x = int(x * resize_factor)

                # 放置缩放后的块
                resized_img[new_y:new_y + block_resized.shape[0],
                new_x:new_x + block_resized.shape[1]] = block_resized

        return resized_img

    def generate_pyramid_levels(self, levels=5):
        """
        生成金字塔各层级的图像
        :param levels: 金字塔层数
        :return: 金字塔各层级图像列表
        """
        pyramid = [self.image]
        for i in tqdm(range(1, levels), desc="生成金字塔层级"):
            # 使用分块处理大图像
            resized = self.process_large_image(pyramid[-1], resize_factor=0.5)
            pyramid.append(resized)
        return pyramid

    def save_as_svs(self):
        """将图像保存为SVS格式"""
        print("开始转换为SVS格式...")

        # 生成金字塔图像
        pyramid_levels = self.generate_pyramid_levels()

        # SVS文件描述信息
        description = (
            "Aperio Image Library Fake\n"
            "ABC |AppMag = 20|Filename = converted_image.svs|MPP = 0.25"
        )

        # 写入TIFF文件
        with tifffile.TiffWriter(self.output_path, bigtiff=True) as tif:
            # JPEG压缩参数
            compression = ['JPEG', 80, dict(outcolorspace='YCbCr')]
            kwargs = dict(
                photometric='rgb',
                planarconfig='CONTIG',
                compression=compression,
                dtype=np.uint8,
                tile=(self.tile_size, self.tile_size))

            # 写入第一层(最高分辨率)
            tif.write(
                pyramid_levels[0],
                subifds=len(pyramid_levels) - 1,
                description=description,
                **kwargs
            )

            # 写入其他层级(低分辨率)
            for i, level in enumerate(pyramid_levels[1:], 1):
                tif.write(
                    level,
                    subfiletype=1 if i == len(pyramid_levels) - 1 else 0,
                    **kwargs
                )

            print(f"SVS文件已保存到: {self.output_path}")
        del pyramid_levels
        gc.collect()
# if __name__ == '__main__':
# # 使用示例
#     for i in os.listdir(rf'all_out_png'):
#         if i.endswith('.png'):
#             input_jpg = rf"all_out_png\{i}"# 替换为您的JPG文件路径
#             output_svs = rf"all_out_png\{i.split('.')[0]}.svs" # 输出的SVS文件路径
#
#             converter = JPGtoSVSConverter(input_jpg, output_svs)
#             converter.load_image()
#             converter.save_as_svs()
#     input_jpg = r"D:\3d\all_out_png\Maximum_DAPI.png" # 替换为您的JPG文件路径
#     output_svs = r"D:\3d\all_out_png\Maximum_DAPI.svs" # 输出的SVS文件路径
# #
#     converter = JPGtoSVSConverter(input_jpg, output_svs)
#     converter.load_image()
#     converter.save_as_svs()

import concurrent.futures
import time
from typing import List, Callable

#
def threaded_executor(
        task_func: Callable,
        params,
        max_workers: int = 1,
        timeout: float = None
) -> dict:
    """
    多线程执行器框架
    :param task_func: 目标任务函数（需自行实现）
    :param task_args_list: 每个任务参数组成的元组列表
    :param max_workers: 线程数 (默认16)
    :param timeout: 单个任务超时时间(秒)
    :return: 结构为 {任务参数: (是否成功, 结果/错误信息)}
    """
    task_args_list = [(rf"{params['save_dir']}\{i}",) for i in os.listdir(f'{params["save_dir"]}') if i.endswith('.png') ]
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        future_to_args = {
            executor.submit(task_func, *args): args
            for args in task_args_list
        }

        # 异步获取结果
        for future in concurrent.futures.as_completed(future_to_args, timeout=timeout):
            args = future_to_args[future]
            try:
                results[args] = (True, future.result())
            except Exception as e:
                print(3)
                results[args] = (False, str(e))
    successful = sum(1 for r in results.values() if r[0])
    # print(f"\n执行完成！耗时 {time.time() - start_time:.2f}秒")
    print(f"成功率: {successful}/{len(task_args_list)}")

    # 打印前5个结果示例
    print("\n示例结果:")
    for args, (status, result) in list(results.items())[:]:
        print(f"参数 {args} => {'成功' if status else '失败'}: {result}")

    return results
#
#
# # ============== 以下是你的具体任务实现 ==============
def task_function(param1):
    print(param1)
    """TODO: 在这里实现你的线程任务逻辑"""
    # 示例任务：模拟耗时操作
    # time.sleep(1)
    input_jpg = param1# 替换为您的JPG文件路径
    output_svs = fr'{param1.split(".")[0]}.svs' # 输出的SVS文件路径

    converter = JPGtoSVSConverter(input_jpg, output_svs)
    converter.load_image()
    converter.save_as_svs()
    del converter
    gc.collect()
    # 返回计算结果（替换为实际逻辑）
    return f"Processed {param1}"


# if __name__ == "__main__":
    # ===== 任务参数生成（根据需求修改） =====
    # 示例：生成100个任务的参数列表
    # for i in os.listdir(rf'all_out_png'):
    #     if i.endswith('.png'):
    #         input_jpg = rf"all_out_png\{i}"# 替换为您的JPG文件路径
    #         task_args.append(input_jpg)
    #         output_svs = rf"all_out_png\{i.split('.')[0]}.svs" # 输出的SVS文件路径
    #
    #         converter = JPGtoSVSConverter(input_jpg, output_svs)
    #         converter.load_image()
    #         converter.save_as_svs()
    # ===== 执行多线程任务 =====




    # task_args = [(rf"all_out_png\{i}",) for i in os.listdir("all_out_png") if i.endswith('.png') ]

    # start_time = time.time()
    # print(task_args)

