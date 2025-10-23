import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')

import cv2
import shutil
import os
from itertools import repeat

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
# os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
# os.add_dll_directory(r'D:\servicebio_deblurred\c12_p311\bin')
from PIL import Image, ImageOps
import shutil

Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation
import gc
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED256, DoGHardNet
from lightglue.utils import load_image, rbd, resize_image, load_image1
import numpy as np
import os
import time
import torch
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
# os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
# os.add_dll_directory(r'D:\servicebio_deblurred\c12_p311\bin')
import cv2
import torch


# clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(25, 25))

def get_color_img(img, color):
    res_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(img.shape[2]):
        color_img = Image.fromarray(img[:, :, i])
        color_img = ImageOps.colorize(color_img, black='black', white=color)
        res_img = np.clip(res_img + np.array(color_img), 0, 255)
    res_img = Image.fromarray(res_img)
    return res_img


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading


def auto_register(fixed_path, moving_path, device, extractor, matcher):
    try:
        # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
        # extractor = DISK(max_num_keypoints=256).eval().cuda()  # load the extractor
        # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

        # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
        # print('进入函数')
        # print(fixed_path.shape)
        # fixed_path=get_color_img(fixed_path, 'blue')
        # moving_path=get_color_img(moving_path, 'blue')
        # fixed_path=np.array(fixed_path)
        # moving_path=np.array(moving_path)
        fixed_path = load_image1(fixed_path, resize=(fixed_path.shape[0] // 40, fixed_path.shape[1] // 40))
        moving_path = load_image1(moving_path, resize=(moving_path.shape[0] // 40, moving_path.shape[1] // 40))
        print(fixed_path.shape)
        # fixed_path = clahe.apply(cv2.cvtColor(fixed_path, cv2.COLOR_BGR2GRAY))
        # moving_path = clahe.apply(cv2.cvtColor(moving_path, cv2.COLOR_BGR2GRAY))
        # fixed_path=cv2.cvtColor(fixed_path, cv2.COLOR_GRAY2BGR)
        # moving_path=cv2.cvtColor(moving_path, cv2.COLOR_GRAY2BGR)

        # print(fixed_path.shape)

        # print(fixed_path.shape)
        # print(moving_path.shape)
        fixed = load_image(fixed_path)
        fixed = fixed.to(device)
        # print('加载图片1')
        moving = load_image(moving_path)
        moving = moving.to(device)
        s = time.time()
        # print(s)
        # print('加载图片2')

        # extract local features
        feats0 = extractor.extract(fixed)  # auto-resize the image, disable with resize=None
        feats1 = extractor.extract(moving)
        print('时间', (time.time() - s) * 1000)

        # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        print('时间', (time.time() - s) * 1000)

        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        print('时间', (time.time() - s) * 1000)

        pts0 = points0.cpu().numpy().astype(np.float32)  # 匹配点来自你的代码
        pts1 = points1.cpu().numpy().astype(np.float32)

        # 方法一：单应性矩阵（适用于平面透视变换）
        H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_DEFAULT	)
        print(f"RANSAC 结果 | 内点: {sum(mask)} | 误匹配点: {len(mask) - sum(mask)}")
        #
        # tx = H[0, 2]  # 水平偏移（左右）
        # ty = H[1, 2]  # 垂直偏移（上下）
        # shift = [int(ty * 40), int(tx * 40)]
        # print('shif',shift)
        offsets = pts1 - pts0

        translation = np.median(offsets, axis=0)
        t_x, t_y = translation[0], translation[1]
        shift = [-int(t_y * 40), -int(t_x * 40)]
        # print(t_x,t_y)
        # 方法二：光流法（适用于非平面透视变换）
        print(shift)
        # print(H)

        angle_rad = np.arctan2(H[1, 0], H[0, 0])  # atan2(b, a)
        angle_deg = np.degrees(angle_rad)
        # s=time.time()
        result = evaluate_homography(
            H,
            matches=matches,
            img_shape=fixed.shape[1:],  # (h, w)
        )
        if not result['valid']:
            print(f"❌ 配准失败：{result['reason']}")
            print("具体参数：", result['metrics'])
        else:
            print("✅ 变换合理")
            print("详细参数：", result['metrics'])
            # print((time.time()-s)*1000)
        # 4. 应用变换
    except Exception as e:  # 捕获所有异常
        print(e)
    del feats0, feats1, matches01, points0, points1, pts0, pts1
    torch.cuda.empty_cache()
    gc.collect()
    return shift, angle_deg, result['valid']


def evaluate_homography(
        H,
        matches=None,
        img_shape=None,
        max_translation=500,  # 最大允许平移距离（px）
        max_angle_deg=30,  # 最大允许旋转角度
        min_scale=0.5,  # 最小允许缩放（`<1` = 缩小）
        max_scale=1.5,  # 最大允许缩放（`>1` = 放大）
        max_aspect_ratio=1.5,  # 最大宽高比变化（`scale_x / scale_y`）
        max_shear_deg=15,  # 最大允许剪切角度
        min_matches=10,  # 最少匹配点数
        min_inlier_ratio=0.5  # RANSAC 最小内点比例
):
    """
    评价单应性矩阵 H 是否合理

    Args:
        H (np.ndarray): 3×3 单应性矩阵
        matches (list): OpenCV 的匹配点列表（可选）
        img_shape (tuple): 图像大小 (h, w)（用于检查平移是否越界）

    Returns:
        dict: 包含评分 {'valid': bool, 'reason': str, 'metrics': dict}
    """
    # 提取参数
    tx, ty = H[0, 2], H[1, 2]  # 平移
    angle_rad = np.arctan2(H[1, 0], H[0, 0])  # 旋转角（弧度）
    angle_deg = np.degrees(angle_rad)  # 转为角度
    # 计算缩放因子
    scale_x = np.sqrt(H[0, 0] ** 2 + H[0, 1] ** 2)
    scale_y = np.sqrt(H[1, 0] ** 2 + H[1, 1] ** 2)
    print(angle_deg)
    # 计算剪切角
    shear_deg = np.degrees(np.arctan2(H[0, 1], H[0, 0]))
    # 检查平移是否越界（如果给定图像大小）
    if img_shape:
        h, w = img_shape
        if (abs(tx) > w) or (abs(ty) > h):
            return {
                'valid': False,
                'reason': '平移超出图像边界',
                'metrics': {
                    'tx': tx, 'ty': ty,
                    'max_allowed_tx': w,
                    'max_allowed_ty': h
                }
            }
    # 检查旋转
    if abs(angle_deg) > max_angle_deg:
        return {
            'valid': False,
            'reason': f'旋转角度过大 ({angle_deg:.1f} > {max_angle_deg})',
            'metrics': {
                'angle_deg': angle_deg,
                'max_angle_deg': max_angle_deg
            }
        }
    # 检查缩放
    if (scale_x < min_scale) or (scale_x > max_scale):
        return {
            'valid': False,
            'reason': f'X 方向缩放异常 ({scale_x:.2f} 不在 [{min_scale}, {max_scale}] 内)',
            'metrics': {'scale_x': scale_x, 'min_scale': min_scale, 'max_scale': max_scale}
        }
    if (scale_y < min_scale) or (scale_y > max_scale):
        return {
            'valid': False,
            'reason': f'Y 方向缩放异常 ({scale_y:.2f} 不在 [{min_scale}, {max_scale}] 内)',
            'metrics': {'scale_y': scale_y, 'min_scale': min_scale, 'max_scale': max_scale}
        }
    # 检查宽高比（防止非均匀缩放）
    aspect_ratio = scale_x / scale_y
    if (aspect_ratio > max_aspect_ratio) or (aspect_ratio < 1 / max_aspect_ratio):
        return {
            'valid': False,
            'reason': f'宽高比变化异常 ({aspect_ratio:.2f} 不在 [{1 / max_aspect_ratio:.2f}, {max_aspect_ratio:.2f}] 内)',
            'metrics': {
                'scale_x': scale_x, 'scale_y': scale_y,
                'aspect_ratio': aspect_ratio,
                'max_aspect_ratio': max_aspect_ratio
            }
        }
    # 检查剪切
    if abs(shear_deg) > max_shear_deg:
        return {
            'valid': False,
            'reason': f'剪切变形过强 ({shear_deg:.1f} > {max_shear_deg})',
            'metrics': {'shear_deg': shear_deg, 'max_shear_deg': max_shear_deg}
        }
    # 如果全部通过，返回 True
    return {
        'valid': True,
        'reason': '变换矩阵合理',
        'metrics': {
            'tx': tx, 'ty': ty,
            'angle_deg': angle_deg,
            'scale_x': scale_x, 'scale_y': scale_y,
            'aspect_ratio': aspect_ratio,
            'shear_deg': shear_deg
        }
    }


#
def merge_with_offset(moving, shift):
    """创建一个背景图像并将移动图像粘贴到相应位置"""
    # 获取fixed图像的大小
    h, w = moving.shape[:2]

    # 创建与fixed同样大小的背景图像，初始化为零（黑色背景）
    result = np.zeros_like(moving)
    # print(111)
    # 将moving图像粘贴到result图像中，偏移后进行粘贴
    y_offset, x_offset = shift
    # print(222)
    # 确保粘贴的位置不会超出背景图像边界
    if y_offset >= 0:
        start_y = y_offset
        end_y = min(start_y + moving.shape[0], h)
    else:
        start_y = 0
        end_y = min(moving.shape[0] + y_offset, h)

    if x_offset >= 0:
        start_x = x_offset
        end_x = min(start_x + moving.shape[1], w)
    else:
        start_x = 0
        end_x = min(moving.shape[1] + x_offset, w)

    # # 获取要粘贴的部分
    moving_part = moving[start_y - y_offset:end_y - y_offset, start_x - x_offset:end_x - x_offset]

    # # 粘贴到背景图像相应位置
    result[start_y:end_y, start_x:end_x] = moving_part
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # dest_y1 = max(0, y_offset)          # = max(0, 2) = 2
    # dest_y2 = min(h, y_offset + h)  # = min(5, 2+5) = 5
    # src_y1 = max(0, -y_offset)          # = max(0, -2) = 0
    # src_y2 = min(h, h - y_offset)   # = min(5, 5-2) = 3
    # dest_x1 = max(0, x_offset)          # = max(0, -1) = 0
    # dest_x2 = min(w, x_offset + w)  # = min(5, -1+5) = 4
    # src_x1 = max(0, -x_offset)          # = max(0, 1) = 1
    # src_x2 = min(w, w - x_offset)   # = min(5, 5-(-1)) = 5
    # # 正确切片和粘贴
    # moving_part = moving_img[src_y1:src_y2, src_x1:src_x2]  # moving_img[0:3, 1:5]
    # result[dest_y1:dest_y2, dest_x1:dest_x2] = moving_part
    # 转换为 PIL 图像
    # result = Image.fromarray(result)
    # print(result.size)
    return result


def process_subdir(subdir_path, shift, angle):
    """处理单个子目录的任务函数（在每个线程中执行）"""
    print(f"线程 {os.getpid()}-{threading.current_thread().name} 正在处理子目录: {subdir_path}")
    for i in os.listdir(subdir_path):
        # print(i)
        # print(angle)
        if i.endswith('image.png'):
            # print('存在这个图片',i)
            moving_img = cv2.imread(os.path.normpath(os.path.join(subdir_path, i)), cv2.IMREAD_GRAYSCALE)
            # print('存在这个图片,且它的shape是',moving_img.shape)
            # print('偏移量',shift)
            # try:
            result_img = merge_with_offset(moving_img, shift)
            #     # print('偏移成功',moving_img.size)
            # except Exception as e:
            #     print(e)

            output_path = os.path.join(subdir_path, i).replace('.png', '_moving.png')

            if abs(angle) > 2:
                result_img = Image.fromarray(result_img)
                result_img = result_img.rotate(-angle)
                result_img.save(output_path)
            else:
                cv2.imwrite(output_path, result_img)
                print(f"已保存到: {output_path}")
                # 使用线程池处理每个子目录（每个进程使用两个线程）
                # del fixed_img, moving_img, result_img
                # torch.cuda.empty_cache()
                # gc.collect()

            # if abs(angle)>1:
            # result_img = merge_with_offset(moving_img, shift)
    #        h, w = result_img.shape[:2]
    #
    #        # 旋转中心（默认图像中心）
    #        center = (w // 2, h // 2)
    #        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
    #        result_img = cv2.warpAffine(result_img, rotation_matrix, (w, h))
    #        #     moving_img=moving_img.rotate(-angle)
    #        # moving_img = np.array(moving_img)
    #        output_path = os.path.join(subdir_path, i).replace('.png', '_moving.png')
    # # shape=(H, W), dtype=uint8/uint16
    #        # print(output_path,moving_img.shape)
    #        cv2.imwrite(output_path,result_img)
    # moving_path = os.path.join(subdir_path, i)
    # 这里可以添加对每个子目录的处理逻辑
    # 例如：读取图像、进行处理等
    # print(moving_path)
    # shutil.copy(moving_path, moving_path.replace('.png', '_moving.png'))
    # 处理图像
    # img = cv2.imread(moving_path)
    # 进行一些处理
    # cv2.imwrite(moving_path.replace('.png', '_moving.png'), img)
    # 在这里添加你的实际处理逻辑
    # 例如：遍历文件、处理数据等
    # pass


def process_directory(dir_path, fixed_img, fixed_name):
    """在进程中处理目录的函数（每个进程处理一个目录）"""
    print(f"进程 {os.getpid()} 正在处理目录: {dir_path}")
    # SuperPoint+LightGlue
    # print(fixed_name)

    device = torch.device("cuda")  # 也可根据需求选择其他GPU（如cuda:1/2/3）
    # extractor = DISK(max_num_keypoints=2048).eval().to(device)  # load the extractor
    extractor = DISK(max_num_keypoints=256, preprocess_conf={
        "resize": 256,
        "grayscale": True,
    }).eval().to(device)
    matcher = LightGlue(features='disk').eval().to(device)  # load the matcher
    # 获取该目录下的所有子目录
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path)
               if os.path.isdir(os.path.join(dir_path, d)) and d != 'DAPI']
    # print(subdirs)
    p1 = dir_path.split('\\')[-1]
    # print(dir_path)
    print(p1)
    moving_path = fr'{dir_path}/DAPI/{p1}_DAPI_merged_image.png'
    moving_img = cv2.imread(os.path.normpath(moving_path), cv2.IMREAD_GRAYSCALE)
    # print('moving_img',moving_img.shape)
    shift, angle, _ = auto_register(fixed_img, moving_img, device, extractor, matcher)

    print('旋转角度为', angle, '偏移量为', shift)
    print("旋转偏移计算完成，判断旋转角度...")
    # if abs(angle) > 2:
    #     print("存在旋转，重新计算偏移...")
    #     moving_img1 = Image.fromarray(moving_img)
    #     moving_img1 = moving_img1.rotate(-angle)
    #     moving_img1 = np.array(moving_img1)
    #     shift, angle1, _ = auto_register(fixed_img, moving_img1,device,extractor,matcher)
    #     angle=angle+angle1
    #     print('旋转角度为', angle1, '偏移量为', shift)
    # print("正在将图像分块处理并合成...")
    # try:
    result_img = merge_with_offset(moving_img, shift)
    # h, w = result_img.shape[:2]
    #
    # # 旋转中心（默认图像中心）
    # center = (w // 2, h // 2)
    # rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
    # result_img = cv2.warpAffine(result_img, rotation_matrix, (w, h))
    # except Exception as e:
    #     print(e)
    # result_img = result_img.rotate(-angle)
    output_path = moving_path.replace('.png', '_moving.png')

    if abs(angle) > 2:
        result_img = Image.fromarray(result_img)
        result_img = result_img.rotate(-angle)
        result_img.save(output_path)
    else:
        cv2.imwrite(output_path, result_img)
        print(f"已保存到: {output_path}")
        # 使用线程池处理每个子目录（每个进程使用两个线程）
        del fixed_img, moving_img, result_img
        torch.cuda.empty_cache()
        gc.collect()
    # print("正在保存结果...")
    # # result_img.save(output_path)
    # result_img = np.array(result_img)

    # shift1,angle1,_ = auto_register(fixed_img, result_img,device,extractor,matcher)
    # result_img = merge_with_offset(result_img, shift1)
    # if abs(angle1)>1:

    #     result_img = result_img.rotate(-angle1)
    # result_img = np.array(result_img)

    # shift2,angle2,_ = auto_register(fixed_img, result_img,device,extractor,matcher)

    # result_img = merge_with_offset(result_img, shift2)
    # if abs(angle1)>1:
    # else:
    #     result_img = result_img.rotate(-angle)
    # result_img = np.array(result_img)

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_subdir, subdirs, repeat(shift), repeat(angle))


def main(params):  # 修改主函数接收参数字典
    A_dir = params["data_dir"]
    s = time.time()
    # 原始目录结构
    # A_dir = r'HC356_Gray_date/HC356'

    # 确保A目录存在
    if not os.path.exists(A_dir):
        print(f"目录 {A_dir} 不存在！")
        return

    # 获取A目录下所有子目录
    all_subdirs = [os.path.join(A_dir, d) for d in sorted(os.listdir(A_dir))
                   if os.path.isdir(os.path.join(A_dir, d))]
    len_w = len(all_subdirs)
    # if len(all_subdirs) < 4:
    #     print("A目录下需要至少4个子目录！")
    #     return

    # 步骤1：取出第一个子目录（可根据需要处理）
    first_dir = all_subdirs[0]
    print(f"取出第一个子目录: {first_dir}")
    fixed_name = ''
    # 这里可以添加对第一个子目录的处理
    for i in os.listdir(first_dir):
        if i == 'DAPI':
            for j in os.listdir(os.path.join(first_dir, i)):
                if j.endswith('image.png'):
                    fixed_path = os.path.join(first_dir, i, j)
                    # print(j)
                    fixed_name = j
                    # print(fixed_name)
                    fixed_img = cv2.imread(os.path.normpath(fixed_path), cv2.IMREAD_GRAYSCALE)
                    # print(fixed_path)
                    shutil.copy(fixed_path, fixed_path.replace('.png', '_moving.png'))
                    shutil.copy(fixed_path, fixed_path.replace('merged_image', 'out1'))

        else:
            for j in os.listdir(os.path.join(first_dir, i)):
                if j.endswith('image.png'):
                    shutil.copy(os.path.join(first_dir, i, j),
                                os.path.join(first_dir, i, j).replace('.png', '_moving.png'))
                    shutil.copy(os.path.join(first_dir, i, j),
                                os.path.join(first_dir, i, j).replace('merged_image', 'out1'))

    remaining_dirs = all_subdirs[1:len_w]  # 只取接下来的3个
    # print(fixed_img.shape)
    # 步骤3：对这3个子目录创建3个进程，每个进程再创建2个线程
    print("\n开始并行处理（3个进程，每个进程3个线程）:")
    with ProcessPoolExecutor(max_workers=len_w - 1) as executor:
        executor.map(process_directory, remaining_dirs, repeat(fixed_img), repeat(fixed_name))

    print("\n处理完成！")
    print(time.time() - s)
# if __name__ == "__main__":
#     # 测试用默认参数
#     default_params = {
#         "input_dir": "HC356_Gray_date",
#         "ref_name": "HC356_1_DAPI"
#     }
#     main(default_params)