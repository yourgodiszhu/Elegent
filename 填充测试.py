import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
import cv2
from trt_model_lightglue import TRTInference
import re
from typing import List
import torch
import time
import math
import numpy as np
import glob
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED256, DoGHardNet, ALIKED2048
from lightglue.utils import load_image, rbd, load_image1

def histogram_matching(input_img, reference_img):
    """
    超快直方图匹配实现（CPU/GPU双版本可选）
    优化手段：
    1. 跳过冗余颜色空间转换（直接处理灰度图）
    2. 使用快速直方图计算
    3. 向量化LUT映射
    4. 支持批处理（未来扩展）

    参数:
        input_img (np.ndarray): 灰度输入图像 (H,W)
        reference_img (np.ndarray): 灰度参考图像 (H,W)

    返回:
        np.ndarray: 匹配后的灰度图像
    """
    # --- 预处理检查 ---
    assert input_img.ndim == 2, "输入必须是灰度图"
    assert reference_img.ndim == 2, "参考图必须是灰度图"

    # --- 核心优化步骤 ---
    # 1. 快速直方图计算（比np.histogram快3倍）
    hist_input = cv2.calcHist([input_img], [0], None, [256], [0, 256]).flatten()
    hist_ref = cv2.calcHist([reference_img], [0], None, [256], [0, 256]).flatten()

    # 2. 向量化CDF计算（取代循环）
    cdf_input = hist_input.cumsum()
    cdf_input = (cdf_input - cdf_input.min()) * 255 / max(cdf_input.max() - cdf_input.min(), 1e-6)
    cdf_input = cdf_input.astype('uint8')

    cdf_ref = hist_ref.cumsum()
    cdf_ref = (cdf_ref - cdf_ref.min()) * 255 / max(cdf_ref.max() - cdf_ref.min(), 1e-6)
    cdf_ref = cdf_ref.astype('uint8')

    # 3. 快速LUT生成（比循环快100倍）
    lut = np.argmin(np.abs(cdf_input.reshape(-1, 1) - cdf_ref), axis=1).astype('uint8')

    # 4. 应用优化后的LUT
    return cv2.LUT(input_img, lut)
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
    # print(angle_deg)
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

def Aliked_trt(
    # input_path=r"image/2448",
    #
    # device="cuda",
    # top_k=3000,
    # scores_th=0.2,
    # n_limit=20000,
    fixed_path,moving_path,extractor
):
    # args = parse_args()
    # logging.basicConfig(level=logging.INFO)

    # image_loader = ImageLoader(input_path)
    # if trt_model_path is None:
    #     model = ALIKED(
    #         device=device,
    #         top_k=top_k,
    #         scores_th=scores_th,
    #         n_limit=n_limit,
    #     )
        # model.half()
    # else:  # Use TRT version.


    # logging.info("Press 'space' to start. \n Press 'q' or 'ESC' to stop!")

    # img_ref = image_loader[0]
    # img = cv2.imread(filename)
    scale=fixed_path.shape[0]/2048
    fixed_path1 = cv2.resize(fixed_path, (2048, 2048))
    moving_path1 = cv2.resize(moving_path, (2048, 2048))
    img_rgb_ref = fixed_path1
    s=time.time()
    # print(extractor)
    # print(img_rgb_ref.shape)
    feats0 = extractor.run(img_rgb_ref)
    feats0 = {'keypoints': feats0['keypoints'] * scale, 'descriptors': feats0['descriptors'],
              'keypoint_scores': feats0['keypoint_scores'], 'image_size': feats0['image_size']}
    top_k = 1000
    feats0 = {
        'keypoints': feats0['keypoints'][:, :top_k:, :],
        'descriptors': feats0['descriptors'][:, :top_k, :],
        'keypoint_scores': feats0['keypoint_scores'][:, :top_k],
        'image_size': feats0['image_size']
    }
    # print(time.time()-s)

    # kpts_ref = pred_ref["keypoints"]
    # desc_ref = pred_ref["descriptors"]
    # # desc_ref = np.copy(desc_ref)

    # for i in range(1, len(image_loader)):
    img_rgb = moving_path1
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    time2=time.time()
    feats1 = extractor.run(img_rgb)
    feats1 = {'keypoints': feats1['keypoints'] * scale, 'descriptors': feats1['descriptors'],
              'keypoint_scores': feats1['keypoint_scores'], 'image_size': feats1['image_size']}
    # top_k = 1000
    feats1 = {
        'keypoints': feats1['keypoints'][:, :top_k:, :],
        'descriptors': feats1['descriptors'][:, :top_k, :],
        'keypoint_scores': feats1['keypoint_scores'][:, :top_k],
        'image_size': feats1['image_size']
    }
    feats0_ori = feats0.copy()
    feats1_ori = feats1.copy()
    count = (feats1['keypoint_scores'] > 0.8).sum().item()
    print(r'>0.8特征点数', count)

    count1 = (feats1['keypoint_scores'] > 0.1).sum().item()
    # with open('count_result.txt', 'w') as f:  # 'w' 表示写入模式，会覆盖原内容
    #     f.write(str(feats1['keypoint_scores']))  # 写入 count1 的整数值
    count2 = (feats0['keypoint_scores'] > 0.1).sum().item()
    print(r'>0.1特征点数', count1, count2)
    # if count < 3 and count1 < 10 and count2 < 10:
    #     return None

    print('pt')
    # with torch.no_grad():
    if count>=900:
        matcher = LightGlue(features='aliked', filter_threshold=0.00000001).eval().to('cuda')
    else:
        matcher = LightGlue(features='aliked').eval().to('cuda')
    torch.cuda.empty_cache()

    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches']  # indices with shape (K,2)
    # print(matches)
    # print(matches.shape)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    # print('时间', (time.time() - s) * 1000)

    pts0 = points0.cpu().numpy().astype(np.float32)  # 匹配点来自你的代码
    pts1 = points1.cpu().numpy().astype(np.float32)

    # if len(pts0)==0:
    #     print('没有匹配点',feats0_ori['keypoint_scores'][:,:3])
    #     print('没有匹配点',feats1_ori['keypoint_scores'][:,:3])
    # offsets = pts1 - pts0
    # translation = np.median(offsets, axis=0)
    # t_x, t_y = translation[0], translation[1]
    # shift = [-int(t_y ), -int(t_x )]
    # print(shift)
    # 方法一：单应性矩阵（适用于平面透视变换）
    print('计算单应性矩阵')
    try:
        H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_DEFAULT)
    except Exception as e:
        print(e)




    print('计算成功')
    result = evaluate_homography(
        H,
        matches=matches,
        img_shape=fixed_path.shape[:2],  # (h, w)
    )
    if not result['valid']:
        print(f"❌ 配准失败：{result['reason']}")
        # print("具体参数：", result['metrics'])
    else:
        print("✅ 变换合理")
        # print("详细参数：", result['metrics'])
    # print((time.time()-s)*1000)
    # 4. 应用变换

    # del feats0, feats1, matches01, points0, points1, pts0, pts1
    # torch.cuda.empty_cache()
    # gc.collect()
    return H, 0, result['valid']
def flow_cuda(fixed, moving,cpu=None):
    if cpu==False:
        gpu_mats = {
            'fixed': cv2.cuda_GpuMat(),
            'moving_img': cv2.cuda_GpuMat(),
            'corrected': cv2.cuda_GpuMat(),
            'flow': cv2.cuda_GpuMat(),
            'map_x': cv2.cuda_GpuMat(),
            'map_y': cv2.cuda_GpuMat()
        }
        stream = cv2.cuda_Stream()  # 使用异步流

        # 第一次光流计算
        gpu_mats['fixed'].upload(fixed.astype(np.float32), stream)
        gpu_mats['moving_img'].upload(moving.astype(np.float32), stream)
        stream.waitForCompletion()

        farneback1 = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=1, pyrScale=0.2, fastPyramids=False,
            winSize=55, numIters=1, polyN=5, polySigma=1.2, flags=0
        )
        gpu_mats['flow'] = farneback1.calc(gpu_mats['fixed'], gpu_mats['moving_img'], None, stream)
        stream.waitForCompletion()

        # 生成第一次映射
        flow = gpu_mats['flow'].download()
        h, w = fixed.shape[:2]
        y_coords, x_coords = np.indices((h, w))

        # ★关键修改1：确保连续内存和正确类型
        map_x = np.ascontiguousarray((x_coords + flow[..., 0]).astype(np.float32))
        map_y = np.ascontiguousarray((y_coords + flow[..., 1]).astype(np.float32))
        # 释放不再需要的对象
        # del flow, x_coords, y_coords, farneback1
        # torch.cuda.empty_cache()
        # stream = cv2.cuda_Stream()  # 使用异步流

        # 第一次remap - ★关键修改2：显式创建目标GpuMat
        gpu_mats['corrected'].upload(moving.astype(np.float32), stream)
        gpu_mats['map_x'].upload(map_x, stream)
        gpu_mats['map_y'].upload(map_y, stream)
        stream.waitForCompletion()

        # 创建输出GpuMat
        gpu_corrected = cv2.cuda_GpuMat(gpu_mats['corrected'].size(), gpu_mats['corrected'].type())

        # ★关键修改3：使用命名参数的remap调用
        cv2.cuda.remap(
            src=gpu_mats['corrected'],
            dst=gpu_corrected,
            xmap=gpu_mats['map_x'],
            ymap=gpu_mats['map_y'],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            stream=stream
        )
        # 下载结果
        moving = gpu_corrected.download()
        return moving,flow
    else:
        # flow = cv2.calcOpticalFlowFarneback(
        #     fixed, moving_img,
        #     None, pyr_scale=0.2, levels=1, winsize=55, iterations=1, poly_n=5, poly_sigma=1.2, flags=0
        # )
        # h, w = fixed.shape[0], fixed.shape[1]
        #
        # # 光流修正
        # new_coords = np.float32([np.mgrid[0:h, 0:w][1] + flow[..., 0], np.mgrid[0:h, 0:w][0] + flow[..., 1]])
        # moving_img = cv2.remap(moving_img, new_coords[0], new_coords[1], cv2.INTER_LINEAR)
        moving,flow,map_x, map_y=fast_optical_flow_alignment(fixed,moving)
        return moving,flow,map_x, map_y

def fast_optical_flow_alignment(fixed_2048, moving_2048):
    """基于多尺度光流的快速对齐方案"""
    # 1. 降采样到512分辨率（保持宽高比）
    shape1=fixed_2048.shape[0]
    scale_factor = 2048 / shape1
    fixed_512 = cv2.resize(fixed_2048, (0, 0), fx=scale_factor, fy=scale_factor,
                           interpolation=cv2.INTER_AREA)
    moving_512 = cv2.resize(moving_2048, (0, 0), fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_AREA)

    # 2. 计算512分辨率的光流（快速计算）
    flow_512 = cv2.calcOpticalFlowFarneback(
        fixed_512, moving_512,
        None,
        pyr_scale=0.2,  # 更激进的金字塔缩放
        levels=1,  # 增加金字塔层数
        winsize=55,  # 减小窗口大小提速
        iterations=1,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # 3. 上采样光流到原尺寸（保持向量方向）
    flow_2048 = cv2.resize(flow_512, (shape1, shape1), interpolation=cv2.INTER_LINEAR)
    flow_2048 *= (shape1 / 2048)  # 缩放向量幅度

    # 4. 应用光流变形（高性能实现）
    h, w = fixed_2048.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow_2048[..., 0]).astype(np.float32)
    map_y = (y + flow_2048[..., 1]).astype(np.float32)

    aligned_2048 = cv2.remap(
        moving_2048, map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT101
    )

    return aligned_2048, flow_2048,map_x, map_y

def expand_to_2048_multiple(image):
    h, w = image.shape[:2]

    min_h = h
    min_w = w
    target_h = math.ceil(min_h / 2048) * 2048
    target_w = math.ceil(min_w / 2048) * 2048

    expanded_img = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype) if image.ndim == 3 else np.zeros((target_h, target_w), dtype=image.dtype)

    start_y = (target_h - h) // 2
    start_x = (target_w - w) // 2

    expanded_img[start_y:start_y + h, start_x:start_x + w] = image
    return expanded_img, start_x, start_y, h, w



def remove_padding_with_offsets(expanded_img, start_x, start_y, h, w):
    return expanded_img[start_y:start_y + h, start_x:start_x + w]


def pad_to_square(img, pad_value=0):
    """填充为正方形并返回填充后的图像及填充值"""
    h, w = img.shape[:2]
    size = max(h, w)

    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return img_padded, (pad_top, pad_bottom, pad_left, pad_right)
def unpad(img_padded, pads):
    """移除填充，恢复原始大小"""
    pad_top, pad_bottom, pad_left, pad_right = pads
    h, w = img_padded.shape[:2]
    return img_padded[pad_top:h-pad_bottom, pad_left:w-pad_right]

import tensorrt as trt
from multiprocessing import Pool
from functools import partial
import threading

def offset_image(img, offset_y, offset_x, fill_value=0):
    h, w = img.shape[:2]
    result = np.full_like(img, fill_value)

    # 计算源图像的有效区域
    src_y0 = max(0, -offset_y)
    src_y1 = min(h, h - offset_y)
    src_x0 = max(0, -offset_x)
    src_x1 = min(w, w - offset_x)

    # 目标区域
    dst_y0 = max(0, offset_y)
    dst_y1 = min(h, h + offset_y)
    dst_x0 = max(0, offset_x)
    dst_x1 = min(w, w + offset_x)

    # 填充到目标位置
    result[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return result


def paste_nonzero_gray(merged_image, moving_img_end, start_y, start_x):
    """
    将 moving_img_end 粘贴到 merged_image 的指定位置，只替换非0像素，不做边界过渡。

    参数：
        merged_image: 背景图 (灰度图)
        moving_img_end: 前景图 (灰度图)
        start_y, start_x: 放置位置
    返回：
        merged_image: 粘贴后的结果
    """
    h1, w1 = moving_img_end.shape
    end_y = start_y + h1
    end_x = start_x + w1

    # 只替换非0区域
    mask_nonzero = (moving_img_end != 0)
    merged_image[start_y:end_y, start_x:end_x][mask_nonzero] = moving_img_end[mask_nonzero]


    return merged_image
def detect_black_border_contours(img, thresh=0):
    """
    检测灰度图像中位于边缘的不规则黑色区域，返回mask
    """
    h, w = img.shape
    # 阈值分割，黑色区域=255
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)

    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img, dtype=np.uint8)

    for cnt in contours:
        # 判断轮廓是否与边界相交
        if np.any(cnt[:, 0, 0] == 0) or np.any(cnt[:, 0, 0] == w-1) or \
           np.any(cnt[:, 0, 1] == 0) or np.any(cnt[:, 0, 1] == h-1):
            cv2.drawContours(mask, [cnt], -1, 255, -1)  # 填充黑色区域

    return mask
def make_mask(expanded_moving_img, pads):
    h_exp, w_exp = expanded_moving_img.shape[:2]
    pad_top, pad_bottom, pad_left, pad_right = pads
    orig_h = h_exp - pad_top - pad_bottom
    orig_w = w_exp - pad_left - pad_right

    # 1. 构造原始图像区域 mask（原始区域=1，padding=0）
    orig_mask = np.zeros((h_exp, w_exp), dtype=np.uint8)
    orig_mask[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w] = 1

    # 3. warp 原始区域 mask（使用最近邻，保持 mask 离散）
    warped_mask = cv2.warpPerspective(
        orig_mask, H,
        (w_exp, h_exp),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    # warped_mask==1 表示有效像素，可以用于合并
    # 例如：
    # roi = merged_image[start_y:end_y, start_x:end_x]
    # roi[warped_mask==1] = moving_img_end[warped_mask==1]
    warped_mask = warped_mask * 255
    # warped_mask = cv2.bitwise_and(warped_mask, valid_mask_flow)
    warped_mask = unpad(warped_mask, pads)
    warped_mask = remove_padding_with_offsets(warped_mask, sx, sy, h, w)  # 4. 可选：保存调试
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 1像素收缩
    warped_mask = cv2.erode(warped_mask, kernel, iterations=2)  # iterations 可调 1-2
    return warped_mask
fixed_img = cv2.imread(r"D:\3d\temp_ceshi_name\cell1_2_['1_3_21.png', '1_4_21.png']_fixed.png", cv2.IMREAD_GRAYSCALE)
moving_img = cv2.imread(r"D:\3d\temp_ceshi_name\cell1_2_['1_3_21.png', '1_4_21.png']_moving.png", cv2.IMREAD_GRAYSCALE)
img3=cv2.imread(r"D:\3d\temp_ceshi_name\cell1_2_['1_3_21.png', '1_4_21.png']_moving_before.png", cv2.IMREAD_GRAYSCALE)
# 扩充到2048
expanded_fixed_img, sx, sy, h, w = expand_to_2048_multiple(fixed_img)
expanded_fixed_img, pads = pad_to_square(expanded_fixed_img, pad_value=0)
expanded_moving_img, _, _, _, _ = expand_to_2048_multiple(moving_img)
expanded_moving_img, _ = pad_to_square(expanded_moving_img, pad_value=0)

# 直方图匹配
expanded_fixed_img = histogram_matching(expanded_fixed_img, expanded_moving_img)

# 提取特征
extractor1 = TRTInference("aliked-n16_2048.trt", "aliked-n16", trt.Logger(trt.Logger.ERROR))
H, angle_deg, end_flag = Aliked_trt(expanded_fixed_img, expanded_moving_img, extractor1)

# warpPerspective
moving_img_end = cv2.warpPerspective(expanded_moving_img, H,
                                     (expanded_moving_img.shape[1], expanded_moving_img.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
s=time.time()
warped_mask=make_mask(moving_img_end, pads)
print(time.time()-s)
s=time.time()
moving_img_end, flow_,map_x, map_y = flow_cuda(expanded_fixed_img, moving_img_end, cpu=True)
print(time.time()-s)
# valid_mask_flow = (
#                           (map_x >= 0) & (map_x < expanded_moving_img.shape[1]) &
#                           (map_y >= 0) & (map_y < expanded_moving_img.shape[0])
#                   ).astype(np.uint8) * 255

moving_img_end = unpad(moving_img_end, pads)
moving_img_end = remove_padding_with_offsets(moving_img_end, sx, sy, h, w)
img3[warped_mask==255]=moving_img_end[warped_mask==255]
cv2.imwrite("img3.png", img3)
cv2.imwrite("moving_img_end.png", moving_img_end)
cv2.imwrite("warped_mask.png", warped_mask)
