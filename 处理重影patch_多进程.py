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
def group_connected_images(file_list: List[str]) -> List[List[str]]:
    """
    根据文件名中的坐标判断哪些图片是连通的（四邻域相邻）。
    返回分组后的文件名列表。
    """
    coords = {}
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)\.png')

    # 解析文件名并存储坐标
    for f in file_list:
        m = pattern.match(f)
        if m:
            _, y, x = map(int, m.groups())
            coords[(y, x)] = f

    visited = set()
    groups = []

    def get_neighbors(y, x):
        return [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]

    # DFS/BFS 搜索连通区域
    for c in coords:
        if c in visited:
            continue
        stack = [c]
        group = []
        while stack:
            pt = stack.pop()
            if pt in visited:
                continue
            visited.add(pt)
            group.append(coords[pt])
            for nb in get_neighbors(*pt):
                if nb in coords and nb not in visited:
                    stack.append(nb)
        groups.append(group)

    return groups
def save_result(result, filename):
    """保存结果到 .npz 文件"""
    np.savez_compressed(
        filename,
        # 标量和列表
        dir_path=result['dir_path'],
        first_dir=result['first_dir'],
        subdirs=result['subdirs'],
        out_path=result['out_path'],
        error_patchs=result['error_patchs'],
        # 大数组单独存储
        # merged_image=result['merged_image'],
        # # 字典转成item列表
        # merged_memory_keys=list(result['merged_memory'].keys()),
        # merged_memory_values=[v for v in result['merged_memory'].values()]
    )


def apply_flow_to_other_image(other_img, flow):
    """
    使用已计算的 flow（位移场）应用到另一张同尺寸图像上

    参数：
    - other_img: 要变换的图像（numpy.ndarray）
    - flow: 从 flow_cuda() 返回的 flow（形状 (H,W,2)，x/y 位移）

    返回：
    - warped_img: 应用光流后的图像
    """
    h, w = other_img.shape[:2]

    # 计算坐标映射
    y_coords, x_coords = np.indices((h, w))
    map_x = (x_coords + flow[..., 0]).astype(np.float32)  # x 方向位移
    map_y = (y_coords + flow[..., 1]).astype(np.float32)  # y 方向位移

    # 应用 remap（支持多通道图像，如 RGB）
    warped_img = cv2.remap(
        other_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped_img

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
        moving,flow=fast_optical_flow_alignment(fixed,moving)
        return moving,flow

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
        pyr_scale=0.5,  # 更激进的金字塔缩放
        levels=3,  # 增加金字塔层数
        winsize=25,  # 减小窗口大小提速
        iterations=1,
        poly_n=5,
        poly_sigma=1.1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
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

    return aligned_2048, flow_2048
def expand_to_2048_multiple(image):
    h, w = image.shape[:2]

    min_h = h + 400
    min_w = w + 400
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
# 线程工作函数(处理单个g分组)
def process_group(g, dir_path, first_dir, subdirs, HEIGHT, WIDTH,
                  fixed_merged_image, merged_image_ori,
                  merged_memory_ori, merged_memory,merged_image,extractor1):
    import pycuda.autoinit
    torch.cuda.empty_cache()


    try:
        # 原处理逻辑(你的循环体内容)
        coords = []
        for i in g:

            match = re.match(r'(\d+)_(\d+)_(\d+)\.(\w+)', i)
            if match:
                _, x, y, _ = match.groups()
                coords.append((int(x), int(y)))

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        start_y = max(0, min_y * 1648)
        start_x = max(0, min_x * 1648)
        end_y = min(HEIGHT, max_y * 1648 + 2048)
        end_x = min(WIDTH, max_x * 1648 + 2048)

        fixed_img = fixed_merged_image[start_y:end_y, start_x:end_x]
        # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{os.path.basename(dir_path)}_{g}_fixed_before.png', fixed_img)

        expanded_fixed_img, sx, sy, h, w =  expand_to_2048_multiple(fixed_img)
        target_h, target_w = expanded_fixed_img.shape[:2]
        moving_img = merged_image_ori[start_y:end_y, start_x:end_x]

        # 计算在 merged_image 中的新区域
        new_start_y = start_y - (target_h - h) // 2
        new_start_x = start_x - (target_w - w) // 2
        new_end_y = new_start_y + target_h
        new_end_x = new_start_x + target_w
        print(sx, sy, h, w)
        # 超出边界处理
        offset_y=0
        offset_x=0
        if new_start_y < 0:
            offset_y = -new_start_y
            new_start_y = 0

        if new_start_x < 0:
            offset_x = -new_start_x
            new_start_x = 0
        if new_end_y > HEIGHT:
            offset_y = new_end_y - HEIGHT
            new_end_y = HEIGHT

        if new_end_x > WIDTH:
            offset_x = new_end_x - WIDTH
            new_end_x = WIDTH
        # 根据更新后的区域裁剪
        expanded_fixed_img = fixed_merged_image[new_start_y:new_end_y, new_start_x:new_end_x]

        # 更新 sx, sy 偏移

        # print(sx, sy, h, w)
        expanded_fixed_img = fixed_merged_image[new_start_y:new_end_y, new_start_x:new_end_x]
        expanded_fixed_img, pads = pad_to_square(expanded_fixed_img, pad_value=0)
        # if expanded_fixed_img.shape[0]>8000:
        #     continue

        # expanded_moving_img, sx, sy, h, w = expand_to_2048_multiple(moving_img)

        expanded_moving_img = merged_image_ori[new_start_y:new_end_y, new_start_x:new_end_x]

        expanded_moving_img, pads = pad_to_square(expanded_moving_img, pad_value=0)

        print('模型开始')

        print('lightglue加载完成')
        # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_fixed_RIGHT.png', expanded_fixed_img)
        # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_moving_RIGHT.png', expanded_moving_img)
        expanded_fixed_img = histogram_matching(expanded_fixed_img, expanded_moving_img)
        # print('直方图匹配', time.time() - s)
        try:
            s1 = time.time()
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{os.path.basename(dir_path)}_{g}_fixed_before.png', expanded_fixed_img)
            H, angle_deg, end_flag = Aliked_trt(expanded_fixed_img, expanded_moving_img, extractor1)
            torch.cuda.empty_cache()

            print('模型计算', time.time() - s1)
            print('模型结束')
            # print('变换开始')

            moving_img_end = cv2.warpPerspective(expanded_moving_img, H,
                                                 (expanded_moving_img.shape[1], expanded_moving_img.shape[0]))
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{dir_path}_{g}_fixed_aliked.png', expanded_fixed_img)
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{os.path.basename(dir_path)}_{g}_moving_aliked.png', moving_img_end)
            print('变换结束')
            # print('光流之前')
            # torch.cuda.empty_cache()
            s2 = time.time()
            moving_img_end, flow_ = flow_cuda(expanded_fixed_img, moving_img_end, cpu=True)
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{os.path.basename(dir_path)}_{g}_moving_flow.png', moving_img_end)
            # print(expanded_fixed_img.shape)
            print('光流结束', time.time() - s2)
            moving_img_end = unpad(moving_img_end, pads)
            moving_img_end = remove_padding_with_offsets(moving_img_end, sx, sy, h, w)


            # 如果需要偏移
            if offset_x != 0 or offset_y != 0:
                moving_img_end = offset_image(moving_img_end, offset_y, offset_x, fill_value=0)
            merged_image[start_y:end_y, start_x:end_x] = moving_img_end
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{os.path.basename(dir_path)}_{g}_moving_flow.png', moving_img_end)

            print('dapi结束', time.time() - s1)
            torch.cuda.empty_cache()
            print(merged_memory_ori)
            for xxx in merged_memory_ori:

                moving_other_img = merged_memory_ori[xxx]
                # print(moving_other_img.shape)
                moving_other_img = moving_other_img[new_start_y:new_end_y, new_start_x:new_end_x]
                moving_other_img, pads = pad_to_square(moving_other_img, pad_value=0)
                # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_fixed.png', moving_other_img)
                moving_other_img = cv2.warpPerspective(moving_other_img, H,
                                                       (moving_other_img.shape[1], moving_other_img.shape[0]))
                # moving_other_img = cv2.remap(moving_other_img, flow_, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                #                            borderValue=0, dst=moving_other_img)
                # if flow_ is not None:
                moving_other_img = apply_flow_to_other_image(moving_other_img, flow_)
                moving_other_img = unpad(moving_other_img, pads)
                moving_other_img = remove_padding_with_offsets(moving_other_img, sx, sy, h, w)



                # 如果需要偏移
                if offset_x != 0 or offset_y != 0:
                    moving_other_img = offset_image(moving_other_img, offset_y, offset_x, fill_value=0)
                # moving_other_img = moving_other_img[offset_y0:target_h - offset_y1,
                #                          offset_x0:target_w - offset_x1]
                # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_before.png', merged_memory[xxx])
                merged_memory[xxx][start_y:end_y, start_x:end_x] = moving_other_img
                # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_after.png', merged_memory[xxx])
            print('运行成功')
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_fixed_RIGHT.png', expanded_fixed_img)
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_moving_RIGHT.png', moving_img_end)
            # try:
            #     print('[gpu光流失败]开始使用cpu光流')
            #     flow_ = cv2.calcOpticalFlowFarneback(
            #         expanded_fixed_img, moving_img_end, None,
            #         pyr_scale=0.2, levels=1, winsize=55,
            #         iterations=1, poly_n=5, poly_sigma=1.2,
            #         flags=0
            #     )
            #     h, w = flow_.shape[:2]
            #     x, y = np.meshgrid(np.arange(w), np.arange(h))
            #     map_x = x + flow_[..., 0]
            #     map_y = y + flow_[..., 1]
            #
            #     # 重映射图像
            #     moving_img_end = cv2.remap(
            #         moving_img_end,
            #         map_x.astype(np.float32),
            #         map_y.astype(np.float32),
            #         interpolation=cv2.INTER_LINEAR,
            #         borderMode=cv2.BORDER_REFLECT)
            # except   Exception as e:
            # print('[cpu光流失败]', e)

            # flow_ = None
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_fixed_ERROR.png', expanded_fixed_img)
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_moving_ERROR.png', moving_img_end)
            # print('[光流失败]', e)

            # print('光流之后')
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_fixed_ori.png', fixed_img)
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_ori.png', moving_img)

            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_moving.png', moving_other_img)
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}.png', moving_img_end)
            # print("img:", type(moving_img_end), moving_img_end.shape if moving_img_end is not None else None)
            # print("expanded:", type(moving_img_end))
            # print("sx, sy, h, w:", sx, sy, h, w)
        except Exception as e:
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_fixed_ori.png', expanded_fixed_img)
            # cv2.imwrite(fr'D:\3d\temp_ceshi_name\{g}_moving_ori.png', expanded_moving_img)
            # print( torch.cuda.memory_allocated() / 1024 / 1024, "MB used")

            # print("img:", type(moving_img_end), moving_img_end.shape if moving_img_end is not None else None)
            # print("expanded:", type(moving_img_end))
            # print("sx, sy, h, w:", sx, sy, h, w)
            print('[处理失败]', e)
            pass
        # ... 剩余处理逻辑不变 ...

    except Exception as e:
        print(f'处理分组{g}时出错:', e)
    finally:
        torch.cuda.empty_cache()
        if 'extractor1' in locals():  # 检查变量是否存在
            del extractor1
            import gc
            gc.collect()

# 进程工作函数(处理单个NPZ文件)
def process_single_file(npz_file,HEIGHT,WIDTH,fixed_merged_image):
    try:
        print(f"开始处理文件: {npz_file}")
        s = time.time()
        extractor1 = TRTInference("aliked-n16_2048.trt", "aliked-n16", trt.Logger(trt.Logger.ERROR))

        data = np.load(npz_file, allow_pickle=True)

        # 数据加载(与原代码一致)
        dir_path, first_dir, subdirs, out_path, error_patchs = data['dir_path'], data['first_dir'], data['subdirs'], \
        data['out_path'], data['error_patchs']
        dir_path = str(dir_path)
        first_dir = str(first_dir)
        subdirs = list(subdirs)
        out_path = str(out_path)
        error_paths = list(error_patchs)

        # 图像加载(与原代码一致)

        merged_image_ori = cv2.imread(
            os.path.join(dir_path, 'DAPI', f'{os.path.basename(dir_path)}_DAPI_merged_image_moving.png'),
            cv2.IMREAD_GRAYSCALE)

        # 分组(与原代码一致)
        result = group_connected_images(error_patchs)

        # 初始化字典(与原代码一致)
        merged_memory_ori = {}
        merged_image = cv2.imread(os.path.join(dir_path, 'DAPI', f'{os.path.basename(dir_path)}_DAPI_out1.png'),
                                  cv2.IMREAD_GRAYSCALE)
        merged_memory = {}

        # 加载子目录图像
        for subdir_path in subdirs:
            img = cv2.imread(os.path.join(subdir_path,
                                          f'{subdir_path.split(os.sep)[-2]}_{subdir_path.split(os.sep)[-1]}_merged_image_moving.png'),
                             cv2.IMREAD_GRAYSCALE)
            merged_memory_ori[subdir_path] = img
            img = cv2.imread(
                os.path.join(subdir_path, f'{subdir_path.split(os.sep)[-2]}_{subdir_path.split(os.sep)[-1]}_out1.png'),
                cv2.IMREAD_GRAYSCALE)
            merged_memory[subdir_path] = img

        # 多线程处理各分组(改为线程池更佳)
        # threads = []
        # for g in result:
        #     t = threading.Thread(
        #         target=process_group,
        #         args=(g, dir_path, first_dir, subdirs, HEIGHT, WIDTH,
        #               fixed_merged_image, merged_image_ori,
        #               merged_memory_ori, merged_memory,merged_image)
        #     )
        #     threads.append(t)
        #     t.start()
        #
        # for t in threads:
        #     t.join()
        for g in result:
            # if g != ['1_0_23.png', '1_1_23.png']:
            #     continue
            process_group(g, dir_path, first_dir, subdirs, HEIGHT, WIDTH,
                      fixed_merged_image, merged_image_ori,
                      merged_memory_ori, merged_memory,merged_image,extractor1)  # 直接串行处理
        # 保存结果(与原代码一致)
        output_path = f'{os.path.dirname(out_path)}/{dir_path.split(os.sep)[-1]}_DAPI_out1.png'
        cv2.imwrite(output_path, merged_image)
        print(f'[写图成功]', output_path)
        for i in merged_memory:
            try:
                output_path = f'{i}/{i.split(os.sep)[-2]}_{i.split(os.sep)[-1]}_out1.png'
                cv2.imwrite(output_path, merged_memory[i])
                print('[写图成功]', output_path)
            except Exception as e:
                print(f'[写图失败] {i}:', e)

        print(f'文件{npz_file}处理完成,耗时:{time.time() - s:.2f}秒')

    except Exception as e:
        print(f'处理文件{npz_file}时出错:', e)

def main(params):
    data_dir = params['output_dir']
    print(data_dir)
    # x = []
    x = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npz")
    ]
    # for i in os.listdir(data_dir):
    #     print(i)
    #     if i.endswith('.npz'):
    #         x.append(i)


    data = np.load(x[0], allow_pickle=True)

    # 数据加载(与原代码一致)
    dir_path, first_dir, subdirs, out_path, error_patchs = data['dir_path'], data['first_dir'], data['subdirs'], \
        data['out_path'], data['error_patchs']
    dir_path = str(dir_path)
    first_dir = str(first_dir)
    fixed_merged_image = cv2.imread(
        os.path.join(dir_path.replace(os.path.basename(dir_path), first_dir), 'DAPI', f'{first_dir}_DAPI_out1.png'),
        cv2.IMREAD_GRAYSCALE)
    HEIGHT, WIDTH = fixed_merged_image.shape[:2]
    # 多进程处理
    # with Pool(processes=len(x)) as pool:
    #     pool.map(process_single_file, x)
    with ProcessPoolExecutor(max_workers=len(x)) as executor:
        # with ProcessPoolExecutor(max_workers=1) as executor:

        executor.map(process_single_file, x, repeat(HEIGHT), repeat(WIDTH), repeat(fixed_merged_image))
    print("所有处理完成!")
if __name__ == '__main__':
   main()