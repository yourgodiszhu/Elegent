import onnxruntime as ort
import os
from logging import Manager
from trt_model_lightglue import TRTInference
from debugpy.server.cli import in_range
import tensorrt as trt
from src.loftr import LoFTR, default_cfg
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # 在程序开始时设置
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 添加到代码最开头
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # 启用设备端断言（Debug用）
import cv2
import shutil
import os
from itertools import repeat

# print(cv2.cuda.getCudaEnabledDeviceCount())

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
# os.add_dll_directory(r'C:\Program Files \NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
# os.add_dll_directory(r'D:\servicebio_deblurred\c12_p311\bin')
from PIL import Image, ImageOps
import shutil
import pycuda.autoinit
Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation
import gc
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED256, DoGHardNet, ALIKED1024
from lightglue.utils import load_image, rbd, load_image1
import numpy as np
import os
import time
import torch

torch.set_float32_matmul_precision('high')  # 或 'medium'/'highest'

# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
# os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
# os.add_dll_directory(r'D:\servicebio_deblurred\c12_p311\bin')
import cv2
import torch

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading


# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
def convert_lightglue_onnx_output(matches_sparse_np, scores_sparse_np, m, n, n_layers=4):
    """
    将 LightGlue ONNX 输出转换为 PyTorch 的 _forward() 输出格式的字典
    :param matches_sparse_np: numpy array [K, 2]
    :param scores_sparse_np: numpy array [K]
    :param m: image0 keypoints count
    :param n: image1 keypoints count
    :return: dict with 9 elements like PyTorch _forward
    """
    device = torch.device("cpu")  # 如果你有GPU，可以改成cuda

    # 转 numpy → torch
    matches_sparse = torch.from_numpy(matches_sparse_np).to(device).long()
    scores_sparse = torch.from_numpy(scores_sparse_np).to(device).float()

    # 稠密匹配 [1, M] 和 [1, N]
    matches0 = torch.full((1, m), -1, dtype=torch.long, device=device)
    matching_scores0 = torch.zeros((1, m), dtype=torch.float32, device=device)

    matches1 = torch.full((1, n), -1, dtype=torch.long, device=device)
    matching_scores1 = torch.zeros((1, n), dtype=torch.float32, device=device)

    if matches_sparse.numel() > 0:
        idx0 = matches_sparse[:, 0]
        idx1 = matches_sparse[:, 1]

        matches0[0, idx0] = idx1
        matching_scores0[0, idx0] = scores_sparse

        matches1[0, idx1] = idx0
        matching_scores1[0, idx1] = scores_sparse

    # stop：ONNX 没有早停机制，这里给个默认值
    stop = n_layers

    # matches [1, K, 2]
    matches = matches_sparse.unsqueeze(0)  # shape [1, K, 2]
    scores = scores_sparse.unsqueeze(0)  # shape [1, K]

    # pruning：默认没裁剪，全部标记为 n_layers
    prune0 = torch.full((1, m), n_layers, dtype=torch.float32, device=device)
    prune1 = torch.full((1, n), n_layers, dtype=torch.float32, device=device)

    return {
        "matches0": matches0,
        "matches1": matches1,
        "matching_scores0": matching_scores0,
        "matching_scores1": matching_scores1,
        "stop": stop,
        "matches": matches,
        "scores": scores,
        "prune0": prune0,
        "prune1": prune1,
    }

# clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
def load_image_as_tensor(path, device):
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(path, (512, 512))
    tensor = torch.from_numpy(img).float().half()[None][None] / 255.0
    return tensor.to(device)
# def load_image_as_tensor(image_path,device):
#     # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     tensor = torch.from_numpy(image_path).float()[None][None].half().cuda()  # 转为FP16
#     return (tensor / 255.0).to(device)  # 归一化
def Loftr_pt(
        # input_path=r"image/2448",
        #
        # device="cuda",
        # top_k=3000,
        # scores_th=0.2,
        # n_limit=20000,
        fixed_path, moving_path, device, matcher
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
    s=time.time()
    img0 = load_image_as_tensor(fixed_path, device)
    img1 =  load_image_as_tensor(moving_path, device)
    batch = {'image0': img0, 'image1': img1}
    print('数据预处理 时间:',time.time()-s)
    s=time.time()
    matcher = matcher.eval().cuda()
    default_cfg['coarse']

    print('模型加载时间:',time.time()-s)
    # print(img_rgb_ref.min(), img_rgb_ref.max())
    # print(img_rgb_ref.is_contiguous())
    # print(batch)
    # print('这里没有问题1')
    s=time.time()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        matcher(batch)  # 异步执行
    stream.synchronize()  # 显式同步（确保结果就绪）
    print('模型推理时间:',time.time()-s)
    # if xx is None:
    #     return None,None
    # print(xx)
    # print('这里没有问题2')
    s=time.time()
    mkpts0 = batch['mkpts0_f'].cpu().numpy() * 4
    mkpts1 = batch['mkpts1_f'].cpu().numpy() * 4
    # mconf = batch['mconf'].cpu().numpy()
    H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.USAC_DEFAULT)
    print('计算变换矩阵时间:',time.time()-s)
    # print(f"RANSAC 结果 | 内点: {sum(mask)} | 误匹配点: {len(mask) - sum(mask)}")
    # print(sum(mask).count())
    # if  int(sum(mask))<20:
    #     # print('1111')
    #     time_name=int(time.time())
    #     cv2.imwrite(fr'D:\3d\test/{time_name}_{int(sum(mask))}_1.png',fixed_path)
    #     cv2.imwrite(fr'D:\3d\test/{time_name}_{int(sum(mask))}_2.png',moving_path)

    # 方法一：单应性矩阵（适用于平面透视变换）
    # 可视化匹配点（区分内点和误匹配点）
    # plt.figure(figsize=(16, 8))
    # plt.imshow(np.hstack([img_rgb_ref,
    #                       img_rgb]))
    #
    # # 绘制关键点
    # plt.scatter(kpts0[:, 0], kpts0[:, 1], c='blue', s=10, alpha=0.6, label='Fixed Keypoints')
    # plt.scatter(kpts1[:, 0] + offset, kpts1[:, 1], c='orange', s=10, alpha=0.6, label='Moving Keypoints')
    # # 绘制匹配连线（绿色 = 内点，红色 = 误匹配点）
    # for i, (idx0, idx1) in enumerate(matches[:1000]):  # 仅显示前1000对
    #     x0, y0 = kpts0[idx0]
    #     x1, y1 = kpts1[idx1]
    #     if mask[i] == 1:  # 内点（绿色）
    #         plt.plot([x0, x1 + offset], [y0, y1], '-', color='green', linewidth=1, alpha=0.3,
    #                  label='Inlier' if i == 0 else "")
    #     else:  # 误匹配点（红色）
    #         plt.plot([x0, x1 + img_rgb_ref.shape[2]], [y0, y1], '-', color='red', linewidth=1, alpha=0.1,
    #                  label='Outlier' if i == 0 else "")
    # plt.legend()
    # plt.title(f"RANSAC 结果 | 内点: {sum(mask)} | 误匹配点: {len(mask) - sum(mask)}")
    # plt.tight_layout()
    # plt.show()

    # tx = H[0, 2]  # 水平偏移（左右）
    # ty = H[1, 2]  # 垂直偏移（上下）
    # shift = [int(ty * 40), int(tx * 40)]
    # print(shift)
    # print(H)
    # angle_deg = 0
    # angle_rad = np.arctan2(H[1, 0], H[0, 0])  # atan2(b, a)
    # angle_deg = np.degrees(angle_rad)
    # s=time.time()
    result = evaluate_homography(
        H,
        # matches=matches,
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
    return H, result['valid']


def auto_register(fixed_path, moving_path, device, extractor, matcher):
    # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
    # extractor = DISK(max_num_keypoints=256).eval().cuda()  # load the extractor
    # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    # print('进入函数')
    # print(fixed_path.shape)
    # fixed_path=np.array(get_color_img(fixed_path,'blue'))
    # moving_path=np.array(get_color_img(moving_path,'blue'))
    # fixed_path = clahe.apply(cv2.cvtColor(fixed_path, cv2.COLOR_BGR2GRAY))
    # moving_path = clahe.apply(cv2.cvtColor(moving_path, cv2.COLOR_BGR2GRAY))
    # fixed_path=cv2.cvtColor(fixed_path, cv2.COLOR_GRAY2BGR)
    # moving_path=cv2.cvtColor(moving_path, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('')
    # fixed_path = load_image1(fixed_path, resize=(fixed_path.shape[0] // 2, fixed_path.shape[1] // 2))
    # moving_path = load_image1(moving_path, resize=(moving_path.shape[0] // 2, moving_path.shape[1] // 2))
    fixed = load_image(fixed_path)
    fixed = fixed.to(device)
    # print('加载图片1')
    moving = load_image(moving_path)
    moving = moving.to(device)
    s = time.time()
    # print('加载图片2')

    # extract local features
    feats0 = extractor.extract(fixed)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(moving)
    # print('时间', (time.time() - s) * 1000)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    # print('时间', (time.time() - s) * 1000)

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    # print('时间', (time.time() - s) * 1000)

    pts0 = points0.cpu().numpy().astype(np.float32)  # 匹配点来自你的代码
    pts1 = points1.cpu().numpy().astype(np.float32)

    # 方法一：单应性矩阵（适用于平面透视变换）
    H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_DEFAULT)
    # if  int(sum(mask))<20:
    #     # print('1111')
    #     time_name=int(time.time())
    #     cv2.imwrite(fr'D:\3d\test/{time_name}_{int(sum(mask))}_1.png',fixed_path)
    #     cv2.imwrite(fr'D:\3d\test/{time_name}_{int(sum(mask))}_2.png',moving_path)
    # tx = H[0, 2]  # 水平偏移（左右）
    # ty = H[1, 2]  # 垂直偏移（上下）
    # shift = [int(ty * 40), int(tx * 40)]
    # print(shift)
    # print(H)
    angle_deg = 0
    # angle_rad = np.arctan2(H[1, 0], H[0, 0])  # atan2(b, a)
    # angle_deg = np.degrees(angle_rad)
    # s=time.time()
    result = evaluate_homography(
        H,
        matches=matches,
        img_shape=fixed.shape[1:],  # (h, w)
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
    return H, angle_deg, result['valid']


def auto_register1(fixed_path, moving_path, device, extractor, matcher):
    # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
    # extractor = DISK(max_num_keypoints=256).eval().cuda()  # load the extractor
    # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    # print('进入函数')
    # print(fixed_path.shape)
    # fixed_path=np.array(get_color_img(fixed_path,'blue'))
    # moving_path=np.array(get_color_img(moving_path,'blue'))
    # fixed_path = clahe.apply(cv2.cvtColor(fixed_path, cv2.COLOR_BGR2GRAY))
    # moving_path = clahe.apply(cv2.cvtColor(moving_path, cv2.COLOR_BGR2GRAY))
    # fixed_path=cv2.cvtColor(fixed_path, cv2.COLOR_GRAY2BGR)
    # moving_path=cv2.cvtColor(moving_path, cv2.COLOR_GRAY2BGR)

    fixed = load_image(fixed_path)
    fixed = fixed.to(device)
    # print('加载图片1')
    moving = load_image(moving_path)
    moving = moving.to(device)
    s = time.time()
    # print('加载图片2')

    # extract local features
    feats0 = extractor.extract(fixed)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(moving)
    # print('时间', (time.time() - s) * 1000)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    # print('时间', (time.time() - s) * 1000)

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    # print('时间', (time.time() - s) * 1000)

    pts0 = points0.cpu().numpy().astype(np.float32)  # 匹配点来自你的代码
    pts1 = points1.cpu().numpy().astype(np.float32)

    # 方法一：单应性矩阵（适用于平面透视变换）
    # H, _ = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)
    H, mask = cv2.findHomography(
        pts1, pts0,
        method=cv2.USAC_ACCURATE
    )
    # tx = H[0, 2]  # 水平偏移（左右）
    # ty = H[1, 2]  # 垂直偏移（上下）
    # shift = [int(ty * 40), int(tx * 40)]
    # print(shift)
    # print(H)
    angle_deg = 0
    # angle_rad = np.arctan2(H[1, 0], H[0, 0])  # atan2(b, a)
    # angle_deg = np.degrees(angle_rad)
    # s=time.time()
    result = evaluate_homography(
        H,
        matches=matches,
        img_shape=fixed.shape[1:],  # (h, w)
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
    return H, angle_deg, result['valid']


def get_color_img(img, color):
    res_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(img.shape[2]):
        color_img = Image.fromarray(img[:, :, i])
        color_img = ImageOps.colorize(color_img, black='black', white=color)
        res_img = np.clip(res_img + np.array(color_img), 0, 255)
    res_img = Image.fromarray(res_img)
    return res_img


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


#
def merge_with_offset(moving, shift):
    """创建一个背景图像并将移动图像粘贴到相应位置"""
    # 获取fixed图像的大小
    h, w = moving.shape[:2]

    # 创建与fixed同样大小的背景图像，初始化为零（黑色背景）
    result = np.zeros_like(moving)

    # 将moving图像粘贴到result图像中，偏移后进行粘贴
    y_offset, x_offset = shift

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

    # 获取要粘贴的部分
    moving_part = moving[start_y - y_offset:end_y - y_offset, start_x - x_offset:end_x - x_offset]

    # 粘贴到背景图像相应位置
    result[start_y:end_y, start_x:end_x] = moving_part
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # 转换为 PIL 图像
    result = Image.fromarray(result)
    return result


# def process_subdir(subdir_path, H_dict, HEIGHT, WIDTH,merged_memory,first_dir):
#     """处理单个子目录的任务函数（在每个线程中执行）"""
#     # black_image = np.zeros((2048, 2048), dtype=np.uint8)
#     black_image = np.full((2048, 2048), 0, dtype=np.uint8)
#
#     # merged_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
#     # print(merged_memory)
#     if subdir_path in merged_memory:
#         merged_image = merged_memory[subdir_path]
#     else:
#         # merged_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
#         merged_image = np.full((HEIGHT, WIDTH), dtype=np.uint8, fill_value=0)
#
#     print(f"线程 {os.getpid()} - {threading.current_thread().name} 正在处理子目录: {subdir_path}")
#     out_path = ''
#
#     # crop_moving_dir = os.path.join(subdir_path, 'crop_moving_img')
#     count=0
#     # is_black=False
#     H_global=None
#     for i in H_dict:
#
#         moving_img = cv2.imread(os.path.join(subdir_path, 'crop_moving_img', i), cv2.IMREAD_GRAYSCALE)
#
#         if moving_img.max() < 10:
#             is_black = True
#         else:
#             is_black = False
#
#         if not is_black:
#         # print(H_dict)
#         # print(H_dict[i])
#             if H_dict[i] is None:
#                 H=H_global
#                 # print('H_dict[i] is None')
#                 fixed_img = cv2.imread(os.path.join(subdir_path, 'crop_moving_img', i).replace(
#                     os.path.basename(os.path.dirname(subdir_path)), first_dir), cv2.IMREAD_GRAYSCALE)
#                 print(fixed_img.shape,moving_img.shape)
#                 registered_color = cv2.warpPerspective(moving_img, H, (moving_img.shape[1], moving_img.shape[0]))
#                 new_coords_list = []
#
#                 h, w = moving_img.shape
#                 stream = cv2.cuda_Stream()
#                 gpu_gray_fixed = cv2.cuda_GpuMat()
#                 gpu_gray_registered = cv2.cuda_GpuMat()
#                 gpu_gray_fixed.upload(fixed_img, stream)
#                 gpu_gray_registered.upload(registered_color, stream)
#
#                 farneback1 = cv2.cuda_FarnebackOpticalFlow.create(
#                     numLevels=1, pyrScale=0.2, fastPyramids=False, winSize=55,
#                     numIters=1, polyN=5, polySigma=1.2, flags=0
#                 )
#                 gpu_flow = farneback1.calc(gpu_gray_fixed, gpu_gray_registered, None, stream)
#                 stream.waitForCompletion()
#                 flow = gpu_flow.download()
#
#                 grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
#                 map_x = (grid_x + flow[..., 0]).astype(np.float32)
#                 map_y = (grid_y + flow[..., 1]).astype(np.float32)
#                 new_coords_list.append((map_x, map_y))
#
#                 gpu_registered_color = cv2.cuda_GpuMat()
#                 gpu_registered_color.upload(registered_color, stream)
#                 gpu_map_x = cv2.cuda_GpuMat()
#                 gpu_map_y = cv2.cuda_GpuMat()
#                 gpu_map_x.upload(map_x, stream)
#                 gpu_map_y.upload(map_y, stream)
#
#                 gpu_corrected = cv2.cuda.remap(
#                     gpu_registered_color, gpu_map_x, gpu_map_y,
#                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, stream=stream
#                 )
#                 stream.waitForCompletion()
#                 registered_color = gpu_corrected.download()
#
#                 # registered_color = black_image
#             else:
#                 H_global=H_dict[i][0]
#                 registered_color = cv2.warpPerspective(moving_img, H_dict[i][0], (moving_img.shape[1], moving_img.shape[0]))
#                 for j in range(1):
#                     registered_color = cv2.remap(registered_color, H_dict[i][1][j][0], H_dict[i][1][j][1], cv2.INTER_LINEAR)
#         else:
#             registered_color = black_image
#
#             # registered_color = cv2.warpPerspective(registered_color, H_dict[i][1], (moving_img.shape[1], moving_img.shape[0]))
#
#         # if out_path == '':
#         #     # print(i)
#         #     # print(subdir_path)
#         #     # print(i.replace(i.split("/")[1],subdir_path.split("/")[-2])).replace('crop_moving_img','crop_ending_img')
#         #     out_path = os.path.join(subdir_path, 'crop_ending_img')
#         #     os.makedirs(out_path, exist_ok=True)
#             # print('保存路径',out_path)
#         # print(i.replace(i.split("/")[1],subdir_path.split("/")[-2]).replace('crop_moving_img','crop_ending_img'))
#         # match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
#         # if match:
#         #     y, x = int(match.group(1)), int(match.group(2))
#         #     patch_dict[(y, x)] = registered_color  # 存入内存
#
#         match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
#         if match:
#             # print('match成功')
#             patch_y, patch_x = int(match.group(1)), int(match.group(2))
#             patch = registered_color[:2048, :2048]
#             # print('patch_y', patch_y, 'patch_x', patch_x)
#
#             # 计算实际覆盖区域
#             end_y = min(patch_y + patch.shape[0], HEIGHT)
#             end_x = min(patch_x + patch.shape[1], WIDTH)
#             actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
#             # print('merged_image', merged_image[patch_y:end_y, patch_x:end_x].shape)
#             # print('actual_patch', actual_patch.shape)
#             # 使用最大值合并策略
#             # merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
#             #     merged_image[patch_y:end_y, patch_x:end_x],
#             #     actual_patch
#             # )
#             merged_image[patch_y:end_y, patch_x:end_x]=actual_patch
#             # region = merged_image[patch_y:end_y, patch_x:end_x]
#             # mask = region == 0  # 找出非0区域
#             # region[mask] = actual_patch[mask]
#             # print(f'merged_image合并成功{count}')
#             # count+=1
#             # cv2.imwrite(f'test/{time.time()}.png',merged_image)
#     # merged_memory[subdir_path] = merged_image
#     # print('开始存储')
#         # 可选：保存每张变换后图（如果需要）
#         # cv2.imwrite(os.path.join(out_path, i), registered_color)
#     merged_memory[subdir_path] = merged_image
#     # print('线程存储成功')
#     # 保存本子目录的最终图像
#     # try:
#     #     # output_path = f'{subdir_path}/{subdir_path.split(os.sep)[-2]}_{subdir_path.split(os.sep)[-1]}_out1.png'
#     #     merged_memory[subdir_path] = merged_image
#     #
#     #     # cv2.imwrite(output_path, merged_image)
#     # except Exception as e:
#     #     print('[线程写图失败]', e)



def process_subdir(subdir_path, H_dict, HEIGHT, WIDTH, merged_memory, first_dir):
    black_image = np.full((2048, 2048), 0, dtype=np.uint8)

    if subdir_path in merged_memory:
        merged_image = merged_memory[subdir_path]
    else:
        merged_image = np.full((HEIGHT, WIDTH), dtype=np.uint8, fill_value=0)

    print(f"线程 {os.getpid()} - {threading.current_thread().name} 正在处理子目录: {subdir_path}")

    for i in H_dict:
        # print(os.path.join(subdir_path, 'crop_moving_img', i).replace(
        #     os.path.basename(os.path.dirname(subdir_path)), first_dir))
        # fixed_img = cv2.imread(os.path.join(subdir_path, 'crop_moving_img', i).replace(
        #     os.path.basename(os.path.dirname(subdir_path)), first_dir), cv2.IMREAD_GRAYSCALE)
        # print(fixed_img.shape)
        moving_img = cv2.imread(os.path.join(subdir_path, 'crop_moving_img', i), cv2.IMREAD_GRAYSCALE)
        # print(H_dict)
        # print(H_dict[i])
        if H_dict[i] is None:
            # print('H_dict[i] is None')
            registered_color =black_image
        else:
            # if len(H_dict[i][1])==2:
            #     registered_color = cv2.warpPerspective(moving_img, H_dict[i][0],
            #                                            (moving_img.shape[1], moving_img.shape[0]))
            #     for j in range(2):
            #         registered_color = cv2.remap(registered_color, H_dict[i][1][j][0], H_dict[i][1][j][1],
            #                                      cv2.INTER_LINEAR)
            # else:
            registered_color = cv2.warpPerspective(moving_img, H_dict[i][0], (moving_img.shape[1], moving_img.shape[0]))
            registered_color = cv2.remap(registered_color, H_dict[i][1][0], H_dict[i][1][1],
                                         cv2.INTER_LINEAR)
                # registered_color = cv2.warpPerspective(registered_color, H_dict[i][1], (moving_img.shape[1], moving_img.shape[0]))
            # registered_color = cv2.warpPerspective(registered_color, H_dict[i][1], (moving_img.shape[1], moving_img.shape[0]))

        # if out_path == '':
        #     # print(i)
        #     # print(subdir_path)
        #     # print(i.replace(i.split("/")[1],subdir_path.split("/")[-2])).replace('crop_moving_img','crop_ending_img')
        #     out_path = os.path.join(subdir_path, 'crop_ending_img')
        #     os.makedirs(out_path, exist_ok=True)
        # print('保存路径',out_path)
        # print(i.replace(i.split("/")[1],subdir_path.split("/")[-2]).replace('crop_moving_img','crop_ending_img'))
        # match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
        # if match:
        #     y, x = int(match.group(1)), int(match.group(2))
        #     patch_dict[(y, x)] = registered_color  # 存入内存

        match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
        if match:
            # print('match成功')
            patch_y, patch_x = int(match.group(1)), int(match.group(2))
            patch = registered_color[:2048, :2048]
            # print('patch_y', patch_y, 'patch_x', patch_x)

            # 计算实际覆盖区域
            end_y = min(patch_y + patch.shape[0], HEIGHT)
            end_x = min(patch_x + patch.shape[1], WIDTH)
            actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
            # print('merged_image', merged_image[patch_y:end_y, patch_x:end_x].shape)
            # print('actual_patch', actual_patch.shape)
            # 使用最大值合并策略
            merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
                merged_image[patch_y:end_y, patch_x:end_x],
                actual_patch
            )
            # merged_image[patch_y:end_y, patch_x:end_x] = actual_patch
            # region = merged_image[patch_y:end_y, patch_x:end_x]
            # mask = region == 0  # 找出非0区域
            # region[mask] = actual_patch[mask]
            # print(f'merged_image合并成功{count}')
            # count+=1
            # cv2.imwrite(f'test/{time.time()}.png',merged_image)
    # merged_memory[subdir_path] = merged_image
    # print('开始存储')
    # 可选：保存每张变换后图（如果需要）
    # cv2.imwrite(os.path.join(out_path, i), registered_color)
    merged_memory[subdir_path] = merged_image
    print('线程结束没有问题')
from typing import Tuple, List
import re
import tqdm


# def merge_patches_from_folder(
#         folder_path: str,
#         window_size: Tuple[int, int],
#         overlap: Tuple[int, int],
#         original_size: Tuple[int, int],
# ) -> np.ndarray:
#     """合并文件夹中的图像碎片到完整图像
#
#     Args:
#         folder_path: 包含图像碎片的文件夹路径
#         window_size: 原始碎片大小 (height, width)
#         overlap: 重叠区域大小 (height, width)
#         original_size: 目标图像尺寸 (height, width)
#
#     Returns:
#         合并后的numpy数组 (H,W,C)
#     """
#     height, width = original_size
#     channels = 3  # 假设是RGB图像
#     merged_image = np.zeros((height, width, channels), dtype=np.uint8)
#     patch_files = []
#     # print(f"Processing folder: {folder_path}")
#     # Step 1: 扫描并排序所有有效碎片文件
#     for filename in os.listdir(folder_path):
#         # 使用正则匹配文件名中的坐标 (假设格式如 patch_0123_y1000_x2000.png)
#         match = re.search(r'patch+_y(\d+)_x(\d+)\.', filename)
#         # print(filename)
#         if match:
#             y, x = int(match.group(1)), int(match.group(2))
#             patch_files.append((y, x, filename))
#             # print(y,x,filename)
#     # 按y,x坐标排序确保正确叠加顺序
#     patch_files.sort(key=lambda x: (x[0], x[1]))
#     # print(patch_files)
#     # Step 2: 逐个加载并合并碎片
#     for y, x, filename in tqdm.tqdm(patch_files, desc=f"Merging {os.path.basename(folder_path)}"):
#         if x >= width or y >= height:  # 跳过超出边界的坐标
#             continue
#
#         # 加载图像碎片
#         patch_path = os.path.join(folder_path, filename)
#         patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
#
#         if patch is None:
#             print(f"Warning: Failed to load {filename}")
#             continue
#
#         # 确保patch是3通道 (处理可能的灰度图)
#         if len(patch.shape) == 2:
#             patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
#
#         # 自动适配窗口尺寸 (处理不规则的最后一个碎片)
#         patch = patch.astype(np.uint8)
#         if patch.shape[0] > window_size[0] or patch.shape[1] > window_size[1]:
#             patch = patch[:window_size[0], :window_size[1]]
#
#         # 计算实际覆盖区域 (处理图像边界)
#         end_y = min(y + patch.shape[0], height)
#         end_x = min(x + patch.shape[1], width)
#         actual_patch = patch[:end_y - y, :end_x - x]
#
#         # 使用最大值合并策略（避免重叠区域变暗）
#         merged_image[y:end_y, x:end_x] = np.maximum(
#             merged_image[y:end_y, x:end_x],
#             actual_patch
#         )
#
#     # 转换回8位图像
#     return np.clip(merged_image, 0, 255).astype(np.uint8)


def merge_patches_from_memory(
        patch_dict: dict,  # 格式: {(y, x): image_array}
        window_size: Tuple[int, int],
        overlap: Tuple[int, int],
        original_size: Tuple[int, int],
) -> np.ndarray:
    """直接从内存合并碎片"""
    height, width = original_size
    channels = 3  # 假设RGB图像
    merged_image = np.zeros((height, width, channels), dtype=np.uint8)

    # 按坐标排序
    sorted_patches = sorted(patch_dict.items(), key=lambda item: (item[0][0], item[0][1]))

    for (y, x), patch in sorted_patches:
        if x >= width or y >= height:  # 跳过越界坐标
            continue

        # 自动适配窗口尺寸
        patch = patch[:window_size[0], :window_size[1]]

        # 计算实际覆盖区域
        end_y = min(y + patch.shape[0], height)
        end_x = min(x + patch.shape[1], width)
        actual_patch = patch[:end_y - y, :end_x - x]

        # 使用最大值合并策略
        merged_image[y:end_y, x:end_x] = np.maximum(
            merged_image[y:end_y, x:end_x],
            actual_patch
        )

    return merged_image.astype(np.uint8)


def merge_patches_from_memory_gray(
        patch_dict: dict,  # 格式: {(y, x): image_array}
        window_size: Tuple[int, int],
        overlap: Tuple[int, int],
        original_size: Tuple[int, int],
) -> np.ndarray:
    """直接从内存合并碎片"""
    height, width = original_size
    # channels = 3  # 假设RGB图像
    merged_image = np.zeros((height, width), dtype=np.uint8)

    # 按坐标排序
    sorted_patches = sorted(patch_dict.items(), key=lambda item: (item[0][0], item[0][1]))

    for (y, x), patch in sorted_patches:
        if x >= width or y >= height:  # 跳过越界坐标
            continue

        # 自动适配窗口尺寸
        patch = patch[:window_size[0], :window_size[1]]

        # 计算实际覆盖区域
        end_y = min(y + patch.shape[0], height)
        end_x = min(x + patch.shape[1], width)
        actual_patch = patch[:end_y - y, :end_x - x]

        # 使用最大值合并策略
        merged_image[y:end_y, x:end_x] = np.maximum(
            merged_image[y:end_y, x:end_x],
            actual_patch
        )

    return merged_image.astype(np.uint8)


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


# def process_directory(dir_path, images_path, HEIGHT, WIDTH,first_dir):
#     # 声明使用全局变量
#
#     """在进程中处理目录的函数（每个进程处理一个目录）"""
#     print(f"进程 {os.getpid()} 正在处理目录: {dir_path}")
#     # SuperPoint+LightGlue
#     # print(fixed_name)
#     # print(images_path)
#     device = 'cuda'  # 也可根据需求选择其他GPU（如cuda:1/2/3）
#     # extractor = ALIKED256().eval().to(device)  # load the extractor
#     # trt_logger = LOGGER_DICT["verbose"]
#     # trt_model_path = "aliked-n16512.trt",
#     # model_name = "aliked-n16",
#     extractor = TRTInference("aliked-n16512.trt", "aliked-n16", trt.Logger(trt.Logger.ERROR))
#
#     matcher_pt = LightGlue(features='aliked',filter_threshold=0.05).eval().to(device)  # load the matcher
#     # sess_opts = ort.SessionOptions()
#     # providers = (
#     #     ["CUDAExecutionProvider"]
#     #     if device == "cuda"
#     #     else ["CPUExecutionProvider"]
#     # )
#
#     # lightglue_path = fr"D:\LightGlue-ONNX-aliked_support\weights\aliked_lightglue.onnx"
#     #
#     # lightglue = ort.InferenceSession(
#     #     lightglue_path,
#     #     sess_options=sess_opts,
#     #     providers=providers,
#     # )
#     # matcher_trt = lightglue
#     trt_flag = False
#     if trt_flag:
#         matcher = matcher_pt
#     else:
#         matcher = matcher_pt
#
#     # matcher.compile(mode='reduce-overhead')
#     # extractor1 = ALIKED1024(max_num_keypoints=256).eval().to(device)  # load the extractor
#     # matcher1 = LightGlue(features='aliked').eval().to(device)  # load the matcher
#
#     # 获取该目录下的所有子目录
#     subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path)
#                if os.path.isdir(os.path.join(dir_path, d)) and d != 'DAPI']
#     import copy
#     from concurrent.futures import ThreadPoolExecutor
#     from itertools import repeat
#
#     # 初始化
#     H_dict = {}
#     merged_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
#     merged_image=np.full((HEIGHT,WIDTH),dtype=np.uint8,fill_value=127)
#     error_img = np.full((2048, 2048), 0, dtype=np.uint8)
#     white_img = np.full((2048, 2048), 0, dtype=np.uint8)
#     merged_memory = {}  # key: subdir_path, value: merged_image（np.ndarray）
#
#     # 计算切分点
#     total = len(images_path)
#     chunk_size = 10
#     chunk_index = 0
#     out_path=''
#     # 用于记录所有线程的句柄
#     all_threads = []
#     # print(1)
#     # 定义异步处理函数
#     def process_subdirs_with_dict(subdirs, H_dict_chunk, HEIGHT, WIDTH,merged_memory):
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             executor.map(process_subdir, subdirs, repeat(H_dict_chunk), repeat(HEIGHT), repeat(WIDTH),repeat(merged_memory),repeat(first_dir))
#     H_global=None
#     # 主处理循环
#     for idx, i in enumerate(tqdm.tqdm(images_path)):
#         # print(2)
#
#         fixed_img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
#         moving_img = cv2.imread(i.replace(i.split("\\")[1], dir_path.split("\\")[-1]), cv2.IMREAD_GRAYSCALE)
#         if moving_img.max() <10:
#             is_black =True
#         else:
#             is_black = False
#         if out_path == '':
#             out_path = os.path.dirname(
#                 i.replace(i.split("\\")[1], dir_path.split("\\")[-1]).replace('crop_moving_img', 'crop_ending_img'))
#             out_path1 = os.path.dirname(i.replace(i.split("\\")[1], dir_path.split("\\")[-1]).replace('crop_moving_img',
#                                                                                                       'crop_ending_img_aliked'))
#             os.makedirs(out_path1, exist_ok=True)
#
#             os.makedirs(out_path, exist_ok=True)
#
#         # print(3)
#         if not  is_black:
#             try:
#                 H, angle, flag = Aliked_trt(fixed_img, moving_img, device, extractor, matcher=matcher, trt_flag=False)
#                 clear_memory()
#                 if flag:
#                     H_global = H.copy()
#
#                     registered_color = cv2.warpPerspective(moving_img, H, (fixed_img.shape[1], fixed_img.shape[0]))
#                     new_coords_list = []
#
#                     h, w = fixed_img.shape
#                     stream = cv2.cuda_Stream()
#                     gpu_gray_fixed = cv2.cuda_GpuMat()
#                     gpu_gray_registered = cv2.cuda_GpuMat()
#                     gpu_gray_fixed.upload(fixed_img, stream)
#                     gpu_gray_registered.upload(registered_color, stream)
#
#                     farneback1 = cv2.cuda_FarnebackOpticalFlow.create(
#                         numLevels=1, pyrScale=0.2, fastPyramids=False, winSize=55,
#                         numIters=1, polyN=5, polySigma=1.2, flags=0
#                     )
#                     gpu_flow = farneback1.calc(gpu_gray_fixed, gpu_gray_registered, None, stream)
#                     stream.waitForCompletion()
#                     flow = gpu_flow.download()
#
#                     grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
#                     map_x = (grid_x + flow[..., 0]).astype(np.float32)
#                     map_y = (grid_y + flow[..., 1]).astype(np.float32)
#                     new_coords_list.append((map_x, map_y))
#
#                     gpu_registered_color = cv2.cuda_GpuMat()
#                     gpu_registered_color.upload(registered_color, stream)
#                     gpu_map_x = cv2.cuda_GpuMat()
#                     gpu_map_y = cv2.cuda_GpuMat()
#                     gpu_map_x.upload(map_x, stream)
#                     gpu_map_y.upload(map_y, stream)
#
#                     gpu_corrected = cv2.cuda.remap(
#                         gpu_registered_color, gpu_map_x, gpu_map_y,
#                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, stream=stream
#                     )
#                     stream.waitForCompletion()
#                     registered_color = gpu_corrected.download()
#
#
#                     H_dict[os.path.basename(i)] = H, new_coords_list
#
#                     match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
#                     if match:
#                         patch_y, patch_x = int(match.group(1)), int(match.group(2))
#                         patch = registered_color[:2048, :2048]
#                         end_y = min(patch_y + patch.shape[0], HEIGHT)
#                         end_x = min(patch_x + patch.shape[1], WIDTH)
#                         actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
#                         merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
#                             merged_image[patch_y:end_y, patch_x:end_x], actual_patch
#                         )
#                         # merged_image[patch_y:end_y, patch_x:end_x]=actual_patch
#                         # region = merged_image[patch_y:end_y, patch_x:end_x]
#                         # mask = region == 0  # 找出非0区域
#                         # region[mask] = actual_patch[mask]
#                 else:
#                     # registered_color = error_img
#                     # if not is_black:
#                     #     registered_color = white_img.copy()
#                         # registered_color = white_img
#
#                         # font = cv2.FONT_HERSHEY_SIMPLEX
#                         # font_scale = 2
#                         # thickness = 3
#                         # color = 0  # 白色（灰度图中用一个标量）
#                         # text = fr'bh_error||MAX_value:{moving_img.max()}'
#                         # # 获取文字尺寸
#                         # (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#                         #
#                         # # 计算位置（居中）
#                         # img_height, img_width = registered_color.shape
#                         # x = (img_width - text_width) // 2
#                         # y = (img_height + text_height) // 2
#                         #
#                         # # 写字
#                         # cv2.putText(registered_color, text, (x, y), font, font_scale, color, thickness,
#                         #             lineType=cv2.LINE_AA)
#                         # cv2.imwrite(rf'test_error\{os.path.basename(i)}', registered_color)
#                     # else:
#                     registered_color = error_img
#                     H_dict[os.path.basename(i)] = None
#
#                     match=re.search(r'patch+_y(\d+)_x(\d+)\.', i)
#                     if match:
#                         patch_y, patch_x = int(match.group(1)), int(match.group(2))
#                         patch = registered_color[:2048, :2048]
#                         end_y = min(patch_y + patch.shape[0], HEIGHT)
#                         end_x = min(patch_x + patch.shape[1], WIDTH)
#                         actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
#                         merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
#                             merged_image[patch_y:end_y, patch_x:end_x], actual_patch
#                         )
#                         # region = merged_image[patch_y:end_y, patch_x:end_x]
#                         # mask = region == 0  # 找出非0区域
#                         # region[mask] = actual_patch[mask]
#
#                         # merged_image[patch_y:end_y, patch_x:end_x]=actual_patch
#
#             except Exception as e:
#                 print(e)
#                 # if not is_black:
#                 # registered_color = error_img
#
#                     # font = cv2.FONT_HERSHEY_SIMPLEX
#                     # font_scale = 2
#                     # thickness = 3
#                     # color = 0  # 白色（灰度图中用一个标量）
#                     # text = fr'match_error||MAX_value：{moving_img.max()}'
#                     # # 获取文字尺寸
#                     # (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#                     #
#                     # # 计算位置（居中）
#                     # img_height, img_width = registered_color.shape
#                     # x = (img_width - text_width) // 2
#                     # y = (img_height + text_height) // 2
#                     #
#                     # # 写字
#                     # cv2.putText(registered_color, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
#                     # cv2.imwrite(rf'test_error\{os.path.basename(i)}', registered_color)
#                 # else:
#                     # registered_color = error_img
#
#                 # H_global = H.copy()
#
#                 # H_global = H.copy()
#                 H=H_global
#                 print(H)
#                 registered_color = cv2.warpPerspective(moving_img, H_global, (fixed_img.shape[1], fixed_img.shape[0]))
#                 new_coords_list = []
#                 print(fixed_img.shape)
#                 print(moving_img.shape)
#                 h, w = fixed_img.shape
#                 stream = cv2.cuda_Stream()
#                 gpu_gray_fixed = cv2.cuda_GpuMat()
#                 gpu_gray_registered = cv2.cuda_GpuMat()
#                 gpu_gray_fixed.upload(fixed_img, stream)
#                 gpu_gray_registered.upload(registered_color, stream)
#
#                 farneback1 = cv2.cuda_FarnebackOpticalFlow.create(
#                     numLevels=1, pyrScale=0.2, fastPyramids=False, winSize=55,
#                     numIters=1, polyN=5, polySigma=1.2, flags=0
#                 )
#                 gpu_flow = farneback1.calc(gpu_gray_fixed, gpu_gray_registered, None, stream)
#                 stream.waitForCompletion()
#                 flow = gpu_flow.download()
#
#                 grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
#                 map_x = (grid_x + flow[..., 0]).astype(np.float32)
#                 map_y = (grid_y + flow[..., 1]).astype(np.float32)
#                 new_coords_list.append((map_x, map_y))
#
#                 gpu_registered_color = cv2.cuda_GpuMat()
#                 gpu_registered_color.upload(registered_color, stream)
#                 gpu_map_x = cv2.cuda_GpuMat()
#                 gpu_map_y = cv2.cuda_GpuMat()
#                 gpu_map_x.upload(map_x, stream)
#                 gpu_map_y.upload(map_y, stream)
#
#                 gpu_corrected = cv2.cuda.remap(
#                     gpu_registered_color, gpu_map_x, gpu_map_y,
#                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, stream=stream
#                 )
#                 stream.waitForCompletion()
#                 registered_color = gpu_corrected.download()
#
#                 H_dict[os.path.basename(i)] = H, new_coords_list
#
#                 match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
#                 if match:
#                     patch_y, patch_x = int(match.group(1)), int(match.group(2))
#                     patch = registered_color[:2048, :2048]
#                     end_y = min(patch_y + patch.shape[0], HEIGHT)
#                     end_x = min(patch_x + patch.shape[1], WIDTH)
#                     actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
#                     merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
#                         merged_image[patch_y:end_y, patch_x:end_x], actual_patch
#                     )
#                     # merged_image[patch_y:end_y, patch_x:end_x] = actual_patch
#
#                     # region = merged_image[patch_y:end_y, patch_x:end_x]
#                     # mask = region == 0  # 找出非0区域
#                     # region[mask] = actual_patch[mask]
#         else:
#             registered_color = error_img
#             H_dict[os.path.basename(i)] = None
#             match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
#             if match:
#                 patch_y, patch_x = int(match.group(1)), int(match.group(2))
#                 patch = registered_color[:2048, :2048]
#
#                 end_y = min(patch_y + patch.shape[0], HEIGHT)
#                 end_x = min(patch_x + patch.shape[1], WIDTH)
#                 actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
#                 merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
#                     merged_image[patch_y:end_y, patch_x:end_x], actual_patch
#                 )
#         torch.cuda.empty_cache()
#         cv2.cuda.resetDevice()
#
#         # === 满足 20%、40%、60%、80%、100% 触发一次异步处理 ===
#         if (idx + 1) % chunk_size == 0 or (idx + 1) == total:
#             print(f"\n>>> 第 {chunk_index + 1}/{total // chunk_size + 1} 段处理完成，准备提交线程处理 H_dict")
#             sub_H_dict = copy.deepcopy(H_dict)
#             H_dict.clear()
#             chunk_index += 1
#             thread_executor = ThreadPoolExecutor(max_workers=1)
#             # print('sub_H',len(sub_H_dict))
#             future = thread_executor.submit(process_subdirs_with_dict, subdirs, sub_H_dict, HEIGHT, WIDTH,merged_memory)
#             all_threads.append(future)
#     # 等待所有线程完成
#     for f in all_threads:
#         f.result()
#     output_path = f'{os.path.dirname(out_path)}/{dir_path.split(os.sep)[-1]}_DAPI_out1.png'
#     # print(out_path)
#     cv2.imwrite(output_path, merged_image)
#
#     # print(merged_memory)
#     for subdir_path, merged_image in merged_memory.items():
#         try:
#             output_path = f'{subdir_path}/{subdir_path.split(os.sep)[-2]}_{subdir_path.split(os.sep)[-1]}_out1.png'
#             # print(out_path)
#             # print(subdir_path)
#             # print(merged_image.shape)
#             # exit()
#             cv2.imwrite(output_path, merged_image)
#         except Exception as e:
#             print('[主进程写图失败]', e)
#     # 保存最终大图











def process_directory(dir_path, images_path, HEIGHT, WIDTH, first_dir,matcher):
    # 声明使用全局变量

    """在进程中处理目录的函数（每个进程处理一个目录）"""
    global map_x, map_y
    print(f"进程 {os.getpid()} 正在处理目录: {dir_path}")
    # SuperPoint+LightGlue
    # print(fixed_name)
    # print(images_path)
    device = 'cuda'  # 也可根据需求选择其他GPU（如cuda:1/2/3）

    # extractor = ALIKED256().eval().to(device)  # load the extractor
    # trt_logger = LOGGER_DICT["verbose"]
    # trt_model_path = "aliked-n16512.trt",
    # model_name = "aliked-n16",
    from src.loftr import LoFTR, default_cfg
    # global_stream = torch.cuda.Stream()  # 创建全局流对象

    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    from copy import deepcopy

    # %%
    # sess_opts = ort.SessionOptions()
    # providers = (
    #     ["CUDAExecutionProvider"]
    #     if device == "cuda"
    #     else ["CPUExecutionProvider"]
    # )

    # lightglue_path = fr"D:\LightGlue-ONNX-aliked_support\weights\aliked_lightglue.onnx"
    #
    # lightglue = ort.InferenceSession(
    #     lightglue_path,
    #     sess_options=sess_opts,
    #     providers=providers,
    # )
    # matcher_trt = lightglue
    # trt_flag = False
    # if trt_flag:
    #     matcher = matcher_pt
    # else:
    #     matcher = matcher_pt
    extractor = TRTInference("aliked-n16512.trt", "aliked-n16", trt.Logger(trt.Logger.ERROR))
    matcher_lightglue = LightGlue(features='aliked', filter_threshold=0.05).eval().to(device)

    # matcher.compile(mode='reduce-overhead')
    # extractor1 = ALIKED1024(max_num_keypoints=256).eval().to(device)  # load the extractor
    # matcher1 = LightGlue(features='aliked').eval().to(device)  # load the matcher
    # from copy import deepcopy
    # cfg = deepcopy(default_cfg)  # 每个进程自己一份 config

    # 获取该目录下的所有子目录
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path)
               if os.path.isdir(os.path.join(dir_path, d)) and d != 'DAPI']
    import copy
    from concurrent.futures import ThreadPoolExecutor
    from itertools import repeat

    # 初始化
    H_dict = {}
    merged_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    # merged_image = np.full((HEIGHT, WIDTH), dtype=np.uint8, fill_value=127)
    error_img = np.full((2048, 2048), 0, dtype=np.uint8)
    # white_img = np.full((2048, 2048), 255, dtype=np.uint8)
    merged_memory = {}  # key: subdir_path, value: merged_image（np.ndarray）
    # error_img1 = np.full((2048, 2048), 127, dtype=np.uint8)

    # 新增：缓存失败patch列表，格式[(路径, fixed_img, moving_img), ...]
    pending_warp_list = []

    # 计算切分点
    total = len(images_path)
    chunk_size = 100
    chunk_index = 0
    out_path = ''
    # 用于记录所有线程的句柄
    all_threads = []

    # 定义异步处理函数
    def process_subdirs_with_dict(subdirs, H_dict_chunk, HEIGHT, WIDTH, merged_memory):
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(process_subdir, subdirs, repeat(H_dict_chunk), repeat(HEIGHT), repeat(WIDTH), repeat(merged_memory),
                         repeat(first_dir))

    H_global = None  # 初始化全局H矩阵为空

    # 主处理循环
    for idx, i in enumerate(tqdm.tqdm(images_path)):

        # if os.path.basename(dir_path) != 'MOUSIFvimentinLHSP60HAQP2H_2' or os.path.basename(
        #         i) != f'patch_y{21424}_x18128.png':
        #     continue
        # print(os.path.basename(dir_path))
        # print(os.path.basename(i))
        s=time.time()
        fixed_img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        moving_img = cv2.imread(i.replace(i.split("\\")[1], dir_path.split("\\")[-1]), cv2.IMREAD_GRAYSCALE)
        is_Loftr = False
        if fixed_img.max() <= 10 :
            is_black = True
        else:
            feats0 = extractor.run(cv2.resize(fixed_img, (512, 512)))
            if (feats0['keypoint_scores'] > 0.1).sum().item() <= 10:

                is_black = True
                print('背景',torch.cuda.memory_allocated() / 1024 / 1024, "MB used")

            else:
                if (feats0['keypoint_scores'] > 0.8).sum().item() <= 100:
                    # print(1)
                    try:

                        # H1, flag = Loftr_pt(fixed_img, moving_img, device, matcher=matcher)
                        # registered_color = cv2.warpPerspective(moving_img, H1,
                        #                                        (moving_img.shape[1], moving_img.shape[0]))
                        # torch.cuda.empty_cache()
                        # gpu_gray_fixed = cv2.cuda_GpuMat()
                        # gpu_gray_registered = cv2.cuda_GpuMat()
                        # gpu_gray_fixed.upload(fixed_img)
                        # gpu_gray_registered.upload(moving_img)
                        #
                        # # 创建第一次光流对象
                        # farneback1 = cv2.cuda_FarnebackOpticalFlow.create(
                        #     numLevels=1,
                        #     pyrScale=0.2,
                        #     fastPyramids=False,
                        #     winSize=55,
                        #     numIters=1,
                        #     polyN=5,
                        #     polySigma=1.2,
                        #     flags=0
                        # )
                        #
                        # s = time.time()
                        # # 第一次光流计算（异步）
                        # gpu_flow = farneback1.calc(gpu_gray_fixed, gpu_gray_registered, None)
                        #
                        # map_list=[]
                        # # 下载flow（同步）
                        # flow = gpu_flow.download()
                        # h, w = moving_img.shape[:2]
                        # # 构造映射坐标
                        # grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                        # map_x = (grid_x + flow[..., 0]).astype(np.float32)
                        # map_y = (grid_y + flow[..., 1]).astype(np.float32)
                        # map_list.append((map_x, map_y))
                        # # 上传原图和映射坐标（异步）
                        # gpu_registered_color = cv2.cuda_GpuMat()
                        # gpu_registered_color.upload(registered_color)
                        # gpu_map_x = cv2.cuda_GpuMat()
                        # gpu_map_y = cv2.cuda_GpuMat()
                        # gpu_map_x.upload(map_x)
                        # gpu_map_y.upload(map_y)
                        #
                        # # remap（异步）
                        # gpu_corrected = cv2.cuda.remap(
                        #     gpu_registered_color, gpu_map_x, gpu_map_y,
                        #     interpolation=cv2.INTER_LINEAR,
                        #     borderMode=cv2.BORDER_CONSTANT,
                        #     stream=stream
                        # )
                        #
                        # # 下载第一次矫正后的结果（同步）
                        # registered_color = gpu_corrected.download()

                        with torch.no_grad():  # 禁用梯度计算
                            H, flag = Loftr_pt(fixed_img, moving_img, device, matcher=matcher)
                            registered_color = cv2.warpPerspective(moving_img, H,
                                                                   (moving_img.shape[1], moving_img.shape[0]))
                            torch.cuda.empty_cache()

                            # 显存优化点1：立即释放不再需要的变量
                            # del H1, flag
                            # torch.cuda.empty_cache()

                        # 第二阶段：光流校正（保持原始参数）
                        stream = cv2.cuda_Stream()  # 使用异步流

                        # 显存优化点2：GPU Mat对象池（复用内存）
                        gpu_mats = {
                            'fixed': cv2.cuda_GpuMat(),
                            'moving_img': cv2.cuda_GpuMat(),
                            'corrected': cv2.cuda_GpuMat(),
                            'flow': cv2.cuda_GpuMat(),
                            'map_x': cv2.cuda_GpuMat(),
                            'map_y': cv2.cuda_GpuMat()
                        }

                        # 第一次光流计算
                        gpu_mats['fixed'].upload(fixed_img.astype(np.float32), stream)
                        gpu_mats['moving_img'].upload(registered_color.astype(np.float32), stream)
                        stream.waitForCompletion()

                        farneback1 = cv2.cuda_FarnebackOpticalFlow.create(
                            numLevels=1, pyrScale=0.2, fastPyramids=False,
                            winSize=55, numIters=1, polyN=5, polySigma=1.2, flags=0
                        )
                        gpu_mats['flow'] = farneback1.calc(gpu_mats['fixed'], gpu_mats['moving_img'], None, stream)
                        stream.waitForCompletion()

                        # 生成第一次映射
                        flow = gpu_mats['flow'].download()
                        h, w = moving_img.shape[:2]
                        y_coords, x_coords = np.indices((h, w))

                        # ★关键修改1：确保连续内存和正确类型
                        map_x = np.ascontiguousarray((x_coords + flow[..., 0]).astype(np.float32))
                        map_y = np.ascontiguousarray((y_coords + flow[..., 1]).astype(np.float32))
                        # 释放不再需要的对象
                        del flow, x_coords, y_coords, farneback1
                        torch.cuda.empty_cache()

                        # 第一次remap - ★关键修改2：显式创建目标GpuMat
                        gpu_mats['corrected'].upload(registered_color.astype(np.float32), stream)
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
                        registered_color = gpu_corrected.download()
                        print('loftr变换成功')

                        is_black = False
                        is_Loftr=True
                    except Exception as e:
                        is_black = True
                    torch.cuda.empty_cache()
                else:
                    # print(2)
                    try:
                        with torch.no_grad():  # 禁用梯度计算
                            feats0 = {'keypoints': feats0['keypoints'] * 4, 'descriptors': feats0['descriptors'],
                                      'keypoint_scores': feats0['keypoint_scores'], 'image_size': feats0['image_size']}
                            feats1 = extractor.run(cv2.resize(moving_img, (512, 512)))
                            feats1 = {'keypoints': feats1['keypoints'] * 4, 'descriptors': feats1['descriptors'],
                                      'keypoint_scores': feats1['keypoint_scores'], 'image_size': feats1['image_size']}
                            matches01 = matcher_lightglue({'image0': feats0, 'image1': feats1})
                            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                            matches = matches01['matches']  # indices with shape (K,2)
                            # print(matches)
                            # print(matches.shape)
                            points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
                            points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
                            # print('时间', (time.time() - s) * 1000)

                            pts0 = points0.cpu().numpy().astype(np.float32)  # 匹配点来自你的代码
                            pts1 = points1.cpu().numpy().astype(np.float32)
                            H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_DEFAULT)
                            result = evaluate_homography(
                                H,
                                matches=matches,
                                img_shape=moving_img.shape[:2],  # (h, w)
                            )
                            if not result['valid']:
                                print(f"❌ 配准失败：{flag['reason']}")
                                # print("具体参数：", result['metrics'])
                            else:
                                print("✅ 变换合理")
                            flag=result['valid']
                            registered_color = cv2.warpPerspective(moving_img, H,
                                                                   (moving_img.shape[1], moving_img.shape[0]))
                            del feats0, feats1, matches01, matches, points0, points1, mask
                            torch.cuda.empty_cache()

                        # gray_fixed = cv2.cvtColor(fixed1, cv2.COLOR_BGR2GRAY)
                        # gray_registered = cv2.cvtColor(registered_color1, cv2.COLOR_BGR2GRAY)
                        # h, w = gray_fixed.shape
                        torch.cuda.empty_cache()

                        # map_list=[]
                        # # 创建异步 CUDA Stream
                        # stream = cv2.cuda_Stream()
                        #
                        # # 上传灰度图（异步）
                        # gpu_gray_fixed = cv2.cuda_GpuMat()
                        # gpu_gray_registered = cv2.cuda_GpuMat()
                        # gpu_gray_fixed.upload(fixed_img, stream)
                        # gpu_gray_registered.upload(moving_img, stream)
                        #
                        # # 创建第一次光流对象
                        # farneback1 = cv2.cuda_FarnebackOpticalFlow.create(
                        #     numLevels=1,
                        #     pyrScale=0.2,
                        #     fastPyramids=False,
                        #     winSize=55,
                        #     numIters=1,
                        #     polyN=5,
                        #     polySigma=1.2,
                        #     flags=0
                        # )
                        #
                        # s = time.time()
                        # # 第一次光流计算（异步）
                        # gpu_flow = farneback1.calc(gpu_gray_fixed, gpu_gray_registered, None, stream)
                        #
                        # # 等待计算完成
                        # stream.waitForCompletion()
                        #
                        # # 下载flow（同步）
                        # flow = gpu_flow.download()
                        # h,w=moving_img.shape[:2]
                        # # 构造映射坐标
                        # grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                        # map_x = (grid_x + flow[..., 0]).astype(np.float32)
                        # map_y = (grid_y + flow[..., 1]).astype(np.float32)
                        # map_list.append((map_x, map_y))
                        #
                        # # 上传原图和映射坐标（异步）
                        # gpu_registered_color = cv2.cuda_GpuMat()
                        # gpu_registered_color.upload(registered_color, stream)
                        # gpu_map_x = cv2.cuda_GpuMat()
                        # gpu_map_y = cv2.cuda_GpuMat()
                        # gpu_map_x.upload(map_x, stream)
                        # gpu_map_y.upload(map_y, stream)
                        #
                        # # remap（异步）
                        # gpu_corrected = cv2.cuda.remap(
                        #     gpu_registered_color, gpu_map_x, gpu_map_y,
                        #     interpolation=cv2.INTER_LINEAR,
                        #     borderMode=cv2.BORDER_CONSTANT,
                        #     stream=stream
                        # )
                        #
                        # # 等待 remap 完成
                        # stream.waitForCompletion()
                        #
                        # # 下载第一次矫正后的结果（同步）
                        # corrected_final = gpu_corrected.download()
                        #
                        # print("大光流时间:", time.time() - s)
                        # s = time.time()
                        #
                        # # 第二次矫正迭代
                        #
                        # # 转灰度并上传（异步）
                        # # gray_corrected = cv2.cvtColor(corrected_final, cv2.COLOR_BGR2GRAY)
                        # gpu_gray_corrected = cv2.cuda_GpuMat()
                        # gpu_gray_corrected.upload(corrected_final, stream)
                        #
                        # # 创建第二次光流对象（参数不同）
                        # farneback2 = cv2.cuda_FarnebackOpticalFlow.create(
                        #     numLevels=1,
                        #     pyrScale=0.2,
                        #     fastPyramids=False,
                        #     winSize=10,
                        #     numIters=1,
                        #     polyN=5,
                        #     polySigma=1.2,
                        #     flags=0
                        # )
                        #
                        # # 第二次光流计算（异步）
                        # gpu_flow2 = farneback2.calc(gpu_gray_fixed, gpu_gray_corrected, None, stream)
                        #
                        # # 等待计算完成
                        # stream.waitForCompletion()
                        #
                        # # 下载flow2（同步）
                        # flow2 = gpu_flow2.download()
                        #
                        # # 构造第二次映射坐标
                        # map_x2 = (grid_x + flow2[..., 0]).astype(np.float32)
                        # map_y2 = (grid_y + flow2[..., 1]).astype(np.float32)
                        # gpu_map_x2 = cv2.cuda_GpuMat()
                        # gpu_map_y2 = cv2.cuda_GpuMat()
                        # gpu_map_x2.upload(map_x2, stream)
                        # gpu_map_y2.upload(map_y2, stream)
                        # map_list.append((map_x2, map_y2))
                        #
                        # # 第二次 remap（异步）
                        # gpu_corrected_final = cv2.cuda.remap(
                        #     gpu_corrected, gpu_map_x2, gpu_map_y2,
                        #     interpolation=cv2.INTER_LINEAR,
                        #     borderMode=cv2.BORDER_CONSTANT,
                        #     stream=stream
                        # )
                        #
                        # # 等待 remap 完成
                        # stream.waitForCompletion()
                        #
                        # # 下载第二次矫正结果（同步）
                        # registered_color = gpu_corrected_final.download()
                        # print('aliked',torch.cuda.memory_allocated() / 1024 / 1024, "MB used")
                        print('aliked变换成功')
                        stream = cv2.cuda_Stream()  # 使用异步流

                        # 显存优化点2：GPU Mat对象池（复用内存）
                        gpu_mats = {
                            'fixed': cv2.cuda_GpuMat(),
                            'moving_img': cv2.cuda_GpuMat(),
                            'corrected': cv2.cuda_GpuMat(),
                            'flow': cv2.cuda_GpuMat(),
                            'map_x': cv2.cuda_GpuMat(),
                            'map_y': cv2.cuda_GpuMat()
                        }

                        # 第一次光流计算
                        gpu_mats['fixed'].upload(fixed_img.astype(np.float32), stream)
                        gpu_mats['moving_img'].upload(registered_color.astype(np.float32), stream)
                        stream.waitForCompletion()

                        farneback1 = cv2.cuda_FarnebackOpticalFlow.create(
                            numLevels=1, pyrScale=0.2, fastPyramids=False,
                            winSize=55, numIters=1, polyN=5, polySigma=1.2, flags=0
                        )
                        gpu_mats['flow'] = farneback1.calc(gpu_mats['fixed'], gpu_mats['moving_img'], None, stream)
                        stream.waitForCompletion()

                        # 生成第一次映射
                        flow = gpu_mats['flow'].download()
                        h, w = moving_img.shape[:2]
                        y_coords, x_coords = np.indices((h, w))

                        # ★关键修改1：确保连续内存和正确类型
                        map_x = np.ascontiguousarray((x_coords + flow[..., 0]).astype(np.float32))
                        map_y = np.ascontiguousarray((y_coords + flow[..., 1]).astype(np.float32))
                        # 释放不再需要的对象
                        del flow, x_coords, y_coords, farneback1
                        torch.cuda.empty_cache()

                        # 第一次remap - ★关键修改2：显式创建目标GpuMat
                        gpu_mats['corrected'].upload(registered_color.astype(np.float32), stream)
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
                        registered_color = gpu_corrected.download()
                        is_black = False
                        is_Loftr = False
                    except Exception as e:
                        print(f"处理过程中出错: {str(e)}")
                        # import traceback
                        # traceback.print_exc()
                        is_black = True

                    finally:
                        # 显存释放

                        torch.cuda.empty_cache()

                    # torch.cuda.empty_cache()
            # del feats0,feats1
            # gc.collect()
        if out_path == '':
            out_path = os.path.dirname(
                i.replace(i.split("\\")[1], dir_path.split("\\")[-1]).replace('crop_moving_img', 'crop_ending_img'))
            out_path1 = os.path.dirname(i.replace(i.split("\\")[1], dir_path.split("\\")[-1]).replace('crop_moving_img',
                                                                                                      'crop_ending_img_aliked'))
            os.makedirs(out_path1, exist_ok=True)

            os.makedirs(out_path, exist_ok=True)

        if not is_black:
            # print(1)
            try:
                # s=time.time()
                if flag:
                    # print(H)
                    # print('配准耗时：',time.time()-s)
                    # clear_memory()
                    # if  H is not None:
                    #     成功匹配，更新全局H矩阵
                    # --- 新增：首次成功后，处理之前缓存的失败patch ---
                    # 上传灰度图到 GPU


                    H_dict[os.path.basename(i)] = H,(map_x,map_y)
                    match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)

                    if match:
                        patch_y, patch_x = int(match.group(1)), int(match.group(2))
                        patch = registered_color[:2048, :2048]
                        end_y = min(patch_y + patch.shape[0], HEIGHT)
                        end_x = min(patch_x + patch.shape[1], WIDTH)
                        actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
                        merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
                            merged_image[patch_y:end_y, patch_x:end_x], actual_patch
                        )
                        # merged_image[patch_y:end_y, patch_x:end_x] = actual_patch

                else:
                    # print(2)
                    registered_color = error_img
                    H_dict[os.path.basename(i)] = None
                    match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
                    if match:
                        patch_y, patch_x = int(match.group(1)), int(match.group(2))
                        patch = registered_color[:2048, :2048]
                        end_y = min(patch_y + patch.shape[0], HEIGHT)
                        end_x = min(patch_x + patch.shape[1], WIDTH)
                        actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
                        merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
                            merged_image[patch_y:end_y, patch_x:end_x], actual_patch
                        )
                        # merged_image[patch_y:end_y, patch_x:end_x] = actual_patch
            except Exception as e:
                # exit()
                # print(3)

                print('处理失败',e)
                registered_color = error_img
                H_dict[os.path.basename(i)] = None
                match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
                if match:
                    patch_y, patch_x = int(match.group(1)), int(match.group(2))
                    patch = registered_color[:2048, :2048]
                    end_y = min(patch_y + patch.shape[0], HEIGHT)
                    end_x = min(patch_x + patch.shape[1], WIDTH)
                    actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
                    merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
                        merged_image[patch_y:end_y, patch_x:end_x], actual_patch
                    )
                    # merged_image[patch_y:end_y, patch_x:end_x] = actual_patch

        else:
            # print(4)

            registered_color = error_img
            H_dict[os.path.basename(i)] = None
            match = re.search(r'patch+_y(\d+)_x(\d+)\.', i)
            if match:
                patch_y, patch_x = int(match.group(1)), int(match.group(2))
                patch = registered_color[:2048, :2048]

                end_y = min(patch_y + patch.shape[0], HEIGHT)
                end_x = min(patch_x + patch.shape[1], WIDTH)
                actual_patch = patch[:end_y - patch_y, :end_x - patch_x]
                merged_image[patch_y:end_y, patch_x:end_x] = np.maximum(
                    merged_image[patch_y:end_y, patch_x:end_x], actual_patch
                )
                # merged_image[patch_y:end_y, patch_x:end_x] = actual_patch
            print(f"[处理背景]{i}")
        print(out_path)
        cv2.imwrite(os.path.join(out_path,os.path.basename(i)), registered_color)
        from PIL import Image, ImageOps, ImageEnhance

        # def get_color_img(img, color):
        #     # res_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #     # for i in range(img.shape[2]):
        #     img = Image.fromarray(img)
        #     res_img = ImageOps.colorize(img, black='black', white=color)
        #     s = time.time()
        #     enhancer = ImageEnhance.Brightness(res_img)
        #     res_img = enhancer.enhance(2)
        #     print((time.time() - s) * 1000)
        #     # res_img = np.clip(res_img + np.array(color_img), 0, 255)
        #     # res_img = Image.fromarray(res_img)
        #     return res_img


        # get_color_img(registered_color, 'yellow').save(i.replace(i.split("\\")[1], dir_path.split("\\")[-1]).replace('crop_moving_img', 'crop_ending_img'))
        #
        # get_color_img(fixed_img, 'blue').save((i.replace(i.split("\\")[1], dir_path.split("\\")[-1]).replace('crop_moving_img', 'crop_ending_img')).replace('.png','_fixed.png'))
        torch.cuda.empty_cache()
        # cv2.cuda.resetDevice()

        # === 满足 20%、40%、60%、80%、100% 触发一次异步处理 ===
        if (idx + 1) % chunk_size == 0 or (idx + 1) == total:
            print(f"\n>>> 第 {chunk_index + 1}/{total // chunk_size + 1} 段处理完成，准备提交线程处理 H_dict")
            sub_H_dict = copy.deepcopy(H_dict)
            H_dict.clear()
            chunk_index += 1
            thread_executor = ThreadPoolExecutor(max_workers=1)
            # print('sub_H',len(sub_H_dict))
            future = thread_executor.submit(process_subdirs_with_dict, subdirs, sub_H_dict, HEIGHT, WIDTH, merged_memory)
            all_threads.append(future)
        print('循环时间',time.time()-s)

    # 等待所有线程完成
    for f in all_threads:
        f.result()
    output_path = f'{os.path.dirname(out_path)}/{dir_path.split(os.sep)[-1]}_DAPI_out1.png'
    # print(out_path)
    cv2.imwrite(output_path, merged_image)

    # print(merged_memory)
    for subdir_path, merged_image in merged_memory.items():
        try:
            output_path = f'{subdir_path}/{subdir_path.split(os.sep)[-2]}_{subdir_path.split(os.sep)[-1]}_out1.png'
            # print(out_path)
            # print(subdir_path)
            # print(merged_image.shape)
            # exit()
            cv2.imwrite(output_path, merged_image)
        except Exception as e:
            print('[主进程写图失败]', e)










def main(params):
    s = time.time()
    A_dir = params['data_dir']
    # 原始目录结构
    # A_dir = r'HC356_Gray_date/HC356'
    for i in os.listdir(A_dir):
        for j in os.listdir(os.path.join(A_dir, i)):
            # print(j)
            IMG_LISTS = os.listdir(os.path.join(A_dir, i, 'DAPI'))
            for png in IMG_LISTS:
                if png.endswith('.png'):
                    # print(png)
                    img = Image.open(os.path.join(A_dir, i, 'DAPI', png))
                    global HEIGHT, WIDTH
                    WIDTH = img.size[0]
                    HEIGHT = img.size[1]
                    # print(WIDTH, HEIGHT)
                    break
            break
        break
    # 确保A目录存在
    if not os.path.exists(A_dir):
        print(f"目录 {A_dir} 不存在！")
        return

    # 获取A目录下所有子目录
    all_subdirs = [os.path.join(A_dir, d) for d in sorted(os.listdir(A_dir))
                   if os.path.isdir(os.path.join(A_dir, d))]
    len_subdirs = len(all_subdirs)
    # if len(all_subdirs) < 4:
    #     print("A目录下需要至少4个子目录！")
    #     return

    # 步骤1：取出第一个子目录（可根据需要处理）
    first_dir = all_subdirs[0]
    print(f"取出第一个子目录: {first_dir}")
    # fixed_name=''
    images_path = []
    # fixed_img_cache = {}

    for i in os.listdir(os.path.join(first_dir, 'DAPI', 'crop_moving_img')):
        images_path.append(os.path.join(first_dir, 'DAPI', 'crop_moving_img', i))
        # if i not in fixed_img_cache:
        # fixed_img_cache[i] = cv2.imread(os.path.join(first_dir,'DAPI','crop_moving_img',i))  # 只读一次
    # 这里可以添加对第一个子目录的处理
    # for i in os.listdir(first_dir):
    #     if i=='DAPI':
    #         for j in os.listdir(os.path.join(first_dir, i)):
    #             if j.endswith('image.png'):
    #                 fixed_path = os.path.join(first_dir, i, j)
    #                 # print(j)
    #                 fixed_name=j
    #                 # print(fixed_name)
    #                 fixed_img=cv2.imread(fixed_path)
    #                 # print(fixed_path)
    #                 shutil.copy(fixed_path, fixed_path.replace('.png', '_moving.png'))
    #     else:
    #         for j in os.listdir(os.path.join(first_dir, i)):
    #             if j.endswith('image.png'):
    #                 shutil.copy(os.path.join(first_dir, i, j), os.path.join(first_dir, i, j).replace('.png', '_moving.png'))
    # print(images_path)
    remaining_dirs = all_subdirs[1:len_subdirs]  # 只取接下来的3个
    matcher = LoFTR(config=default_cfg).half()

    matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    matcher.share_memory()  # 关键！允许多进程共享模型内存

    # print(fixed_img.shape)
    # 步骤3：对这3个子目录创建3个进程，每个进程再创建2个线程
    # print("\n开始并行处理（3个进程，每个进程2个线程）:")
    # for dir_path in remaining_dirs:
    #     print(f"\n处理目录: {dir_path}")
    #     process_directory(dir_path, images_path)
    with ProcessPoolExecutor(max_workers=len_subdirs - 1) as executor:
    # with ProcessPoolExecutor(max_workers=1) as executor:

        executor.map(process_directory, remaining_dirs, repeat(images_path), repeat(HEIGHT), repeat(WIDTH),repeat(os.path.basename(first_dir)),repeat(matcher))

    print("\n处理完成！")
    print(time.time() - s)


if __name__ == "__main__":
    main()
