import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
import cv2
back=np.zeros((1848,3696,3),dtype=np.uint8)
img1=cv2.imread(rf"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\DAPI\crop_ending_img\1_10_5.png")
img2=cv2.imread(rf"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\DAPI\crop_ending_img\1_11_5.png")
# 2. 定义重叠区域（假设水平方向有400像素重叠）
overlap_width = 400
overlap_left = img1[:, -overlap_width:]  # img1的右侧重叠部分
overlap_right = img2[:, :overlap_width]  # img2的左侧重叠部分

# 3. 对重叠区域进行泊松融合
mask = 255 * np.ones(overlap_left.shape, dtype=np.uint8)  # 全白mask表示融合整个区域
center = (overlap_width // 2, overlap_left.shape[0] // 2)  # 中心点坐标

# 将img2的重叠部分融合到img1的重叠区域
fused_overlap = cv2.seamlessClone(
    overlap_right,  # 源图像（img2的重叠部分）
    overlap_left,   # 目标图像（img1的重叠部分）
    mask,
    center,
    cv2.NORMAL_CLONE  # 模式：保留源图像纹理
)

# 4. 重建最终图像
height, width = img1.shape[:2]
back = np.zeros((height, width + img2.shape[1] - overlap_width, 3), dtype=np.uint8)
back[:, :width] = img1  # 左半部分
back[:, width - overlap_width : width] = fused_overlap  # 融合后的重叠区域
back[:, width:] = img2[:, overlap_width:]  # img2的剩余部分

cv2.imwrite(r"test\back_poisson.png", back)
