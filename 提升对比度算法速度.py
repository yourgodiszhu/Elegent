import time

import numpy as np
import os

from PIL.FontFile import WIDTH

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

import cv2

def color_enhancement( mat, black, white, gamma):
    """
    做图像拉升
    :param mat: 输入图像矩阵
    :param black: 最低界限
    :param white: 最高界限
    :param gamma: gamma值
    :return:
    """
    tmp_mat = mat.astype(np.uint8)  # 确保输入是 uint8
    tmp_mat = np.clip(tmp_mat, black, white)  # 使用 np.clip 替代条件语句
    tmp_mat = ((tmp_mat - black) / (white - black) * 255).astype(np.uint8)

    real_gamma = 2 - gamma
    table = np.array([((i / 255.0) ** real_gamma) * 255 for i in range(256)]).astype(np.uint8)
    tmp_mat = cv2.LUT(tmp_mat, table)

    return tmp_mat
def color_enhancement_fast(mat, black, white, gamma):
    """
    快速图像拉伸 + gamma 校正（与原版一致，但更快）
    """
    # 1️⃣ clip + normalize 一步完成
    # OpenCV 的 LUT/LUTExp 操作底层是 SIMD 加速的，远比 Python 循环快
    mat = np.clip(mat, black, white).astype(np.float32)
    mat = (mat - black) / (white - black) * 255.0

    # 2️⃣ gamma 变换
    real_gamma = 2.0 - gamma
    # 利用 LUT (查表) 代替逐点 pow，避免多次浮点运算
    # 预先构建一次表格，全局缓存可进一步提速
    lut = np.power(np.linspace(0, 1, 256, dtype=np.float32), real_gamma) * 255
    lut = lut.clip(0, 255).astype(np.uint8)

    # 3️⃣ 使用 OpenCV LUT（C 实现，支持并行 SIMD）
    mat = cv2.LUT(mat.astype(np.uint8), lut)

    return mat


def cv_imread_unicode(path,gray=True):
    stream = np.fromfile(path, dtype=np.uint8)
    if gray:
        return cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
img=cv_imread_unicode(r"F:\3d\version3_9_支持中文_调整曝光\R脊髓 ①-1_Gray_date\R脊髓 ①-1\R脊髓 第5轮 ①-1 αSMA-红+Vimentin-绿\DAPI\crop_moving_img\1_2_3.png",cv2.IMREAD_GRAYSCALE)
s=time.time()
img=color_enhancement_fast(img,0,132,1)
print(time.time()-s)
cv2.imshow("img",img)
cv2.waitKey(0)


