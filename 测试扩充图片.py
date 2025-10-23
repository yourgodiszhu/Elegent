import gc
import os

import numpy as np

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
import cv2
import  math


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
# 使用示例
img = cv2.imread("your_image.png")  # 读取原始图
img=np.full((1564,2048,3),255,dtype=np.uint8)
expanded, sx, sy, h, w = expand_to_2048_multiple(img)
fixed_img, pads = pad_to_square(expanded, pad_value=0)


fixed_img = unpad(fixed_img, pads)
cropped = remove_padding_with_offsets(fixed_img, sx, sy, h, w)

cv2.imwrite("expanded.png", img)