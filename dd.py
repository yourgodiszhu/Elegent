import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')

import numpy as np

# 读取图片
import cv2
img=cv2.imread(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\DAPI\cell1_2_DAPI_merged_image_moving.png",cv2.IMREAD_GRAYSCALE)
img2=img[2000:46000,2000:35000]
cv2.imwrite(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\DAPI\cell1_2_DAPI_merged_image_moving.png",img2)


img=cv2.imread(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\CY5\cell1_2_CY5_merged_image_moving.png",cv2.IMREAD_GRAYSCALE)
img2=img[2000:46000,2000:35000]
cv2.imwrite(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\CY5\cell1_2_CY5_merged_image_moving.png",img2)

img=cv2.imread(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\SPorange\cell1_2_SPorange_merged_image_moving.png",cv2.IMREAD_GRAYSCALE)
img2=img[2000:46000,2000:35000]
cv2.imwrite(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\SPorange\cell1_2_SPorange_merged_image_moving.png",img2)
#

img=cv2.imread(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\DAPI\cell1_2_DAPI_merged_image_moving.png",cv2.IMREAD_GRAYSCALE)
img2=img[2000:46000,2000:35000]
cv2.imwrite(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\DAPI\cell1_2_DAPI_merged_image_moving.png",img2)

img=cv2.imread(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\SpGreen\cell1_2_SpGreen_merged_image_moving.png",cv2.IMREAD_GRAYSCALE)
img2=img[2000:46000,2000:35000]
cv2.imwrite(r"F:\3d\version3_6\cell1_Gray_date\cell1\cell1_2\SpGreen\cell1_2_SpGreen_merged_image_moving.png",img2)