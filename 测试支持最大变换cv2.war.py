import os
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

import numpy as np
from skimage.transform import ProjectiveTransform, warp





expanded_moving_img=cv2.imread(r"F:\3d\version3_6\jiechangCa_Gray_date\jiechangCa\jiechangCa_1\DAPI\jiechangCa_1_DAPI_merged_image_moving.png",cv2.IMREAD_GRAYSCALE)
# expanded_moving_img=cv2.resize(expanded_moving_img,(32766,32766),interpolation=cv2.INTER_AREA)
H=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
# moving_img_end = cv2.warpPerspective(expanded_moving_img, H,
#                                                  (expanded_moving_img.shape[1], expanded_moving_img.shape[0]))
s=time.time()
transform = ProjectiveTransform(matrix=H)

# skimage warp 接受 float 图像，范围 [0,1]
img_float = expanded_moving_img.astype(np.float32) / 255.0

warped_img = warp(img_float, transform.inverse)  # 注意这里用 inverse
warped_img = (warped_img * 255).astype(np.uint8)  # 转回 uint8
print("warp time:",time.time()-s)
# warped_img = cv2.remap(
#     other_img,
#     map_x,
#     map_y,
#     interpolation=cv2.INTER_LINEAR,
#     borderMode=cv2.BORDER_CONSTANT,
#     borderValue=0
# )
