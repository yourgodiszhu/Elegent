import os
import shutil
import traceback

# from Config_Utils import IS_FLUORESCENT_SCAN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from glob import glob
import os

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)

import gc
import cv2
import numpy as np
from PIL import Image
# from tqdm import tqdm
import tifffile
import time
from skimage import morphology
import h5py
import re
# from pyzbar.pyzbar import decode
import datetime
import openslide
import json

from trt_model_lightglue import TRTInference
import tensorrt as trt

def do_cul_fov_iqa_fast(fov, is_return_rate=False, resize_rate=1):
    """
    计算fov的清晰度评价指标值，返回fov的清晰度评价指标值
    :param fov: rgb fov图像矩阵
    :return:
    """
    # fov = cv2.medianBlur(fov, 3)

    fov = cv2.resize(fov, None, None, fx=resize_rate, fy=resize_rate)
    fov = cv2.GaussianBlur(fov, (3, 3), 0).astype(np.float32)

    valid_rate = 0

    if len(fov.shape) > 2:

        if is_return_rate:
            # 计算组织占比
            var_img = ((fov[:, :, 0] - fov[:, :, 1]) ** 2 + (
                    fov[:, :, 1] - fov[:, :, 2]) ** 2 + (
                               fov[:, :, 0] - fov[:, :, 2]) ** 2) / 3
            mask = var_img > 50
            valid_rate = np.sum(mask) / (mask.shape[0] * mask.shape[1])

        fov = cv2.cvtColor(fov, cv2.COLOR_RGB2GRAY)

    lap_img = cv2.Laplacian(fov, cv2.CV_32F)
    lap_img = cv2.medianBlur(lap_img, 3)
    iqa_value_lap = np.var(lap_img)

    diff = fov[1:, 1:] - fov[:-1, :-1]
    squared_diff = np.square(diff)

    iqa_value = np.mean(squared_diff) + iqa_value_lap
    # iqa_value = iqa_value_lap

    if is_return_rate:
        return iqa_value, valid_rate
    else:
        return iqa_value

img_path=r'D:\3d\version3_4\MOUPIFCD3LCD20HCD31H_Gray_date\MOUPIFCD3LCD20HCD31H'

extractor = TRTInference("aliked-n16512.trt", "aliked-n16", trt.Logger(trt.Logger.ERROR))
x={}
y={}
count=0
for i in os.listdir(r'D:\3d\version3_4\MOUPIFCD3LCD20HCD31H_Gray_date\MOUPIFCD3LCD20HCD31H\MOUPIFCD3LCD20HCD31H_1\DAPI\crop_moving_img'):
    img1_path=os.path.join(r'D:\3d\version3_4\MOUPIFCD3LCD20HCD31H_Gray_date\MOUPIFCD3LCD20HCD31H\MOUPIFCD3LCD20HCD31H_1\DAPI\crop_moving_img',i)
    fixed1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    feats0 = extractor.run(cv2.resize(fixed1, (512, 512)))
    # print(1)
    if (feats0['keypoint_scores'] > 0.8).sum().item() >= 100:
        qxd=do_cul_fov_iqa_fast(fixed1)
        max_=None
        max_s=[]
        max_s.append(qxd)
        x[1]=qxd
        for j in range(2,4):
            img1_path = os.path.join(rf'D:\3d\version3_4\MOUPIFCD3LCD20HCD31H_Gray_date\MOUPIFCD3LCD20HCD31H\MOUPIFCD3LCD20HCD31H_{j}\DAPI\crop_moving_img', i)
            fixed1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            feats0 = extractor.run(cv2.resize(fixed1, (512, 512)))
            qxd = do_cul_fov_iqa_fast(fixed1)

            x[j] = qxd
        max_key = max(x, key=lambda k: x[k])
        if max_key not in y :
            y[max_key]=1
        else:
            y[max_key]+=1
        count+=1
    if count>=3:
        break
shutil.copy(r"D:\3d\version3_4\MOUPIFCD3LCD20HCD31H_Gray_date\MOUPIFCD3LCD20HCD31H\MOUPIFCD3LCD20HCD31H_2\DAPI\MOUPIFCD3LCD20HCD31H_2_DAPI_out1.png",r"D:\3d\version3_4\all_out_png_MOUPIFCD3LCD20HCD31H\Maximum_DAPI.png")
print(y)
print(max(y, key=lambda k: y[k]))