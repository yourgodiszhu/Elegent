
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')

import cv2
import numpy as np

def cv_imread_unicode(path,gray=True):
    stream = np.fromfile(path, dtype=np.uint8)
    print(stream)
    if gray:
        return cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
def cv_imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    result, encoded_img = cv2.imencode(ext, img)
    if result:
        encoded_img.tofile(path)
    else:
        raise Exception(f"cv2.imencode failed: {path}")


# def cv_imread_unicode(path, gray=True):
#     """修正版读取函数"""
#     with open(path, 'rb') as f:
#         img_bytes = np.frombuffer(f.read(), dtype=np.uint8)  # 关键修正
#
#     flags = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_UNCHANGED
#     img = cv2.imdecode(img_bytes, flags)
#
#     if img is None:
#         raise ValueError(f"解码失败，请检查文件是否损坏: {path}")
#     return img
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 解决PIL解码图片过大问题
def universal_imread(file_path):
    """通用图像读取函数（支持中文路径）"""
    # OpenCV优先模式
    try:
        return cv_imread_unicode(file_path)  # 上面的实现
    except:
        # 回退到PIL模式
        try:
            return np.array(Image.open(file_path))
        except:
            raise RuntimeError(f"所有图像读取方式均失败: {file_path}")
path=r"F:\3d\version3_9_支持中文_调整曝光\结肠Ca2_Gray_date\结肠Ca2\结肠Ca 第2轮 ①-2 CD19-红+CD4-绿\SpGreen\结肠Ca 第2轮 ①-2 CD19-红+CD4-绿_SpGreen_merged_image.png"
print("文件存在:", os.path.exists(path))
img=universal_imread(path)

print(img.shape)