
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.colors import to_rgb

import os
import cv2
import numpy as np
import pyvista as pv
from matplotlib import cm
folder = r"F:\3d\version3_6\test_3d"
files = sorted(os.listdir(folder))

slices = []
for f in files:
    img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (2048, 2048))  # 为了加快点云渲染，先缩小
    slices.append(img)

volume = np.stack(slices, axis=0)  # shape = (z, y, x)

# 2. 构造点坐标 (z, y, x)
zz, yy, xx = np.meshgrid(
    np.arange(volume.shape[0]),
    np.arange(volume.shape[1]),
    np.arange(volume.shape[2]),
    indexing="ij"
)

# 3. 过滤灰度 < 20 的点
mask = volume.ravel() >= 20
points = np.column_stack((xx.ravel()[mask],
                          yy.ravel()[mask],
                          zz.ravel()[mask]))

# 4. 灰度映射到颜色 (使用 matplotlib colormap)
normed = volume.ravel()[mask] / 255.0
colors_gray = cm.gray(normed)[:, :3]  # 取 RGB

# 5. 根据层数给颜色加偏移（让不同层有不同色调）
z_norm = zz.ravel()[mask] / volume.shape[0]
colors_layer = cm.hsv(z_norm)[:, :3]   # HSV 色盘映射层号
colors_final = 0.6 * colors_gray + 0.4 * colors_layer  # 混合

# 6. 构造点云
point_cloud = pv.PolyData(points)
point_cloud["colors"] = (colors_final * 255).astype(np.uint8)

# 7. 显示
plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars="colors", rgb=True, point_size=4.0, opacity=1.0)
plotter.show()