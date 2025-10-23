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
import pyvista as pv

# 1. 读取切片
folder = r"F:\3d\version3_6\test_3d"
files = sorted(os.listdir(folder))

slices = []
for f in files:
    img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))  # 缩小加快渲染
    slices.append(img)

volume = np.stack(slices, axis=0)  # shape = (z, y, x)

# ------------------------------
# 2. 设置灰度区间和颜色
# ------------------------------
thresholds = [30, 80, 150]          # 灰度区间
colors = ['lightblue', 'lightgreen', 'salmon']  # 对应颜色
opacity = 0.6

# ------------------------------
# 3. PyVista 渲染
# ------------------------------
plotter = pv.Plotter()

for i, level in enumerate(thresholds):
    # 对整个 volume 提取等值面
    verts, faces, normals, values = measure.marching_cubes(volume, level=level)
    if len(verts) == 0:
        continue

    # faces 需要加一列表示每个三角形顶点数
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_pv)

    # z 轴渐变色（可叠加灰度区间颜色）
    z_min, z_max = verts[:,2].min(), verts[:,2].max()
    z_norm = (verts[:,2] - z_min) / (z_max - z_min + 1e-8)  # 归一化
    # 将灰度区间颜色叠加 z 渐变
    from matplotlib.colors import to_rgb

    base_color = np.array(to_rgb(colors[i]))  # 返回 0~1 的 RGB
    # base_color = np.array(pv.parse_color(colors[i])) / 255.0
    colors_final = base_color * (0.5 +  0.5 * z_norm[:, np.newaxis])  # 简单叠加
    mesh.point_data['colors'] = colors_final

    # 添加到 plotter
    plotter.add_mesh(mesh, scalars='colors', rgb=True, opacity=opacity)

# 显示网格
plotter.show_grid()
plotter.show()