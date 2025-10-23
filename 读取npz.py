import os

import numpy as np
data_path = r"F:\3d\version3_6\yanqiu_Gray_date\yanqiu_4.npz"

data = np.load(data_path, allow_pickle=True)

# 数据加载(与原代码一致)
dir_path, first_dir, subdirs, out_path, error_patchs = data['dir_path'], data['first_dir'], data['subdirs'], \
    data['out_path'], data['error_patchs']
dir_path = str(dir_path)
first_dir = str(first_dir)
subdirs = list(subdirs)
out_path = str(out_path)
error_paths = list(error_patchs)
error_paths=[]
for i in os.listdir(os.path.join(dir_path, 'DAPI','crop_moving_img')):
    error_paths.append(i)
print("dir_path:", dir_path)
print("first_dir:", first_dir)
print("subdirs:", subdirs)           # 这里的subdirs是list类型
print("out_path:", out_path)
print("error_paths:", error_paths)
np.savez(data_path,
         dir_path=dir_path,
         first_dir=first_dir,
         subdirs=subdirs,
         out_path=out_path,
         error_patchs=error_paths)

