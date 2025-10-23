
import os
import ast
key='Mhaima_1_SpGreen'
params = {
    'input_dir': 'F:\9_17_shenjing\Mhaima'}
txt_path = params['input_dir']
# 读取 txt 文件，获取 DAPI 图像路径
for txt_file in os.listdir(txt_path):
    if txt_file.endswith('.txt'):
        with open(os.path.join(txt_path, txt_file), 'r') as f:
            content = f.read()
            data_dict = ast.literal_eval(content)

def rename_channel(key, data_dict):
    parts = key.split("_", 2)  # 最多分割 2 次，避免被多余的 _ 打散
    parts=[parts[0] + "_" + parts[1], parts[2]]
    print(parts)
    print(data_dict)
    zb = data_dict[parts[0]][parts[1].lower()]
    return f"{parts[0]}_{zb}"


print(rename_channel(key, data_dict))