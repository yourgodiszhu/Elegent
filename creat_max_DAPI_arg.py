# import os
# import shutil
# import time
#
# s=time.time()
#
# def find_and_copy_moving_png(source_dir, target_dir):
#     # 创建目标目录（如果不存在）
#     os.makedirs(target_dir, exist_ok=True)
#
#     count = 0  # 用于统计复制的文件数量
#
#     # 遍历源目录
#     for root, dirs, files in os.walk(source_dir):
#         for file in files:
#             if file.endswith('out1.png'):
#                 src_path = os.path.join(root, file)
#                 # 构建目标路径（保持原文件名）
#                 dst_path = os.path.join(target_dir, file)
#                 print(src_path)
#                 print(dst_path)
#                 # exit()
#                 # 检查目标目录中是否已存在同名文件
#                 # 如果有冲突，添加前缀数字来避免覆盖
#                 # index = 1
#                 # while os.path.exists(dst_path):
#                 #     name, ext = os.path.splitext(file)
#                 #     # PRINT(src_path)
#                 #     print(name,ext,index)
#                 #     print(src_path)
#                 # dst_path = os.path.join(target_dir, )
#                 # index += 1
#
#                 # 复制文件
#                 shutil.copy2(src_path, dst_path)
#                 print(f"已复制: {src_path} -> {dst_path}")
#                 count += 1
#     if count == 0:
#         print(f"未找到任何以'moving_img.png'结尾的文件于: {source_dir}")
#     else:
#         print(f"\n复制完成！共复制了 {count} 个文件到: {target_dir}")
#
#
# # 使用示例
# for i in range(4):
#     source_directory = fr'AR391_Gray_date/AR391/AR391_{i + 1}'  # 源目录路径
#     target_directory = 'all_out_png'  # 目标目录名称
#
#     find_and_copy_moving_png(source_directory, target_directory)
#
#
# import os
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
# import cv2
# import numpy as np
#
# # img=cv2.imread('sdk_crop_merge_4channel_png/AR391/AR391_1/DAPI/AR391_1_DAPI_merged_image.png')
# # print(img.shape)
# finally_result=None
# i=0
# for j in os.listdir('all_out_png'):
#     print(j)
#     if 'DAPI' in j  and i==0 and j!='Maximum_DAPI.png':
#         print(1)
#         img=cv2.imread(os.path.join('all_out_png',j))
#         print('读取成功')
#         finally_result =img
#         i+=1
#     if 'DAPI' in j and i !=0 and j!='Maximum_DAPI.png':
#         img=cv2.imread(os.path.join('all_out_png',j))
#         finally_result=np.maximum(finally_result, img)
# cv2.imwrite(rf'all_out_png/Maximum_DAPI.png',finally_result)
# print(time.time() - s)

# from data_rename import main as rename_main
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
import shutil
import time
import concurrent.futures
import cv2
import numpy as np

# 设置时间统计
start_time = time.time()


# 第一部分：多线程拷贝文件
def copy_file(src_path, dst_dir):
    """单个文件的拷贝任务"""
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)
    return src_path


def find_and_copy_moving_png_parallel(params):
    """使用线程池并行拷贝文件"""
    source_dirs=params['data_dir']
    target_dir=params['save_dir']
    os.makedirs(target_dir, exist_ok=True)
    # print(source_dirs)
    # print(target_dir)
    source_dirs=[f'{source_dirs}/{i}' for i in os.listdir(source_dirs)]
    # 收集所有需要拷贝的文件路径
    print(source_dirs)
    all_files = []
    for source_dir in source_dirs:
        # print(source_dir)
        for root, _, files in os.walk(source_dir):
            # print(files)
            for file in files:
                # print(file)
                if file.endswith('out1.png'):
                    all_files.append(os.path.join(root, file))
    print(all_files)
    # 使用线程池并行拷贝
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for file_path in all_files:
            futures.append(executor.submit(copy_file, file_path, target_dir))

        count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                src_path = future.result()
                print(f"已复制: {src_path}")
                count += 1
            except Exception as e:
                print(f"复制失败: {e}")

    print(f"\n复制完成!共复制了{count}个文件到:{target_dir}")
    return count


# 第二部分：多线程读取图像并计算最大投影
def process_dapi_image(file_path):
    """单个DAPI图像处理任务"""
    if 'DAPI' in os.path.basename(file_path) and not file_path.endswith('Maximum_DAPI.png'):
        return cv2.imread(file_path)


def create_max_projection_parallel(params):
    """使用线程池并行读取图像并计算最大投影"""
    image_dir = params['save_dir']
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    # 使用线程池并行读取
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_dapi_image, f) for f in image_files]

        max_projection = None
        count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                img = future.result()
                if img is not None:
                    if max_projection is None:
                        max_projection = img
                    else:
                        max_projection = np.maximum(max_projection, img)
                    count += 1
            except Exception as e:
                print(f"图像处理失败: {e}")

    if max_projection is not None:
        output_path = os.path.join(image_dir, 'Maximum_DAPI.png')
        cv2.imwrite(output_path, max_projection)
        print(f"成功生成最大投影图: {output_path}")
        return count
    return 0


# 主程序
# if __name__ == "__main__":
#     # 准备源目录列表
#     source_dirs = [f'HC356_Gray_date/HC356/HC356_{i + 1}' for i in range(4)]
#     target_dir = 'all_out_png'
    # print(source_dirs)
    # 并行拷贝文件
    # copy_count = find_and_copy_moving_png_parallel(source_dirs, target_dir)

    # 并行处理DAPI图像
    # if copy_count > 0:
    #     dapi_count = create_max_projection_parallel(target_dir)
    #     print(f"成功处理了{dapi_count}个DAPI图像")
    #
    # print(f"总耗时: {time.time() - start_time:.2f}秒")
