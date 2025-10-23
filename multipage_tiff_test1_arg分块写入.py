# coding: utf-8
import ast
import os
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


class Image_Registration:
    def __init__(self, desc, save_svs, background_image=None, label_path='', macro_im_path=''):
        self.save_svs = save_svs
        self.slide_description_info = desc
        self.objective_rate = 1
        self.background_image = background_image
        self.label_path = label_path
        self.macro_im_path = macro_im_path

    def gen_tiles_whole(self, data, tile_shape):
        for y in range(0, data.shape[0], tile_shape[0]):
            for x in range(0, data.shape[1], tile_shape[1]):
                yield data[y: y + tile_shape[0], x: x + tile_shape[1]]

    def process_large_image(self, img, resize_factor=1.0, block_size=1024):
        """
        Process a large image by dividing it into smaller blocks, resizing each block,
        and then reassembling the blocks.

        Args:
            img: Input image as numpy array
            resize_factor: Factor by which to resize the image (default: 1.0 = no resize)
            block_size: Size of blocks to process at a time

        Returns:
            Processed image as numpy array
        """
        height, width, c = img.shape

        # If no resize is needed, just return the original image
        if resize_factor == 1.0:
            return img

        # Create a large empty matrix for the result
        resized_height = int(height * resize_factor)
        resized_width = int(width * resize_factor)
        resized_img = np.zeros((resized_height, resized_width, c), dtype=img.dtype)

        # Process the image in blocks
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Get the current block
                block = img[y:min(y + block_size, height), x:min(x + block_size, width)]
                if block.shape[0] == 0 or block.shape[1] == 0:
                    continue

                # Resize the block
                block_resized = cv2.resize(block, None, fx=resize_factor, fy=resize_factor,
                                           interpolation=cv2.INTER_LINEAR)

                # Calculate new position
                new_y = int(y * resize_factor)
                new_x = int(x * resize_factor)

                # Place the resized block
                h, w = block_resized.shape[:2]
                h_remaining = min(h, resized_img.shape[0] - new_y)
                w_remaining = min(w, resized_img.shape[1] - new_x)
                resized_img[new_y:new_y + h_remaining, new_x:new_x + w_remaining] = block_resized[:h_remaining, :w_remaining]

        return resized_img

    def gen_pyramid_tiff_for_fluorescence(self, save_svs_path=''):
        import json
        import traceback
        import os
        import numpy as np
        import cv2
        from PIL import Image
        import tifffile

        # 写入信息
        mpp = 0.2738 / self.objective_rate
        mag = 20 * self.objective_rate
        filename = os.path.basename(save_svs_path)
        svs_desc = f'Aperio Image Library Fake\nABC |AppMag = {mag}|Filename = {filename}|MPP = {mpp}'
        metadata_json = json.dumps(self.slide_description_info, ensure_ascii=True)
        svs_desc += f"|METADATA_JSON = {metadata_json}"

        label_desc = 'Aperio Image Library Fake\nlabel {W}x{H}'
        macro_desc = 'Aperio Image Library Fake\nmacro {W}x{H}'

        img_num, bk_h, bk_w, bk_Channel_num = self.background_image.shape

        label_im = np.array(Image.open(self.label_path).rotate(-180, expand=True))
        macro_im = np.array(Image.open(self.macro_im_path).rotate(-180, expand=True))

        tile_hw = [512, 512]
        multi_hw = [[bk_h, bk_w], [bk_h // 2, bk_w // 2], [bk_h // 4, bk_w // 4], [bk_h // 8, bk_w // 8],
                    [bk_h // 16, bk_w // 16]]
        print(multi_hw)
        block_size = 2048
        background_image_x4 = self.process_large_image(self.background_image[0], resize_factor=0.25,
                                                       block_size=block_size)
        scale_rate = max(background_image_x4.shape[0] // 512, 1)
        thumbnail_im = cv2.resize(background_image_x4, None, None, fx=1 / scale_rate, fy=1 / scale_rate,
                                  interpolation=cv2.INTER_LANCZOS4)

        # 关键：以英寸为单位的分辨率，写入为 OpenSlide 可识别的 DPI
        ppi = 25400 / mpp  # 每英寸多少像素
        resolution = (ppi, ppi)

        compression = ['JPEG', 80, dict(outcolorspace='YCbCr')]
        kwargs = dict(photometric='rgb', planarconfig='CONTIG', compression=compression,
                      dtype=np.uint8, metadata=None)
        kwargs_png = dict(
            photometric='rgb',
            planarconfig='CONTIG',
            compression='PNG',  # 改为JPEG2000压缩
            # compressionargs={'level': 80},  # 压缩级别80
            dtype=np.uint8,
            metadata=None
        )
        kwargs_main = dict(
            photometric='rgb',
            planarconfig='CONTIG',
            compression='APERIO_JP2000_RGB',  # 改为JPEG2000压缩
            compressionargs={'level': 90},  # 压缩级别80
            dtype=np.uint8,
            metadata=None
        )
        kwargs_jpeg = dict(
            photometric='rgb',
            planarconfig='CONTIG',
            compression='jpeg',  # 普通JPEG压缩
            compressionargs={'level': 75},  # 质量75
            dtype=np.uint8,
            metadata=None
        )
        try:
            with tifffile.TiffWriter(save_svs_path, bigtiff=True) as tif:
                for i, hw in enumerate(multi_hw):
                    print(i,hw)
                    if i == 0:
                        for each_background_image in self.background_image:
                            gen = self.gen_tiles_whole(each_background_image, tile_hw[::-1])
                            tif.write(
                                data=gen,
                                subfiletype=0,
                                shape=(*hw, 3),
                                tile=tile_hw[::-1],
                                resolution=resolution,
                                resolutionunit=2,
                                description=svs_desc,
                                **kwargs_main  # 使用主图像压缩参数
                            )

                        tif.write(
                            data=thumbnail_im,
                            subfiletype=1,
                            description='Aperio Image Library Fake\nthumbnail',
                            resolution=resolution,
                            resolutionunit=2,
                            **kwargs_jpeg  # 缩略图使用JPEG压缩
                        )

                    else:
                        # print('i！=0，resize之前',i, hw)

                        self.background_image = [
                            self.process_large_image(each, resize_factor=0.5, block_size=block_size)
                            for each in self.background_image
                        ]
                        # print('i！=0',i, hw)
                        # print(len(self.background_image))
                        for each_background_image in self.background_image:
                            # print(each_background_image.shape)
                            gen = self.gen_tiles_whole(each_background_image, tile_hw[::-1])
                            # print('打印成功')
                            tif.write(
                                data=gen,
                                shape=(*hw, 3),
                                tile=tile_hw[::-1],
                                resolution=resolution,
                                resolutionunit=2,
                                description='',
                                **kwargs_main
                            )
                            # print('写入成功')
                label_im_x2 = cv2.resize(label_im, None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
                tif.write(
                    data=label_im_x2,
                    subfiletype=1,
                    description=label_desc.format(W=label_im_x2.shape[1], H=label_im_x2.shape[0]),
                    resolution=resolution,
                    resolutionunit=2,
                    **kwargs_jpeg
                )

                tif.write(
                    data=macro_im,
                    subfiletype=9,
                    description=macro_desc.format(W=macro_im.shape[1], H=macro_im.shape[0]),
                    resolution=resolution,
                    resolutionunit=2,
                    **kwargs_jpeg
                )

        except Exception:
            print('TiffWriter写入异常')
            print(traceback.format_exc())

        return save_svs_path


def print_slide_info(tiff_path):
    slide = openslide.OpenSlide(tiff_path)
    print(slide.level_dimensions)

    print("Level Count:", slide.level_count)
    for level in range(slide.level_count):
        print(
            f"Level {level} dimensions: {slide.level_dimensions[level]}, downsample: {slide.level_downsamples[level]}")
        # 获取该层级的尺寸
        # level_size = slide.level_dimensions[level]
        # 读取该层级的图像数据
        # region = slide.read_region((0, 0), level, level_size)
        # 将图像从 RGBA 转换为 RGB
        # region = region.convert('RGB')
        # image_np = np.array(region)
        # print(image_np.shape)
    print("Properties:")
    for key, value in slide.properties.items():
        print(f"  {key}: {value}")
    slide.close()


def extract_svs_and_save(input_svs_path):
    """
    Extract a full image from an SVS file and save it to a new SVS file.

    Args:
        input_svs_path: Path to the input SVS file
        output_svs_path: Path to save the output SVS file

    Returns:
        Path to the saved SVS file
    """
    # Import needed libraries inside the function
    import cv2

    # Make sure the output directory exists
    # os.makedirs(os.path.dirname(output_svs_path), exist_ok=True)

    # Open the SVS file
    slide = openslide.OpenSlide(input_svs_path)

    # Get dimensions of the highest resolution level (level 0)
    width, height = slide.level_dimensions[0]

    print(f"Image dimensions: {width}x{height}")
    print(f"Number of levels: {slide.level_count}")

    # Get the largest dimension to determine block size
    max_dim = max(width, height)
    if max_dim > 10000:
        # For very large images, use bigger blocks
        block_size = 4096
    else:
        block_size = 2048

    # Create an empty array to store the full image
    # Note: This could use a lot of memory for large images
    full_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Extract the image in blocks to avoid memory issues
    print("Extracting full image...")
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Calculate actual block size (may be smaller at edges)
            block_h = min(block_size, height - y)
            block_w = min(block_size, width - x)

            # Extract the region and convert from RGBA to RGB
            region = slide.read_region((x, y), 0, (block_w, block_h))
            region_rgb = region.convert('RGB')

            # Store in the full image array
            full_image[y:y + block_h, x:x + block_w] = np.array(region_rgb)

    # Close the slide
    slide.close()
    print("Full image extracted successfully")
    return full_image


def main(params):
    # DAPI_svs_path = r'H:\datasets_space\2025\scanner_datasets\fluorescence_datasets\自产荧光扫描三通道结果\数字切片003_DAPI_20250429_135237.svs'
    # SpGreen_svs_path = r'H:\datasets_space\2025\scanner_datasets\fluorescence_datasets\自产荧光扫描三通道结果\数字切片003_SpGreen_20250429_135350.svs'
    # SpOr_svs_path = r'H:\datasets_space\2025\scanner_datasets\fluorescence_datasets\自产荧光扫描三通道结果\数字切片003_SpOr_20250429_135408.svs'
    Filter_params = []
    # label_path = r'H:\datasets_space\2025\scanner_datasets\samples_20250417\DigitalSlide_20250512_200044\label.bmp'
    # macro_im_path = r'H:\datasets_space\2025\scanner_datasets\samples_20250417\DigitalSlide_20250512_200044\macro_im.bmp'
    # background_image = np.load('test_data/0.npy')
    # save_svs = 'pyramid_output/test'

    #
    # print(desc)
    # DAPI_background_image = extract_svs_and_save(DAPI_svs_path)
    # SpGreen_background_image = extract_svs_and_save(SpGreen_svs_path)
    # SpOr_background_image = extract_svs_and_save(SpOr_svs_path)
    #
    # gray_images = {}
    # png_dir = 'all_out_png'
    # # DAPI_background_image=cv2.imread('all_moving_png/AR391_1_CY5_out.png', cv2.IMREAD_GRAYSCALE)
    # for img_path in sorted(os.listdir(png_dir)):  # 确保顺序正确
    #     # if 'AR391_1' not in img_path and img_path.endswith('.png'):
    #     if img_path.endswith('.png'):
    #         img = cv2.imread(os.path.join(png_dir, img_path), cv2.IMREAD_GRAYSCALE)
    #         # gray_images.append(img)
    #         H, W = img.shape
    #         gray_images[img_path.split('_out1.png')[0]] = img
    #         print(f"Loaded: {img_path}, shape={img.shape}")
    import os
    import cv2
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 定义 PNG 文件夹路径
    png_dir = params['save_dir']
    txt_path=params['input_dir']
    # 读取 txt 文件，获取 DAPI 图像路径
    for txt_file in os.listdir(txt_path):
        if txt_file.endswith('.txt'):
            with open(os.path.join(txt_path, txt_file), 'r',encoding="latin1") as f:
                content = f.read()
                data_dict = ast.literal_eval(content)
                print(data_dict)
    def rename_channel(key, data_dict):
        parts = key.split("_", 2)  # 最多分割 2 次，避免被多余的 _ 打散
        parts = [parts[0] + "_" + parts[1], parts[2]]
        # print(parts)
        # print(data_dict)
        zb = data_dict[parts[0]][parts[1].lower()]
        return f"{parts[0].split('_')[1]}_{zb}"
    def rename_channel_zb(key, data_dict):
        if key=='DAPI':
            return 'DAPI'
        parts = key.split("_", 2)  # 最多分割 2 次，避免被多余的 _ 打散
        parts = [parts[0] + "_" + parts[1], parts[2]]
        # print(parts)
        # print(data_dict)
        zb = data_dict[parts[0]][parts[1].lower()]
        return f"{zb}"
    # 定义全局字典用于存储读取的图像
    gray_images = {}

    def load_image(img_path):
        """
        加载单个图像文件
        :param img_path: 图像文件路径
        :return: 图像文件名和图像数据
        """
        if os.path.basename(img_path).startswith('Maximum_DAPI'):
            key = img_path.split('.png')[0]
        else:
            key = img_path.split('_out1.png')[0]

        print(f"Loaded: {img_path}")
        if key.endswith('.png'):
            key=key.split('.png')[0]
        return key
    def gen_tiles_whole( data, tile_shape):
        for y in range(0, data.shape[0], tile_shape[0]):
            for x in range(0, data.shape[1], tile_shape[1]):
                yield data[y: y + tile_shape[0], x: x + tile_shape[1]]
    def process_large_image(img, resize_factor=1.0, block_size=1024):
        """
        Process a large image by dividing it into smaller blocks, resizing each block,
        and then reassembling the blocks.

        Args:
            img: Input image as numpy array
            resize_factor: Factor by which to resize the image (default: 1.0 = no resize)
            block_size: Size of blocks to process at a time

        Returns:
            Processed image as numpy array
        """
        height, width, c = img.shape

        # If no resize is needed, just return the original image
        if resize_factor == 1.0:
            return img

        # Create a large empty matrix for the result
        resized_height = int(height * resize_factor)
        resized_width = int(width * resize_factor)
        resized_img = np.zeros((resized_height, resized_width, c), dtype=img.dtype)

        # Process the image in blocks
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Get the current block
                block = img[y:min(y + block_size, height), x:min(x + block_size, width)]
                if block.shape[0] == 0 or block.shape[1] == 0:
                    continue

                # Resize the block
                block_resized = cv2.resize(block, None, fx=resize_factor, fy=resize_factor,
                                           interpolation=cv2.INTER_LINEAR)

                # Calculate new position
                new_y = int(y * resize_factor)
                new_x = int(x * resize_factor)

                # Place the resized block
                h, w = block_resized.shape[:2]
                h_remaining = min(h, resized_img.shape[0] - new_y)
                w_remaining = min(w, resized_img.shape[1] - new_x)
                resized_img[new_y:new_y + h_remaining, new_x:new_x + w_remaining] = block_resized[:h_remaining, :w_remaining]

        return resized_img
    keys=[]
    # 使用 13 个线程读取图像
    with ThreadPoolExecutor(max_workers=32) as executor:
        # 提交所有任务
        futures = [executor.submit(load_image, img_path) for img_path in sorted(os.listdir(png_dir)) if
                   img_path.endswith('.png') and '_DAPI_' not in img_path and 'Maximum_DAPI.png' not in img_path]

        # 异步获取结果
        for future in as_completed(futures):
            key = future.result()
            keys.append(key)
    from itertools import zip_longest

    # keys = list(gray_images.keys())
    groups = []
    print(keys)
    # 强制第一组为 ["Maximum_DAPI.png", None, None]
    # if "DAPI.png" in keys:
    #     groups.append((None, None,"DAPI"))
    #     keys.remove("Best_DAPI.png")  # 从剩余key中移除DAPI
    #

    # keys = sorted(
    #     (k for k in keys ),  # 过滤掉 DAPI
    #
    #     key=lambda x: rename_channel_zb(x, data_dict)  # 按映射名字排序
    # )
    if "DAPI" in keys:
        keys.remove("DAPI")  # 先从原列表移除
        keys.insert(0, "DAPI")  # 强制插入到第一位
    print(keys)
    # # 剩余key按3个一组分配
    for i in range(0, len(keys), 3):
        group = tuple(keys[i:i + 3] + [None] * (3 - len(keys[i:i + 3])))
        groups.append(group)

    # 打印检查分组
    print("调整后的分组:")
    # print(len(gray_images))
    N=len(groups)
    print(N)
    # background_image = np.zeros((N,) + (H, W, 3), dtype=np.uint8)
    with Image.open(os.path.join(png_dir, os.listdir(png_dir)[0])) as img:
        W, H = img.size
    background_image = np.memmap('temp_mmap.dat', dtype=np.uint8, mode='w+', shape=( H, W, 3))
    print(background_image.shape)
    #0层时存储4层，然后读取
    pseudo_colors = [
        '#00FF00',  # 绿色
        '#FF0000',  # 红色
        '#FFFF00',  # 黄色
        '#00FFFF',  # 青
        '#FFFFFF',  # 白色
        '#FFA500',  # 土黄色
        '#CE1DE0',  # 紫色
        '#F9A9DC',  # 粉红色
        '#009B62',  # 深绿色
        '#B4FFAF',  # 浅绿色

        '#5F32FF',  # 紫蓝色
        '#008080',  # 深青
        '#B8860B',  # 深金
        '#4682B4',  # 钢蓝
        '#DC143C',  # 猩红
        '#7FFF00',  # 黄绿
        '#D2691E',  # 巧克力
        '#000080',  # 海军蓝
        '#FF1493'  # 深粉
    ]

    none_counter = 1
    used_color_index = 0  # 用于给非 Maximum_DAPI 的 key 分配颜色（从前10个里优先选）
    width, height = 100, 100
    image = Image.new("RGB", (width, height), color=(255, 0, 0))  # 红色图像

    # 保存为 BMP 文件
    output_path = params['label_dir']
    image.save(output_path)

    print(f"成功创建 BMP 文件: {output_path}")
    width, height = 100, 100
    image = Image.new("RGB", (width, height), color=(255, 0, 0))  # 红色图像

    # 保存为 BMP 文件
    output_path = params['macro_dir']
    image.save(output_path)

    print(f"成功创建 BMP 文件: {output_path}")
    label_path = params['label_dir']
    macro_im_path = params['macro_dir']

    save_svs = params['save_svs_dir']
    # print(desc)
    bk_h, bk_w, bk_Channel_num = background_image.shape
    multi_hw = [[bk_h, bk_w], [bk_h // 2, bk_w // 2], [bk_h // 4, bk_w // 4], [bk_h // 8, bk_w // 8],
                [bk_h // 16, bk_w // 16]]

    def read_gray_image(png_dir, key):
        """读取单张灰度图"""
        if key is None:
            return None, None
        if key=='DAPI':
            path = os.path.join(png_dir, key + '.png')
        else:
            path=os.path.join(png_dir, key+'_out1.png')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return key, img

    def load_gray_images(png_dir, keys, num_threads=3):
        gray_images = {}
        # print(png_dir,keys)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(read_gray_image, png_dir, key)
                for key in keys
            ]
            for f in as_completed(futures):
                key, img = f.result()
                gray_images[key] = img
        return gray_images
    try:
        with tifffile.TiffWriter(save_svs, bigtiff=True) as tif:
            kwargs_main = dict(
                photometric='rgb',
                planarconfig='CONTIG',
                compression='APERIO_JP2000_RGB',  # 改为JPEG2000压缩
                compressionargs={'level': 90},  # 压缩级别80
                dtype=np.uint8,
                metadata=None
            )
            tile_hw = [512, 512]
            objective_rate = 1
            mpp = 0.2738 / objective_rate
            ppi = 25400 / mpp  # 每英寸多少像素
            mag = 20 * objective_rate
            filename = os.path.basename(save_svs)

            svs_desc = f'Aperio Image Library Fake\nABC |AppMag = {mag}|Filename = {filename}|MPP = {mpp}'

            resolution = (ppi, ppi)
            block_size = 2048
            kwargs_jpeg = dict(
                photometric='rgb',
                planarconfig='CONTIG',
                compression='jpeg',  # 普通JPEG压缩
                compressionargs={'level': 75},  # 质量75
                dtype=np.uint8,
                metadata=None
            )
            label_desc = 'Aperio Image Library Fake\nlabel {W}x{H}'
            macro_desc = 'Aperio Image Library Fake\nmacro {W}x{H}'


            label_im = np.array(Image.open(label_path).rotate(-180, expand=True))
            macro_im = np.array(Image.open(macro_im_path).rotate(-180, expand=True))
            im_true=True
            x=0
            x_end=len(groups)
            for group in groups:
                print("处理组:", group)
                c = 0
                for key in group:
                    if key == "DAPI":
                        pseudo_color = '#0000FF'  # 特殊处理
                    elif key is not None:
                        if used_color_index < 10:
                            pseudo_color = pseudo_colors[used_color_index]
                        else:
                            pseudo_color = pseudo_colors[used_color_index % len(pseudo_colors)]
                        used_color_index += 1
                    else:
                        pseudo_color = '#000000'

                    if key is not None:
                        # background_image[:, :, c] = gray_images[key]
                        Filter_params.append({
                            'Filter_name': 'DAPI' if key =='DAPI' else rename_channel(key, data_dict),
                            # 'Filter_name': key,
                            'Exposure_time': '5',
                            'Excitation_intensity': '100',
                            'Digital_gain': '3',
                            'Number_of_focus_level': None,
                            'Focus_step_size': None,
                            'Bit_depth_per_channel': '8',
                            'Pseudo_color': pseudo_color
                        })
                    else:
                        continue
            desc = {'Optical_parameters': {'Objective_name': 'Plan-Apochromat',
                                           'Objective_magnification': '20x',
                                           'Camera_adapter_magnification': '0.63x',
                                           'Camera_type': 'Basler acA2440-75ucMED',
                                           'Micrometer_per_pixel_X': '0.273810',
                                           'Micrometer_per_pixel_Y': '0.273810',
                                           'Output_resolution': '36.521676'},
                    'Scan_information': {
                        'Scan_camera': 'FL',
                        'Scan_method': 'Multi-channel Fusion',
                    },
                    'Fluorescence_parameters': {'Filter_params': Filter_params},

                    'Image_parameters': {'Image_compression': 'Yes',
                                         'Image_file_format': 'JPEG',
                                         'Quality_factor': '80'}}
            metadata_json = json.dumps(desc, ensure_ascii=True)
            svs_desc += f"|METADATA_JSON = {metadata_json}"
            print(svs_desc)
            for i, hw in enumerate(multi_hw):

                for group in groups:
                    print("处理组:", group)
                    c = 0
                    gray_images = load_gray_images(png_dir, group, num_threads=3)
                    # print(gray_images)
                    for key in group:
                        if key is not None:
                            background_image[ :, :, c] = gray_images[key]
                        else:
                            continue
                        c += 1
                    if i==0:
                        x += 1
                    print(i,hw)
                    if i == 0:
                        gen = gen_tiles_whole(background_image, tile_hw[::-1])
                        tif.write(
                            data=gen,
                            subfiletype=0,
                            shape=(*hw, 3),
                            tile=tile_hw[::-1],
                            resolution=resolution,
                            resolutionunit=2,
                            description=svs_desc,
                            **kwargs_main  # 使用主图像压缩参数
                        )
                        if im_true:
                            background_image_x4 = process_large_image(background_image, resize_factor=0.25,
                                                                      block_size=block_size)
                            scale_rate = max(background_image_x4.shape[0] // 512, 1)
                            thumbnail_im = cv2.resize(background_image_x4, None, None, fx=1 / scale_rate,
                                                      fy=1 / scale_rate,
                                                      interpolation=cv2.INTER_LANCZOS4)
                            tif.write(
                                data=thumbnail_im,
                                subfiletype=1,
                                description='Aperio Image Library Fake\nthumbnail',
                                resolution=resolution,
                                resolutionunit=2,
                                **kwargs_jpeg  # 缩略图使用JPEG压缩

                            )
                            im_true=False
                        else:
                            pass

                    else:
                        # print('i！=0，resize之前',i, hw)

                        background_image_1 =process_large_image(background_image, resize_factor=0.5**i, block_size=block_size)
                        # print('i！=0',i, hw)
                        # print(len(self.background_image))
                            # print(each_background_image.shape)
                        gen = gen_tiles_whole(background_image_1, tile_hw[::-1])
                        # print('打印成功')
                        tif.write(
                            data=gen,
                            shape=(*hw, 3),
                            tile=tile_hw[::-1],
                            resolution=resolution,
                            resolutionunit=2,
                            description='',
                            **kwargs_main
                        )
                            # print('写入成功')
                    background_image[:,:,:]=0
                    print('清空成功')
            label_im_x2 = cv2.resize(label_im, None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
            tif.write(
                data=label_im_x2,
                subfiletype=1,
                description=label_desc.format(W=label_im_x2.shape[1], H=label_im_x2.shape[0]),
                resolution=resolution,
                resolutionunit=2,
                **kwargs_jpeg
            )

            tif.write(
                data=macro_im,
                subfiletype=9,
                description=macro_desc.format(W=macro_im.shape[1], H=macro_im.shape[0]),
                resolution=resolution,
                resolutionunit=2,
                **kwargs_jpeg
            )
    except Exception:
            print('TiffWriter写入异常')
            print(traceback.format_exc())

    # img_obj = Image_Registration(desc, save_svs=save_svs, background_image=background_image, label_path=label_path,
    #                              macro_im_path=macro_im_path)
    # img_obj.gen_pyramid_tiff_for_fluorescence(save_svs_path=save_svs)
    #
    print_slide_info(save_svs)