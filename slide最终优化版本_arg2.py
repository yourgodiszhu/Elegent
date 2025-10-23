import multiprocessing
from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue as ThreadQueue  # 仅用于线程间通信，不用于进程间


import os

from PIL.FontFile import WIDTH

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

import cv2
import time
import numpy as np
import concurrent.futures
from typing import Tuple, List
from multiprocessing import Pool, cpu_count
from functools import partial
import re
import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import time
from openslide import OpenSlide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import os
import math
import threading
from queue import Queue
import concurrent.futures

OPENSLIDE_PATH = r'D:\3d\openslide_bin'
import os

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        from openslide import OpenSlide, ImageSlide
else:
    from openslide import OpenSlide, ImageSlide
import clr
print(clr.__version__)
import sys
import ctypes
from PIL import Image, ImageOps, ImageEnhance
import cv2

# 读取DLL文件
clr.FindAssembly("SlideAC.wrapper.dll")
dll = clr.AddReference("SlideAC.wrapper")
from SlideAC.wrapper import *


class Slide_SDK():
    def __init__(self, path):
        self.slide = TDHSlideAC()
        self.properties_dic = {}
        self.tile_width = 256
        self.tile_height = 256
        self.zoom_level_count = 10
        self.MinimalBoundingBoxesTile = []
        self.openslide(path)
        self._bitmap_refs = []  # 跟踪创建的位图对象

    def cleanup(self):
        """释放非托管资源"""
        for m_img in self._bitmap_refs:
            if hasattr(m_img, 'Dispose'):
                m_img.Dispose()
        self._bitmap_refs.clear()
        if hasattr(self.slide, 'CloseSlide'):
            self.slide.CloseSlide()
    def read_parameters(self):
        self.properties_dic.update(slide_type=self.slide.GetSlideProperties(320))
        self.properties_dic.update(slide_width_pixel=self.slide.GetSlideProperties(306))
        self.properties_dic.update(slide_height_pixel=self.slide.GetSlideProperties(307))
        self.properties_dic.update(bits_per_channel=self.slide.GetSlideProperties(327))
        self.properties_dic.update(channel_count=self.slide.GetSlideProperties(313))

        self.properties_dic.update(brightness=self.slide.GetSlideProperties(317))
        self.properties_dic.update(contrast=self.slide.GetSlideProperties(318))
        self.properties_dic.update(gamma=self.slide.GetSlideProperties(319))

        self.properties_dic.update(has_zstack=self.slide.GetSlideProperties(328))
        self.properties_dic.update(extend_focus_level=self.slide.GetSlideProperties(329))
        self.properties_dic.update(zstack_count=self.slide.GetSlideProperties(330))
        zstack_firstlevel = None
        try:
            zstack_firstlevel = self.slide.GetSlideProperties(370)
        except Exception:
            zstack_firstlevel = self.slide.GetSlideProperties(331)
        self.properties_dic.update(zstack_first_level=zstack_firstlevel)

        # channel
        self.properties_dic.update(channel_info={})
        self.properties_dic.update(channel_order=[])
        for c in range(self.properties_dic['channel_count']):
            channel_name = self.slide.GetChannelProperties(400, c)
            channel_bit_depth = self.slide.GetChannelProperties(409, c)
            self.properties_dic['channel_info'].update({str(channel_name): channel_bit_depth})
            self.properties_dic['channel_order'].append(str(channel_name))

        # zstack
        if self.properties_dic['has_zstack']:
            zstack_values_list = []
            for indx in range(self.properties_dic['zstack_count']):
                va = self.slide.GetSlideProperties(650 + self.properties_dic['zstack_first_level'] + indx)
                zstack_values_list.append(va)
            self.properties_dic.update(zstack_values=zstack_values_list)

        # check label areaimage
        self.properties_dic.update(has_label_area=self.slide.GetSingleImageProperties(100, 502))
        # print(self.properties_dic)

    def openslide(self, slide_path):
        if not os.path.exists(slide_path):
            raise Exception('error slide_path')
        self.slide.OpenSlide(slide_path)
        # original bit depth
        self.slide.set_Properties(207, True)
        self.read_parameters()

    def CalculateMinimalBoundingBoxesTile(self):
        for i in range(self.zoom_level_count):
            m = TDHBitmapImage()
            try:
                self.slide.GetScanMap(i, m)
            except:
                return

            i_min_row = m.Height - 1
            i_max_row = 0
            i_min_col = m.Width - 1
            i_max_col = 0

            ptr = m.LockBits()
            ptr_address = ptr.ToInt64()
            p_scan_map = ctypes.cast(ptr_address, ctypes.POINTER(ctypes.c_ubyte))
            for y in range(m.Height):
                for x in range(m.Width):
                    index = x + y * m.Stride
                    if p_scan_map[index] != 0:
                        if x < i_min_col:
                            i_min_col = x
                        if y < i_min_row:
                            i_min_row = y
                        if x > i_max_col:
                            i_max_col = x
                        if y > i_max_row:
                            i_max_row = y
            m_x = i_min_col
            m_y = i_min_row
            m_w = i_max_col - i_min_col + 1
            m_h = i_max_row - i_min_row + 1
            self.MinimalBoundingBoxesTile.append((m_x, m_y, m_w, m_h))
        # print(self.MinimalBoundingBoxesTile)

    def get_dst_address(self, z_tile_size, z_overlap, level, t_location):
        z_size = (self.properties_dic['slide_width_pixel'], self.properties_dic['slide_height_pixel'])
        z_dimensions = [z_size]
        while z_size[0] > 1 or z_size[1] > 1:
            z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
            z_dimensions.append(z_size)
        z_dimensions = tuple(reversed(z_dimensions))
        dz_levels = len(z_dimensions)

        dst_downsamples = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        dst_level_from_dzlevel = [9 if (dz_levels - dz_level - 1) > 9 else (dz_levels - dz_level - 1) for
                                  dz_level in range(dz_levels)]

        if level < 0 or level >= dz_levels:
            raise Exception('input level error')
        dst_level = dst_level_from_dzlevel[level]

        l0_z_downsamples: tuple[int, ...] = tuple(
            2 ** (dz_levels - dz_level - 1) for dz_level in range(dz_levels)
        )
        dst_z_downsamples = tuple(
            l0_z_downsamples[dz_level]
            / dst_downsamples[dst_level_from_dzlevel[dz_level]]
            for dz_level in range(dz_levels)
        )

        def tiles(z_lim: int) -> int:
            return int(math.ceil(z_lim / z_tile_size))

        t_dimensions = tuple((tiles(z_w), tiles(z_h)) for z_w, z_h in z_dimensions)

        if t_location[0] > t_dimensions[level][0] or t_location[1] > t_dimensions[level][1] or t_location[0] < 0 or \
                t_location[1] < 0:
            raise Exception('input t_location error')

        z_overlap_tl = tuple(z_overlap * int(t != 0) for t in t_location)
        z_overlap_br = tuple(
            z_overlap * int(t != t_lim - 1)
            for t, t_lim in zip(t_location, t_dimensions[level])
        )

        z_size = tuple(
            min(z_tile_size, z_lim - z_tile_size * t) + z_tl + z_br
            for t, z_lim, z_tl, z_br in zip(
                t_location, z_dimensions[level], z_overlap_tl, z_overlap_br
            )
        )

        z_location = [t * z_tile_size for t in t_location]
        l_location = [
            dst_z_downsamples[level] * (z - z_tl)
            for z, z_tl in zip(z_location, z_overlap_tl)
        ]

        dst_tile_size = 256
        x1, y1 = l_location
        x2, y2 = x1 + z_size[0], y1 + z_size[1]
        dst_x1, dst_y1 = int(x1 / dst_tile_size), int(y1 / dst_tile_size)
        dst_x2, dst_y2 = math.ceil(x2 / dst_tile_size), math.ceil(y2 / dst_tile_size)
        dst_x2, dst_y2 = min(dst_x2, math.ceil(self.properties_dic['slide_width_pixel'] / dst_tile_size)), \
            min(dst_y2, math.ceil(self.properties_dic['slide_height_pixel'] / dst_tile_size))
        off_x1 = int(x1 - dst_x1 * dst_tile_size)
        off_x2 = int(dst_x2 * dst_tile_size - x2)
        off_y1 = int(y1 - dst_y1 * dst_tile_size)
        off_y2 = int(dst_y2 * dst_tile_size - y2)

        return dst_level, (dst_x1, dst_y1, dst_x2, dst_y2), (off_x1, off_y1, off_x2, off_y2)

    def get_channel(self):
        return self.properties_dic['channel_order']

    def get_img(self, zlevel, rect_tile, channel_list, offset=(0, 0, 0, 0)):
        channel_idx = self.properties_dic['channel_order'].index(channel_list)  # 获取通道索引
        x1, y1, x2, y2 = rect_tile
        w_destpix = (x2 - x1) * self.tile_width - offset[0] - offset[2]
        h_destpix = (y2 - y1) * self.tile_height - offset[1] - offset[3]
        # img_muti = np.zeros((h_destpix, w_destpix, len(channel_indxs)), dtype=np.uint8)
        img_single = np.zeros((h_destpix, w_destpix), dtype=np.uint8)

        m_img = TDHBitmapImage()
        # for i in range(0, len(channel_indxs), 3):
        #     indices = channel_indxs[i:i + 3] + [-1] * (3 - len(channel_indxs[i:i + 3]))
        #     self.slide.GetImage(x1, y1, x2 - 1, y2 - 1, zlevel, indices[2], indices[1], indices[0], m_img)
        #
        #     ptr = m_img.LockBits().ToInt64() + offset[0] * 3 + offset[1] * m_img.Stride
        #     buffer = (ctypes.c_ubyte * (m_img.Stride * h_destpix)).from_address(ptr)
        #     np_src = np.frombuffer(buffer, dtype=np.uint8).reshape(h_destpix, m_img.Stride)
        #
        #     for k in range(min(3, len(channel_indxs) - i)):
        #         img_muti[:, :, i + k] = np_src[:, k::3][:, :w_destpix]
        # print(img_muti.shape)
        self.slide.GetImage(
            x1, y1, x2 - 1, y2 - 1,
            zlevel,
            -1, -1, channel_idx,  # 只请求目标通道（其余填充 -1）
            m_img
        )

        ptr = m_img.LockBits().ToInt64() + offset[0] + offset[1] * m_img.Stride
        buffer = (ctypes.c_ubyte * (m_img.Stride * h_destpix)).from_address(ptr)
        np_src = np.frombuffer(buffer, dtype=np.uint8).reshape(h_destpix, m_img.Stride)

        # 提取目标通道（假设数据顺序是 BGR 或类似）
        img_single[:, :] = np_src[:, 0::3][:, :w_destpix]  # 需根据实际数据顺序调整
        # print(img_single.shape)
        return img_single
    def get_color_img(self, img, color):
        res_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(img.shape[2]):
            color_img = Image.fromarray(img[:, :, i])
            color_img = ImageOps.colorize(color_img, black='black', white=color)
            res_img = np.clip(res_img + np.array(color_img), 0, 255)
        res_img = Image.fromarray(res_img)
        return res_img

    def get_img_from_dz(self, z_tile_size, z_overlap, level, t_location, channel_list):
        if len(channel_list) == 0:
            return None
        dst_level, dst_coord, offset = self.get_dst_address(z_tile_size, z_overlap, level, t_location)
        dst_img_np = self.get_img(dst_level, dst_coord, channel_list, offset)
        dst_img_np[dst_img_np > 0] = 255
        return dst_img_np

    def img_overview(self, level=5):
        channel_list = self.get_channel()
        self.CalculateMinimalBoundingBoxesTile()
        x, y, w, h = self.MinimalBoundingBoxesTile[level]
        x2, y2 = x + w, y + h
        img_np = self.get_img(level, (x, y, x2, y2), [channel_list[0]])
        img = self.get_color_img(img_np, channel_list)
        return img



def calculate_class_max_size(cls_dir):
    """计算每个分类中的最大宽度和高度"""
    max_width = 0
    max_height = 0
    sdk_list={}
    for target in cls_dir:
        sdk_data_path = target
        # if not os.path.isdir(sdk_data_path):
        #     continue
        sdk_list[target.split('\\')[-1]] = {}

        try:
            new_slide = Slide_SDK(sdk_data_path)
            channel_list = new_slide.get_channel()
            # sdk_list.append(channel_list)
            new_slide.CalculateMinimalBoundingBoxesTile()
            x, y, w, h = new_slide.MinimalBoundingBoxesTile[0]
            sdk_list[target.split('\\')[-1]]['slide']=sdk_data_path
            sdk_list[target.split('\\')[-1]]['channel_list']=channel_list
            sdk_list[target.split('\\')[-1]]['xywh']=x,y,w,h
            # 计算需要的图像尺寸
            width = (((max(range(x, x + w, 1)) - min(range(x, x + w, 1))) // 1) + 1) * 256
            height = (((max(range(y, y + h, 1)) - min(range(y, y + h, 1))) // 1) + 1) * 256

            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
        except Exception as e:
            print(f"Error processing {target}: {str(e)}")
            continue
    return  max_width,max_height,sdk_list


import gc


# 假设的Slide_SDK类
# class Slide_SDK:
#     def __init__(self, slide_path):
#         self.slide_path = slide_path
#         # 这里应该有实际的初始化代码

def split_ranges(total_length, chunk_size):
    ranges = []
    for start in range(0, total_length, chunk_size):
        end = start + chunk_size
        if end > total_length:
            end = total_length  # 最后一组调整到实际结尾
        ranges.append( (start, end) )
    return ranges
# 线程处理函数
def thread_worker(thread_queue, max_w, max_y, output_dir):
    # global max_w,max_y
    while True:
        task = thread_queue.get()
        if task is None:  # 终止信号
            break
        new_slide, p1_val, p2_val, channel, (x, y, w, h) = task
        # 这里处理你的线程任务
        # print(channel)

        # full_image = Image.new('L', ((((max(range(x, x + w, 8)) - min(range(x, x + w, 8))) // 8) + 1) * 2048,
        #                                (((max(range(y, y + h, 8)) - min(range(y, y + h, 8))) // 8) + 1) * 2048))
        # print("x,y,w,h",x,y,w,h)

        # full_image = np.zeros(((((max(range(y, y + h, 8)) - min(range(y, y + h, 8))) // 8) + 1) * 2048,
        #                        (((max(range(x, x + w, 8)) - min(range(x, x + w, 8))) // 8) + 1) * 2048),
        #                       dtype=np.uint8)
        full_image = np.zeros((max_y,max_w),dtype=np.uint8)
        # 使用numpy.memmap处理超大图像
        import tempfile

        h_size,w_size=full_image.shape
        x,y,w,h=x*256,y*256,w*256,h*256
        h=y+h
        w=x+w
        # print(x,y,w,h)
        # exit()

        step_size=2560
        for dy in tqdm.tqdm(range(y, h, step_size)):
            for dx in range(x, w, step_size):
                # 计算切割区域
                # print(dx)
                # print(dy)
                src_x_start = dx  # 直接使用dx，因为dx从x开始
                src_x_end = min(w, dx + step_size)
                src_y_start = dy  # 直接使用dy，同理
                src_y_end = min(h, dy + step_size)
                # 获取切割块
                # print(src_x_start,src_x_end,src_y_start,src_y_end)
                tile=new_slide.get_img(0, (src_x_start//256, src_y_start//256, src_x_end//256, src_y_end//256), channel)
                # 计算在背景图的粘贴位置
                bg_x = dx - x  # 从0开始计算
                bg_y = dy - y

                # 确保不超出背景图范围
                if bg_y >= full_image.shape[0] or bg_x >= full_image.shape[1]:
                    # print(bg_y)
                    # print(full_image.shape[0])
                    # print(bg_x)
                    # print(full_image.shape[1])
                    continue

                bg_x_end = min(bg_x + tile.shape[1], full_image.shape[1])
                bg_y_end = min(bg_y + tile.shape[0], full_image.shape[0])

                # 粘贴有效区域
                full_image[bg_y:bg_y_end, bg_x:bg_x_end] = tile[:bg_y_end - bg_y, :bg_x_end - bg_x]

        os.makedirs(fr'{output_dir}\{p2_val}\{p1_val}\{channel}',exist_ok=True)
        # full_image.save(fr"AR391_Gray_date\{p2_val}\{p1_val}\{channel}\{p1_val}_{channel}_merged_image.png")
        cv2.imwrite(rf"{output_dir}\{p2_val}\{p1_val}\{channel}\{p1_val}_{channel}_merged_image.png",full_image)

import tifffile


# 进程处理函数
def process_worker(process_queue, max_w, max_y, output_dir):
    # 每个进程创建自己的线程队列
    thread_queue = ThreadQueue()

    # 每个进程创建4个线程
    threads = []
    for _ in range(3):
        t = Thread(target=thread_worker, args=(thread_queue,max_w,max_y,output_dir))
        t.start()
        threads.append(t)

    while True:
        task = process_queue.get()
        if task is None:  # 终止信号
            break

        key, content = task
        slide_path = content['slide']
        channel_list = content['channel_list']
        xywh = content['xywh']
        p1 = slide_path.split("\\")[-1]
        p2 = slide_path.split("\\")[-2]
        # 初始化Slide_SDK
        new_slide = Slide_SDK(slide_path)
        # print(new_slide)
        # 将任务分配给线程
        for channel in channel_list:
            thread_queue.put((new_slide, p1, p2, channel, xywh))

    # 向线程发送终止信号
    for _ in range(3):
        thread_queue.put(None)

    # 等待所有线程结束
    for t in threads:
        t.join()


def main(params):
    """主逻辑（适配外部参数）"""
    start_time = time.time()
    print("开始处理所有图片...".center(50, '*'))

    # 从参数中读取路径
    cls_dir = params["input_dir"]
    output_dir = params["output_dir"]

    cls_path_list = [os.path.join(cls_dir, i) for i in os.listdir(cls_dir)
                     if os.path.isdir(os.path.join(cls_dir, i))]

    max_w, max_y, skd_dict = calculate_class_max_size(cls_path_list)
    print(f'计算最大值耗时: {time.time() - start_time:.2f}秒')
    print(f"最大尺寸: {max_w}x{max_y}")

    process_queue = multiprocessing.Queue()
    # if int(max_y)+int(max_w)<80000:
    #     max_workers=4
    # if int(max_y) + int(max_w) < 120000 and int(max_y) + int(max_w) >= 80000:
    #     max_workers=3
    # if int(max_y)+int(max_w)>=120000:
    #     max_workers=2

    max_workers=2
    processes = [Process(target=process_worker, args=(process_queue, max_w, max_y, output_dir))
                 for _ in range(max_workers)]
    for p in processes: p.start()

    for key, content in skd_dict.items():
        process_queue.put((key, content))

    for _ in range(max_workers): process_queue.put(None)
    for p in processes: p.join()

    print(f'总耗时: {time.time() - start_time:.2f}秒')