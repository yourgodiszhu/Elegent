from openslide import OpenSlide, ImageSlide
from PIL import Image, ImageEnhance
import numpy as np
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
import cv2
import traceback
import math
import json
from functools import lru_cache

class Data_Slide:
    def __init__(self, data_path, IMG=False, dz_tile_size=252, dz_overlap=2):
        try:
            self.format_type = ''
            if IMG:
                self.slide = ImageSlide(Image.open(data_path))
                self.format_type = 'IMG'
            else:
                self.slide = OpenSlide(data_path)
                self.format_type = 'SVS'
        except:
            try:
                self.slide = ImageSlide(Image.open(data_path))
                self.format_type = 'IMG'
            except :
                traceback.print_exc()
                raise Exception('Error in loading image')
        # data type
        self.data_type = 'openslide'
        # 添加缓存机制
        self._color_info_cache = None
        self._counts_per_level_cache = None
        self._max_level_cache = None
        self._channel_order_cache = None
        # 预计算颜色映射表
        self._color_lut_cache = {}

        # deepzoom相关的参数
        self.dz_tile_size = dz_tile_size
        self.dz_overlap = dz_overlap
        self.z_dimensions = None
        self.dz_levels = None
        self._precompute_deepzoom_params()

        # 预计算level映射表
        self._precomputed_level_mappings = {}
        self._precompute_level_mappings()

    def _precompute_deepzoom_params(self):
        """预计算所有DeepZoom参数"""
        # 计算金字塔各层尺寸
        z_size = self.slide.dimensions
        self.z_dimensions = [z_size]
        while z_size[0] > 1 or z_size[1] > 1:
            z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
            self.z_dimensions.append(z_size)
        self.z_dimensions = tuple(reversed(self.z_dimensions))
        self.dz_levels = len(self.z_dimensions)

    def _get_counts_per_level(self):
        """缓存每级level的计数"""
        if self._counts_per_level_cache is None:
            self._counts_per_level_cache = self.slide.level_downsamples.count(self.slide.level_downsamples[0])
        return self._counts_per_level_cache

    def _get_max_level(self):
        """缓存最大level"""
        if self._max_level_cache is None:
            self._max_level_cache = self.slide.level_count // self._get_counts_per_level() - 1
        return self._max_level_cache

    def _get_channel_order_list(self):
        """缓存通道顺序列表"""
        if self._channel_order_cache is None:
            color_info = self.get_color_info()
            self._channel_order_cache = list(color_info.keys())
        return self._channel_order_cache

    def _precompute_level_mappings(self):
        """预计算level映射关系"""

        counts_per_level = self._get_counts_per_level()
        channel_order_list = self._get_channel_order_list()

        # 下采样次数
        ds_levels = len(self.slide.level_dimensions)//counts_per_level
        for l in range(ds_levels):
            self._precomputed_level_mappings[l] = self._get_channel_mapping_for_level(l, channel_order_list, counts_per_level)

    def _get_channel_mapping_for_level(self, level, channel_order_list, counts_per_level):
        """预计算某个level的通道映射"""
        mapping = {}
        for channel in channel_order_list:
            index = channel_order_list.index(channel)
            dst_level_0 = index // 3
            rgb_pos = index % 3
            dst_level = level * counts_per_level + dst_level_0
            mapping[channel] = (dst_level, rgb_pos)
        return mapping

    @lru_cache(maxsize=64)
    def _get_lut_cached(self, black, white, gamma):
        """缓存 LUT，避免重复计算"""
        table = np.arange(256, dtype=np.float32)
        table = np.clip((table - black) / max(1, (white - black)), 0, 1) * 255
        real_gamma = 2 - gamma
        table = ((table / 255.0) ** real_gamma) * 255
        return table.astype(np.uint8)


    def _get_enhanced_color_lut(self, black, white, gamma, color_name):
        """获取增强后的颜色查找表，一次性完成增强和颜色映射"""
        enhance_lut = self._get_lut_cached(black, white, gamma)

        # 获取颜色LUT
        color_lut = self._get_color_lut(color_name)

        combined_lut = color_lut[enhance_lut]
        return combined_lut

    def apply_enhancement_and_color_fast(self, gray_img, black, white, gamma, color_name):
        """一次性完成图像增强和颜色映射"""
        if black == 0 and white == 255 and gamma == 1:
            # 无增强时直接使用颜色映射
            return self._apply_color_fast(gray_img, color_name)

        # 获取组合LUT
        combined_lut = self._get_enhanced_color_lut(black, white, gamma, color_name)

        # 应用组合LUT
        if isinstance(gray_img, Image.Image):
            gray_array = np.array(gray_img)
        else:
            gray_array = gray_img

        # 确保是2D数组
        if len(gray_array.shape) == 3:
            gray_array = gray_array[:, :, 0] if gray_array.shape[2] > 0 else gray_array.squeeze()

        return combined_lut[gray_array]

    def color_enhancement_fast(self, mat, black, white, gamma):
        """更快的图像增强：查缓存的 LUT"""
        lut = self._get_lut_cached(black, white, gamma)
        return cv2.LUT(mat, lut)

    def _get_color_lut(self, color_name):
        """获取或创建颜色查找表，支持6位和8位十六进制，但只使用RGB部分"""
        if color_name not in self._color_lut_cache:
            # 创建从黑色到指定颜色的渐变查找表
            try:
                # 尝试解析颜色名称
                if color_name.startswith('#'):
                    # 十六进制颜色，支持6位(#RRGGBB)和9位(#AARRGGBB)，但只取RGB
                    if len(color_name) == 7:  # #RRGGBB
                        color_rgb = tuple(int(color_name[i:i + 2], 16) for i in (1, 3, 5))
                    elif len(color_name) == 9:  # #AARRGGBB，前两位是Alpha，后6位是RGB
                        # 跳过Alpha通道(前两位)，提取RGB部分
                        color_rgb = tuple(int(color_name[i:i + 2], 16) for i in (3, 5, 7))
                        # 忽略Alpha通道 (color_name[1:3])
                    else:
                        raise ValueError(f"Invalid hex color format: {color_name}")
                else:
                    # 颜色名称，使用PIL转换
                    temp_img = Image.new('RGB', (1, 1), color_name)
                    color_rgb = temp_img.getpixel((0, 0))
                # print(f"颜色 {color_name} -> RGB: {color_rgb}")
                # 创建256级渐变查找表，始终为RGB格式
                lut = np.zeros((256, 3), dtype=np.uint8)
                for i in range(256):
                    intensity = i / 255.0
                    lut[i] = [int(c * intensity) for c in color_rgb]

                self._color_lut_cache[color_name] = lut
            except Exception as e:
                print(f"颜色解析失败 {color_name}: {e}")
                # 如果颜色解析失败，使用灰度
                lut = np.zeros((256, 3), dtype=np.uint8)
                for i in range(256):
                    lut[i] = [i, i, i]  # 灰度
                self._color_lut_cache[color_name] = lut

        return self._color_lut_cache[color_name]

    def _apply_color_fast(self, gray_img, color_name):
        """快速颜色应用，使用查找表而不是ImageOps.colorize"""
        if isinstance(gray_img, Image.Image):
            gray_array = np.array(gray_img)
        else:
            gray_array = gray_img

        # 确保是2D数组
        if len(gray_array.shape) == 3:
            gray_array = gray_array[:, :, 0] if gray_array.shape[2] > 0 else gray_array.squeeze()

        # 使用查找表进行快速颜色映射
        color_lut = self._get_color_lut(color_name)
        colored = color_lut[gray_array]
        return colored

    def get_region(self, start_p_x, start_p_y, level, size, img_type='IHC'):
        """获得当前起始点下，当前level下，大小为size的切片
        """
        real_level, resize_fact = self.get_real_level(level)

        # 读取区域，包括alpha通道
        region_data = np.array(self.slide.read_region(
            location=(int(start_p_x), int(start_p_y)), level=real_level,
            size=(int(size[0] * resize_fact), int(size[1] * resize_fact))))

        # 直接将alpha通道转换为3通道RGB图像，保留背景
        # 提取RGB和alpha通道
        rgb = region_data[:, :, :3]
        alpha = region_data[:, :, 3]
        if img_type == 'IHC':
            # 创建纯白背景
            bg = np.ones_like(rgb) * 255
        else:
            bg = np.zeros_like(rgb)
        # 根据alpha通道进行混合
        # 将alpha通道从0-255转换为0-1的浮点数作为权重
        alpha = alpha[:, :, np.newaxis] / 255.0

        # 使用alpha混合原图与白色背景
        result = rgb * alpha + bg * (1 - alpha)

        # 调整大小并返回
        im = cv2.resize(result.astype(np.uint8), (size[0], size[1]))
        return im

    def get_level_dimension(self, level):
        """获得指定level的大小
        """
        real_level, resize_fact = self.get_real_level(level)
        [w, h] = self.slide.level_dimensions[real_level]
        w, h = int(w / resize_fact), int(h / resize_fact)
        return [w, h]

    def get_level_count(self):
        """总共有多少个level
        """
        return self.slide.level_count

    def get_real_level(self, level):
        """
        输入的level所代表的的降采样等级在当前的数据中不一定存在
        给level间的放大倍数不为2的数据插入对应的放大倍数
        :param level:
        :return:
        """
        # 输入level对应的降采样倍数
        down_sample_factor = pow(2, level)
        # 当前降采样倍数对应的该数据所拥有的真实level
        real_level = self.slide.get_best_level_for_downsample(down_sample_factor)
        # 输入的level和实际的level之间的倍率
        resize_fact = pow(2, level - real_level)
        return real_level, resize_fact

    def get_pix_area_in_level(self, level):
        real_level, resize_fact = self.get_real_level(level)
        openslide_mpp_x = self.slide.properties.get('openslide.mpp-x', '0.273809523809524')
        openslide_mpp_y = self.slide.properties.get('openslide.mpp-y', '0.273809523809524')
        # openslide_mpp_x = self.slide.properties['openslide.mpp-x']
        # openslide_mpp_y = self.slide.properties['openslide.mpp-y']
        x = float(openslide_mpp_x) * pow(2, real_level) * resize_fact
        y = float(openslide_mpp_y) * pow(2, real_level) * resize_fact
        # 单位为平方微米
        return x * y / 1000000

    def get_pix_direct_in_level(self, level):
        # 每个像素的多少微米
        real_level, resize_fact = self.get_real_level(level)
        openslide_mpp_x = self.slide.properties.get('openslide.mpp-x', '0.273809523809524')
        x = float(openslide_mpp_x) * pow(2, real_level) * resize_fact
        return x


    def loop_get_patches(self, level=1, size=512, stride=256, region=None, check_foreground=False):
        """一般来说在训练的时候一般会使用level=1，patch大小512，stride256，用这样的方式可以循环的获取patch
        该函数会生成一个generator，然后方便循环的调用该函数进行输出。
        另外这个region的尺寸为level0上的尺寸
        返回的sh和sw为在level0中的左上角坐标
        region: xmin, ymin, xmax, ymax
        """
        real_level, resize_fact = self.get_real_level(level)
        real_size = size * resize_fact
        if region is None:
            [w, h] = self.slide.level_dimensions[real_size]
            # xmin, ymin, xmax, ymax
            region = (0, 0, w, h)
        ratio = np.power(2, real_level)
        step = stride * ratio * resize_fact
        # 计算水平和垂直方向的patch数量
        h_patches = (region[2] - region[0] + step - 1) // step  # 向上取整
        v_patches = (region[3] - region[1] + step - 1) // step  # 向上取整
        total_patch_num = h_patches * v_patches
        cur_patch_num = 0
        for sh in range(region[1], region[3], step):
            for sw in range(region[0], region[2], step):
                cur_patch_num += 1
                patch = cv2.resize(
                    np.array(self.slide.read_region(location=(sw, sh), level=real_level, size=(real_size, real_size)))[
                    :, :, :3], (size, size))
                yield patch, sw, sh, (cur_patch_num, total_patch_num), None

    def loop_get_positions(self, level=1, size=512, stride=256, region=None):
        """与loop_get_patches类似，但只返回坐标位置，不加载图像
        返回level 0中的坐标位置(sw, sh)

        参数与loop_get_patches相同
        """
        real_level, resize_fact = self.get_real_level(level)
        if region is None:
            [w, h] = self.slide.level_dimensions[real_level]
            # xmin, ymin, xmax, ymax
            region = (0, 0, w, h)
        ratio = np.power(2, real_level)
        step = stride * ratio * resize_fact
        for sh in range(region[1], region[3], step):
            for sw in range(region[0], region[2], step):
                yield sw, sh

    def get_slide_type(self):
        slide_type = 'OTHER'
        if 'mirax.GENERAL.SLIDE_TYPE' in self.slide.properties.keys():
            slide_type = self.slide.properties['mirax.GENERAL.SLIDE_TYPE']
        return slide_type

    def check_level_scale(self):
        """检查从0到7层每层的倍率是否为2倍
        Returns:
            bool: 如果所有层级间的倍率都是2倍(允许四舍五入误差)则返回True，否则返回False
        """
        try:
            prev_dim = None
            for level in range(8):
                curr_dim = self.get_level_dimension(level)
                # print(f"Level {level}: {curr_dim}")
                if prev_dim is not None:
                    # 检查当前层与上一层的差值是否在±1像素范围内
                    expected_w = curr_dim[0] * 2
                    expected_h = curr_dim[1] * 2
                    if abs(prev_dim[0] - expected_w) > 1 or abs(prev_dim[1] - expected_h) > 1:
                        return 0

                prev_dim = curr_dim
            return 1
        except:
            return 0


    def get_img(self, level, x, y, w, h, channel_dict, is_gray=False, color_info={}):
        """
        :param level: 层数
        :param x,y,w,h: level0的x,y，level下的w,h
        :param channel_dict: 获取的颜色字典，格式为 {channel_name: color}
        :param is_gray: 是否返回灰度图
        :param color_info: 颜色增强参数
        :return: PIL
        """

        if x < 0 or y < 0 or w <= 0 or h <= 0 or x >= self.slide.level_dimensions[0][0] or y >= \
                self.slide.level_dimensions[0][1]:
            raise ValueError(f"Invalid input, x:{x},y:{y},w:{w},h:{h}")
        if level > math.floor(math.log2(min(self.slide.dimensions))) or level < 0:
            raise ValueError(f"Invalid input, level:{level}")
        if len(channel_dict) == 0:
            return None

        # 计算每一级level有多少个rgb图
        counts_per_level = self._get_counts_per_level()
        max_level = self._get_max_level()
        # print(level)
        # print(math.floor(math.log2(min(self.slide.dimensions))))
        # print(counts_per_level)
        # print(max_level)
        # 若level超出slide.level的范围，先读取最小的max_level层，再resize
        if level > max_level:
            # 计算先读取最小的max_level层的w，h
            max_level_w, max_level_h = self.slide.level_dimensions[-1]
            l = level - max_level
            # 计算x,y在max_level对应的x，y
            dst_x, dst_y = int(x / 2 ** max_level), int(y / 2 ** max_level)
            dst_w, dst_h = w * 2 ** l, h * 2 ** l
            if dst_x == max_level_w or dst_y == max_level_h:
                raise ValueError(f'x:{x},y:{y} is on the boundary in level {max_level}')

            # 计算要读取的max_level的w,h，若越界则限制在范围内，最后计算level层的w，h
            dst_w, dst_h = min(dst_w, max_level_w - dst_x), min(dst_h, max_level_h - dst_y)
            w, h = int(dst_w / 2 ** l), int(dst_h / 2 ** l)
            dst_size = (dst_w, dst_h)

            level = max_level
            resize_flag = True
        else:
            max_dst_w, max_dst_h = self.slide.level_dimensions[level * counts_per_level][0], \
                self.slide.level_dimensions[level * counts_per_level][1]
            # 计算x,y在level上对应的坐标，用于判断w，h是否越界
            dst_x, dst_y = int(x // 2 ** level), int(y // 2 ** level)

            if dst_x == max_dst_w or dst_y == max_dst_h:
                raise ValueError(f'x:{x},y:{y} is on the boundary in level {level}')

            w = min(w, max_dst_w - dst_x)
            h = min(h, max_dst_h - dst_y)
            dst_size = (w, h)
            resize_flag = False

        # 统计通道信息
        channel_list = list(channel_dict.keys())
        channel_count = len(channel_list)

        read_level = {}
        for i in range(channel_count):
            channel = channel_list[i]
            dst_level, rgb_pos = self._precomputed_level_mappings[level][channel]
            read_level.setdefault(dst_level, []).append((i, rgb_pos))

        if is_gray:
            scale = 1 / channel_count
            dst_img = np.zeros((h, w), dtype='uint8')
            for dst_level, ch_info_list in read_level.items():
                region = self.slide.read_region((x, y), dst_level, dst_size)
                image_np = np.asarray(region)[:, :, :3]
                if resize_flag:
                    image_np = cv2.resize(image_np, (w, h), interpolation=cv2.INTER_LINEAR)
                for output_idx, rgb_pos in ch_info_list:
                    gray_img = image_np[:, :, rgb_pos]
                    # 获取black,white,gamma
                    param_dict = color_info.get(channel_list[output_idx], {})
                    black = param_dict.get('black', 0)
                    white = param_dict.get('white', 255)
                    gamma = param_dict.get('gamma', 1)
                    # 图像增强
                    gray_img = self.color_enhancement_fast(gray_img, black, white, gamma)
                    dst_img = cv2.add(dst_img, np.array(gray_img * scale, np.uint8))
            # dst_img = Image.fromarray(dst_img)
            dst_img = ImageEnhance.Brightness(Image.fromarray(dst_img)).enhance(channel_count)
        else:
            # 读取并使用自定义颜色上色
            dst_img = np.zeros((h, w, 3), dtype='uint8')
            for dst_level, ch_info_list in read_level.items():
                region = self.slide.read_region((x, y), dst_level, dst_size)
                image_np = np.asarray(region)[:, :, :3]
                if resize_flag:
                    image_np = cv2.resize(image_np, (w, h), interpolation=cv2.INTER_LINEAR)
                for output_idx, rgb_pos in ch_info_list:
                    gray_img = image_np[:, :, rgb_pos]
                    # 获取black,white,gamma
                    param_dict = color_info.get(channel_list[output_idx], {})
                    black = param_dict.get('black', 0)
                    white = param_dict.get('white', 255)
                    gamma = param_dict.get('gamma', 1)
                    # 图像增强
                    custom_color = channel_dict[channel_list[output_idx]]
                    colored_img = self.apply_enhancement_and_color_fast(gray_img, black, white, gamma, custom_color)
                    dst_img = cv2.add(dst_img, colored_img)

            dst_img = Image.fromarray(dst_img)
        return dst_img

    def get_tile(self, dz_level, address, channel_dict, is_gray=False, color_info={}):
        """
        :param dz_level: deepzoom get_tile 对应的level
        :param address: deepzoom get_tile 对应的address
        :param channel_dict: 获取的颜色字典，格式为 {channel_name: color}
        :return: PIL
        """

        if dz_level < 0 or dz_level >= self.dz_levels:
            raise ValueError(f"Invalid level {dz_level}, level should between 0 and {self.dz_levels - 1}")

        if len(channel_dict) == 0:
            return None

        # 求对应的level，即下采样次数
        level = self.dz_levels - dz_level - 1

        z_dimension = self.z_dimensions[dz_level]
        t_dimension = math.ceil(z_dimension[0] / self.dz_tile_size), math.ceil(z_dimension[1] / self.dz_tile_size)

        # 判断输入adress是否超出范围
        for t, t_lim in zip(address, t_dimension):
            if t < 0 or t >= t_lim:
                raise ValueError(f"Invalid address: {address}")

        # 计算左上右下的偏移
        z_overlap_tl = tuple(self.dz_overlap * int(t != 0) for t in address)
        z_overlap_br = tuple(
            self.dz_overlap * int(t != t_lim - 1)
            for t, t_lim in zip(address, t_dimension)
        )

        # 得到tile的大小
        tile_size = tuple(
            min(self.dz_tile_size, z_lim - self.dz_tile_size * t) + z_tl + z_br
            for t, z_lim, z_tl, z_br in zip(address, z_dimension, z_overlap_tl, z_overlap_br))
        # x,y,w,h
        x = self.dz_tile_size * address[0] - z_overlap_tl[0]
        y = self.dz_tile_size * address[1] - z_overlap_tl[1]
        w, h = tile_size[0], tile_size[1]

        # 计算每个level对应svs中多少个level
        max_level = self._get_max_level()

        # 若level超出slide.level的范围，先读取最小的max_level
        if level > max_level:
            l = level - max_level
            # max_level对应的size和level0对应的xy
            dst_size = (int(w * 2 ** l), int(h * 2 ** l))
            x, y = x * (2 ** level), y * (2 ** level)
            level = max_level
            resize_flag = True
        else:
            resize_flag = False
            x, y = x * (2 ** level), y * (2 ** level)
            dst_size = (w, h)

        # 统计通道信息
        channel_list = list(channel_dict.keys())
        channel_count = len(channel_list)

        read_level = {}
        for i in range(channel_count):
            channel = channel_list[i]
            dst_level, rgb_pos = self._precomputed_level_mappings[level][channel]
            read_level.setdefault(dst_level, []).append((i, rgb_pos))

        if is_gray:
            scale = 1 / channel_count
            dst_img = np.zeros((h, w), dtype='uint8')
            for dst_level, ch_info_list in read_level.items():
                region = self.slide.read_region((x, y), dst_level, dst_size)
                image_np = np.asarray(region)[:, :, :3]
                if resize_flag:
                    image_np = cv2.resize(image_np, (w, h), interpolation=cv2.INTER_LINEAR)

                for output_idx, rgb_pos in ch_info_list:
                    gray_img = image_np[:, :, rgb_pos]
                    # 获取black,white,gamma
                    # param_dict = color_info.get(channel_list[output_idx], {})
                    # black = param_dict.get('black', 0)
                    # white = param_dict.get('white', 255)
                    # gamma = param_dict.get('gamma', 1)
                    # # 图像增强
                    # gray_img = self.color_enhancement_fast(gray_img, black, white, gamma)

                    # dst_img = cv2.add(dst_img, np.array(gray_img * scale, np.uint8))
            # dst_img = ImageEnhance.Brightness(Image.fromarray(dst_img)).enhance(channel_count)
        else:
            # 读取并使用自定义颜色上色
            dst_img = np.zeros((h, w, 3), dtype='uint8')
            for dst_level, ch_info_list in read_level.items():
                region = self.slide.read_region((x, y), dst_level, dst_size)
                image_np = np.asarray(region)[:, :, :3]
                if resize_flag:
                    image_np = cv2.resize(image_np, (w, h), interpolation=cv2.INTER_LINEAR)
                for output_idx, rgb_pos in ch_info_list:
                    gray_img = image_np[:, :, rgb_pos]
                    # 获取black,white,gamma
                    param_dict = color_info.get(channel_list[output_idx], {})
                    black = param_dict.get('black', 0)
                    white = param_dict.get('white', 255)
                    gamma = param_dict.get('gamma', 1)

                    # 增强和上色
                    custom_color = channel_dict[channel_list[output_idx]]
                    colored_img = self.apply_enhancement_and_color_fast(gray_img, black, white, gamma, custom_color)
                    dst_img = cv2.add(dst_img, colored_img)
                    # dst_img = np.clip(dst_img.astype(np.uint16) + colored_img.astype(np.uint16), 0, 255).astype(
                    #     np.uint8)

            dst_img = Image.fromarray(dst_img)
        return dst_img

    def get_color_info(self):
        # 颜色通道名称和对应颜色 - 添加缓存
        if self._color_info_cache is None:
            data_info = json.loads(self.slide.properties['openslide.comment'].split('|METADATA_JSON =')[-1])
            color_dic = {}
            for p in data_info['Fluorescence_parameters']['Filter_params']:
                color_dic[p['Filter_name']] = p['Pseudo_color']
            self._color_info_cache = color_dic
        return self._color_info_cache


    def get_img_svs_np(self, level, x, y, w, h, channel_dict, color_info={}):
        """
        :param level: level下采样次数
        :param x,y,w,h: level0的x,y，level下的w,h
        :param channel_dict: 获取的颜色字典，格式为 {channel_name: color}
        :return: numpy
        """

        if x < 0 or y < 0 or w <= 0 or h <= 0 or x >= self.slide.level_dimensions[0][0] or y >= \
                self.slide.level_dimensions[0][1]:
            raise ValueError(f"Invalid input, x:{x},y:{y},w:{w},h:{h}")
        if level > math.floor(math.log2(min(self.slide.dimensions))) or level < 0:
            raise ValueError(f"Invalid input, level:{level}")

        # 计算每一级level有多少个rgb图
        counts_per_level = self._get_counts_per_level()
        max_level = self._get_max_level()

        # 若level超出slide.level的范围，先读取最小的max_level层，再resize
        if level > max_level:
            # 计算先读取最小的max_level层的w，h
            max_level_w, max_level_h = self.slide.level_dimensions[-1]
            l = level - max_level
            # 计算x,y在max_level对应的x，y
            dst_x, dst_y = int(x / 2 ** max_level), int(y / 2 ** max_level)
            dst_w, dst_h = w * 2 ** l, h * 2 ** l
            if dst_x == max_level_w or dst_y == max_level_h:
                raise ValueError(f'x:{x},y:{y} is on the boundary in level {max_level}')

            # 计算要读取的max_level的w,h，若越界则限制在范围内，最后计算level层的w，h
            dst_w, dst_h = min(dst_w, max_level_w - dst_x), min(dst_h, max_level_h - dst_y)
            w, h = int(dst_w / 2 ** l), int(dst_h / 2 ** l)

            dst_size = (dst_w, dst_h)

            level = max_level
            resize_flag = True
        else:
            max_dst_w, max_dst_h = self.slide.level_dimensions[level * counts_per_level][0], \
                self.slide.level_dimensions[level * counts_per_level][1]
            # 计算x,y在level上对应的坐标，用于判断w，h是否越界
            dst_x, dst_y = int(x // 2 ** level), int(y // 2 ** level)

            if dst_x == max_dst_w or dst_y == max_dst_h:
                raise ValueError(f'x:{x},y:{y} is on the boundary in level {level}')

            w = min(w, max_dst_w - dst_x)
            h = min(h, max_dst_h - dst_y)
            dst_size = (w, h)
            resize_flag = False

        # 统计通道信息
        channel_list = list(channel_dict.keys())
        channel_count = len(channel_list)

        read_level = {}
        for i in range(channel_count):
            channel = channel_list[i]
            dst_level, rgb_pos = self._precomputed_level_mappings[level][channel]
            read_level.setdefault(dst_level, []).append((i, rgb_pos))

        # 创建多通道numpy数组
        dst_img = np.zeros((h, w, channel_count), dtype='uint8')

        for dst_level, ch_info_list in read_level.items():
            region = self.slide.read_region((x, y), dst_level, dst_size)
            image_np = np.asarray(region)[:, :, :3]
            if resize_flag:
                image_np = cv2.resize(image_np, (w, h), interpolation=cv2.INTER_LINEAR)
            for output_idx, rgb_pos in ch_info_list:
                # 直接存储原始通道数据
                gray_img = image_np[:, :, rgb_pos]
                # 获取black,white,gamma
                # param_dict = color_info.get(channel_list[output_idx], {})
                # black = param_dict.get('black', 0)
                # white = param_dict.get('white', 255)
                # gamma = param_dict.get('gamma', 1)
                # 图像增强
                # dst_img[:, :, output_idx] = self.color_enhancement_fast(gray_img, black, white, gamma)
                dst_img[:, :, output_idx] = gray_img
        return dst_img

    def get_svs_dimensions(self, level):
        """
        获取svs的level维度
        """
        if level > math.floor(math.log2(min(self.slide.dimensions))) or level < 0:
            raise ValueError(f"Invalid input, level:{level}")
        counts_per_level = self._get_counts_per_level()
        d = self.slide.level_dimensions[level * counts_per_level]
        return d


def print_properties(tiff_path):
    slide = OpenSlide(tiff_path)
    print("Properties:")
    for key, value in slide.properties.items():
        print(f"  {key}: {value}")
    slide.close()
import concurrent.futures

def fast_get_img_svs_np(slide, x, y, w, h, level, tile_size=2048):

    dst_img = np.zeros((h, w,3), dtype=np.uint8)

    # 分块
    y_tiles = (h + tile_size - 1) // tile_size
    x_tiles = (w + tile_size - 1) // tile_size

    def read_tile(tx, ty):
        x0 = x + tx * tile_size
        y0 = y + ty * tile_size
        w0 = min(tile_size, x + w - x0)
        h0 = min(tile_size, y + h - y0)
        region = slide.read_region((x0, y0), level, (w0, h0))
        image_np = np.array(region)[:, :]
        return tx, ty, image_np

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for ty in range(y_tiles):
            for tx in range(x_tiles):
                futures.append(executor.submit(read_tile, tx, ty))
        for f in concurrent.futures.as_completed(futures):
            tx, ty, tile_np = f.result()
            x0 = tx * tile_size
            y0 = ty * tile_size
            w0, h0 = tile_np.shape[1], tile_np.shape[0]
            # 直接填入 dst_img 对应通道
            # for idx, channel in enumerate(channel_list):
            #     rgb_pos = idx  # 如果你的 mapping 复杂可以替换
            dst_img[y0:y0 + h0, x0:x0 + w0,:] = tile_np[:, :, :3]

    return dst_img
if __name__ == '__main__':
    data_path = r"C:\Users\Administrator\Downloads\Filez\WebTool\20251017 郑州17 自产荧光LG2  魏榕 1张荧光单标+1张明场扫描  路子畅#SWR02479837\MATE1-R0-2 IAA.svs"
    import time
    data_slide = Data_Slide(data_path, IMG=False, dz_tile_size=252, dz_overlap=2)

    color_dict=data_slide.get_color_info()
    print(color_dict)
    # print(data_slide.slide.dimensions)
    # img=data_slide.get_img( 0, 0, 0, data_slide.slide.dimensions[0], data_slide.slide.dimensions[1],
    #                         color_dict, is_gray=True, color_info={})
    # # s1=time.time()
    img = data_slide.get_img_svs_np(0, 0, 0, data_slide.slide.dimensions[0], data_slide.slide.dimensions[1],
                             color_dict)
    # print('优化前大图',time.time()-s1)
    s=time.time()
    # img=fast_get_img_svs_np(data_slide.slide, 0, 0, data_slide.slide.dimensions[0], data_slide.slide.dimensions[1],0)
    # print(img.shape)
    # cv2.imwrite('DAPI.png',img[:,:,0])
    # cv2.imwrite('IF488.png',img[:,:,1])
    print(time.time()-s)

    # img.save('test.png')