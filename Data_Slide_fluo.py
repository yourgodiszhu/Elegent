from openslide import OpenSlide, ImageSlide
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
import cv2
import traceback
import math
import json
# from configurations import MRXS_DEVICE_LIST

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
            result = np.ones_like(rgb) * 255
        else:
            result = np.zeros_like(rgb)
        # 找出不透明区域（alpha == 255），仅复制这些像素
        opaque_mask = alpha == 255
        result[opaque_mask] = rgb[opaque_mask]

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
    def get_hard_id_type(self):
        """检查扫描仪硬件ID是否在支持的设备列表中
        
        Returns:
            bool: 如果硬件ID在支持的设备列表中返回True，否则返回False
        """
        # 需要检查的硬件ID属性列表，按优先级排序
        hardware_id_keys = [
            'mirax.NONHIERLAYER_1_SECTION.SCANNER_HARDWARE_ID',
            'mirax.NONHIERLAYER_0_SECTION.SCANNER_HARDWARE_ID'
        ]
        
        for key in hardware_id_keys:
            if key in self.slide.properties:
                hard_id = self.slide.properties[key]
                print(f"硬件ID: {hard_id}")
                return hard_id in MRXS_DEVICE_LIST
        
        return False
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

    def get_img(self, level, x, y, w, h, channel_dict, is_gray=False, color_info={}, img_color_dict={}):
        """
        :param level: 层数
        :param x,y,w,h: level0的x,y，level下的w,h
        :param channel_dict: 获取的颜色字典，格式为 {channel_name: color}
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
        if self.format_type == 'IMG':
            channel_order_list = channel_list
            if img_color_dict is not None:
                channel_order_list = list(img_color_dict.keys())
        else:
            channel_order_list = self._get_channel_order_list()
        read_level = {}
        for i in range(channel_count):
            channel = channel_list[i]
            if channel not in channel_order_list:
                raise ValueError(f"invalid channel {channel}")
            # 找到当前颜色通道在openslide中对应的level和在这个level中3个通道对应的位置
            index = channel_order_list.index(channel)
            dst_level_0 = index // 3
            rgb_pos = index % 3
            dst_level = level * counts_per_level + dst_level_0
            if dst_level in read_level:
                read_level[dst_level].append((i, rgb_pos))
            else:
                read_level[dst_level] = [(i, rgb_pos)]

        if is_gray:
            scale = 1 / channel_count
            dst_img = np.zeros((h, w), dtype='uint8')
            for dst_level, ch_info_list in read_level.items():
                region = self.slide.read_region((x, y), dst_level, dst_size)
                if resize_flag:
                    region = region.resize((w, h))
                region = region.convert('RGB')
                image_np = np.array(region)
                for output_idx, rgb_pos in ch_info_list:
                    gray_img = image_np[:, :, rgb_pos]
                    # 获取black,white,gamma
                    param_dict = color_info.get(channel_list[output_idx], {})
                    black = param_dict.get('black', 0)
                    white = param_dict.get('white', 255)
                    gamma = param_dict.get('gamma', 1)
                    # 图像增强
                    gray_img = self.color_enhancement(gray_img, black, white, gamma)
                    dst_img += np.array(gray_img * scale, np.uint8)
            # dst_img = Image.fromarray(dst_img)
            dst_img = ImageEnhance.Brightness(Image.fromarray(dst_img)).enhance(channel_count)
        else:
            # 读取并使用自定义颜色上色
            dst_img = np.zeros((h, w, 3), dtype='uint8')
            for dst_level, ch_info_list in read_level.items():
                region = self.slide.read_region((x, y), dst_level, dst_size)
                if resize_flag:
                    region = region.resize((w, h))
                region = region.convert('RGB')
                image_np = np.array(region)
                for output_idx, rgb_pos in ch_info_list:
                    gray_img = image_np[:, :, rgb_pos]
                    # 获取black,white,gamma
                    param_dict = color_info.get(channel_list[output_idx], {})
                    black = param_dict.get('black', 0)
                    white = param_dict.get('white', 255)
                    gamma = param_dict.get('gamma', 1)
                    # 图像增强
                    gray_img = self.color_enhancement(gray_img, black, white, gamma)

                    # 使用快速颜色映射替代ImageOps.colorize
                    custom_color = channel_dict[channel_list[output_idx]]
                    colored_img = self._apply_color_fast(gray_img, custom_color)
                    dst_img = np.clip(dst_img.astype(np.uint16) + colored_img.astype(np.uint16), 0, 255).astype(
                        np.uint8)
            dst_img = Image.fromarray(dst_img)
        return dst_img

    def get_tile(self, dz_level, address, channel_dict, is_gray=False, color_info={}, img_color_dict={}):
        """
        :param dz_tile_size: 构建deepzoom的 tile_size
        :param dz_overlap: 构建deepzoom的 overlap
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
        counts_per_level = self._get_counts_per_level()
        max_level = self._get_max_level()
        max_level_w, max_level_h = self.slide.level_dimensions[-1]

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
        if self.format_type == 'IMG':
            # 图片的原始通道跟传入一致
            channel_order_list = channel_list
            if img_color_dict is not None:
                channel_order_list = list(img_color_dict.keys())
        else:
            channel_order_list = self._get_channel_order_list()
        read_level = {}
        for i in range(channel_count):
            channel = channel_list[i]
            if channel not in channel_order_list:
                raise ValueError(f"invalid channel {channel}")
            # 找到当前颜色通道在openslide中对应的level和在这个level中3个通道对应的位置
            index = channel_order_list.index(channel)
            dst_level_0 = index // 3
            rgb_pos = index % 3
            dst_level = level * counts_per_level + dst_level_0
            if dst_level in read_level:
                read_level[dst_level].append((i, rgb_pos))
            else:
                read_level[dst_level] = [(i, rgb_pos)]

        if is_gray:
            scale = 1 / channel_count
            dst_img = np.zeros((h, w), dtype='uint8')
            for dst_level, ch_info_list in read_level.items():
                region = self.slide.read_region((x, y), dst_level, dst_size)
                if resize_flag:
                    region = region.resize((w, h))
                region = region.convert('RGB')
                image_np = np.array(region)
                for output_idx, rgb_pos in ch_info_list:
                    gray_img = image_np[:, :, rgb_pos]
                    # 获取black,white,gamma
                    param_dict = color_info.get(channel_list[output_idx], {})
                    black = param_dict.get('black', 0)
                    white = param_dict.get('white', 255)
                    gamma = param_dict.get('gamma', 1)
                    # 图像增强
                    gray_img = self.color_enhancement(gray_img, black, white, gamma)
                    dst_img += np.array(gray_img * scale, np.uint8)
            # dst_img = Image.fromarray(dst_img)
            dst_img = ImageEnhance.Brightness(Image.fromarray(dst_img)).enhance(channel_count)
        else:
            # 读取并使用自定义颜色上色
            dst_img = np.zeros((h, w, 3), dtype='uint8')
            for dst_level, ch_info_list in read_level.items():
                region = self.slide.read_region((x, y), dst_level, dst_size)
                if resize_flag:
                    region = region.resize((w, h))
                region = region.convert('RGB')
                image_np = np.array(region)
                for output_idx, rgb_pos in ch_info_list:
                    gray_img = image_np[:, :, rgb_pos]
                    # 获取black,white,gamma
                    param_dict = color_info.get(channel_list[output_idx], {})
                    black = param_dict.get('black', 0)
                    white = param_dict.get('white', 255)
                    gamma = param_dict.get('gamma', 1)
                    # 图像增强
                    gray_img = self.color_enhancement(gray_img, black, white, gamma)

                    # 使用快速颜色映射替代ImageOps.colorize
                    custom_color = channel_dict[channel_list[output_idx]]
                    colored_img = self._apply_color_fast(gray_img, custom_color)
                    dst_img = np.clip(dst_img.astype(np.uint16) + colored_img.astype(np.uint16), 0, 255).astype(
                        np.uint8)
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

    def color_enhancement(self, mat, black, white, gamma):
        """
        做图像拉升
        :param mat: 输入图像矩阵
        :param black: 最低界限
        :param white: 最高界限
        :param gamma: gamma值
        :return:
        """
        tmp_mat = mat.astype(np.uint8)  # 确保输入是 uint8
        tmp_mat = np.clip(tmp_mat, black, white)  # 使用 np.clip 替代条件语句
        tmp_mat = ((tmp_mat - black) / (white - black) * 255).astype(np.uint8)

        real_gamma = 2 - gamma
        table = np.array([((i / 255.0) ** real_gamma) * 255 for i in range(256)]).astype(np.uint8)
        tmp_mat = cv2.LUT(tmp_mat, table)

        return tmp_mat


def print_properties(tiff_path):
    slide = OpenSlide(tiff_path)
    print("Properties:")
    for key, value in slide.properties.items():
        print(f"  {key}: {value}")
    slide.close()


if __name__ == '__main__':
    import traceback
    data_path = r"C:\Users\Administrator\Documents\WXWork\1688854579669486\Cache\File\2025-10\20250924_092940.svs"
    # data_path = r'F:\WXWork\1688853525018058\Cache\Image\2025-08\样本1局部 2025-05-06 15.39.47.jpg'
    data_slide = Data_Slide(data_path, IMG=False)
    color_dic = data_slide.get_color_info()
    properties = data_slide.slide.properties
    json_str = properties.get('openslide.comment', '').split('|METADATA_JSON =')[-1]
    data_info = json.loads(json_str)
    if isinstance(data_info, str):
        data_info = json.loads(data_info)
    # a = data_slide.slide.associated_images.get('macro')
    # 只读取DAPI通道的level 1全图
    # level = 4
    # level_dimensions = data_slide.get_level_dimension(level)
    # width, height = level_dimensions
    # print(f"Level {level} 尺寸: {width} x {height}")
    
    # # 创建只包含DAPI通道的字典
    # dapi_channel_dict = {}
    # for channel_name, color in color_dic.items():
    #     if 'DAPI' in channel_name.upper():
    #         dapi_channel_dict[channel_name] = color
    #         break
    
    # if dapi_channel_dict:
    #     print(f"找到DAPI通道: {list(dapi_channel_dict.keys())}")
    #     # 读取DAPI通道的level 1全图
    #     dapi_image = data_slide.get_img(level, 0, 0, width, height, dapi_channel_dict, is_gray=False, color_info={})
    #     print(f"成功读取DAPI通道Level {level}全图，尺寸: {width} x {height}")
    # else:
    #     print("未找到DAPI通道")
    # 显示level2, level3, level4的全图
    # print("=== 显示不同level的全图 ===")
    for level in [1, 2, 3, 4]:
        try:
            # 获取当前level的完整尺寸
            level_dimensions = data_slide.get_level_dimension(level)
            width, height = level_dimensions
            print(f"Level {level} 尺寸: {width} x {height}")

            # 读取整个level的图像
            # 起始点为(0,0)，读取整个level的尺寸
            full_image = data_slide.get_img(level, 0, 0, width, height, color_dic, is_gray=False,
                                            color_info={})

            # 转换为PIL Image并保存
            # full_image_pil = Image.fromarray(full_image)
            output_filename = f'level_{level}_full_image.png'
            full_image.save(output_filename)
            # print(f"Level {level} 全图已保存为: {output_filename}")

            # 显示图像
            # full_image.show()

        except Exception as e:
            traceback.print_exc()
            print(f"处理Level {level}时出错: {e}")
    
    # 原有的代码
    label_image = data_slide.slide.associated_images.get('label')
    thumbnail = data_slide.slide.associated_images.get('thumbnail')
    if thumbnail is not None:
        thumbnail = thumbnail.convert('RGB')
        width, height = thumbnail.size
        if width > 1000 or height > 1000:
            ratio = min(1000 / width, 1000 / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            thumbnail = thumbnail.resize((new_width, new_height))
        # print(thumbnail_pil.size)
        thumbnail_rotate = thumbnail.transpose(method=Image.Transpose.ROTATE_90)


    # img = data_slide.get_img(0, 14645, 26230, 1500, 1500, {'SpOr': '#FF0000FF'})
    # img.save('test.jp2', format="JPEG2000", quality_mode="dB", quality_layers=[50])
    # Image.open('test.jp2').show()
    # print(data_slide.slide.level_dimensions)
    #
    # color_dic = data_slide.get_color_info()
    # print(color_dic)
    #
    # # 性能测试
    # import time
    #
    # # 单个瓦片测试
    # print("=== 单个瓦片性能测试 ===")
    # start_time = time.time()
    # img = data_slide.get_img(5, 768 * 32, 512 * 32, 256, 256, {'SpOr': '#FFFF0000'}, is_gray=True,
    #                              color_info={'SpOr': {'black': 10, 'white': 150, 'gamma': 1.5}})
    #
    # single_time = time.time() - start_time
    # print(f"单个图像获取时间: {single_time:.3f}秒")
    # img.show()
    #
    # start_time = time.time()
    # img1 = data_slide.get_tile(11, (3, 2), color_dic, is_gray=False, color_info={})
    # tile_time = time.time() - start_time
    # print(f"单个瓦片获取时间: {tile_time:.3f}秒")
    # img1.show()











