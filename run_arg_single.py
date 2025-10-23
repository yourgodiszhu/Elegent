#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主控制脚本 - 参数入口（子进程执行每一步以降低内存占用）
"""

import argparse
import json
import os
import time
import gc
import torch
import multiprocessing

from slide最终优化版本_arg import main as slide_main
# from merge图像配准_new_4channel_gray_png_yh_version1_arg1 import main as merge_main
from merge图像配准_new_4channel_gray_png_yh_version1_arg1优化 import main as merge_main
from crop_4channel_gray_png_yh_single_arg import main_proc as crop_main
# from 改变配准算法_4张_model_4channel_gray_png_single存在内存中_arg1 import main as change_main
from 改变配准算法_4张_model_4channel_gray_png_single存在内存中_arg1_全局参数 import main as change_main
# from version_1.改变配准算法_4张_model_4channel_gray_png_single存在内存中_arg import main as change_main

from creat_max_DAPI_arg import create_max_projection_parallel as max_main
from creat_max_DAPI_arg import find_and_copy_moving_png_parallel as copy_main
from 保存为svs_arg import task_function
from 保存为svs_arg import threaded_executor as svs_main
from multipage_tiff_test1_arg import main as multipage
from 最清晰的DAPI_arg import get_img_path as clear_dapi
#
def parse_args(input_dir,output_dir,save_dir,data_dir,save_svs_dir,label_dir,macro_dir):
    parser = argparse.ArgumentParser(description="数字病理图像处理系统")
    parser.add_argument("--input_dir", default=r"D:\3d\test_datasets\M_GFAP",
                        help="病理切片数据根目录")
    parser.add_argument("--output_dir", default="M_GFAP_Gray_date",
                        help="输出目录")
    parser.add_argument("--save_dir", default="all_out_png_M_GFAP",
                        help="保存目录")
    parser.add_argument("--data_dir", default="M_GFAP_Gray_date/M_GFAP",
                        help="数据目录")
    parser.add_argument("--save_svs_dir", default="M_GFAP.svs",
                        help="保存SVS目录")
    parser.add_argument("--label_dir", default="1.bmp",
                        help="标签图像路径")
    parser.add_argument("--macro_dir", default="2.bmp",
                        help="宏图像路径")
    parser.add_argument("--config", default="config.json",
                        help="JSON配置文件路径")
    return parser.parse_args()


def load_config(config_file):
    if not os.path.exists(config_file):
        return {}
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_step(func, params,task_function=None):
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.cuda.synchronize()

    """子进程运行每一步，结束后自动释放内存"""
    if task_function is None:
        p = multiprocessing.Process(target=func, args=(params,))
        p.start()
        p.join()
        # 清理显存和内存
        gc.collect()
        try:

            torch.cuda.empty_cache()
        except:
            pass
    else:
        p = multiprocessing.Process(target=func, args=(params,task_function))
        p.start()
        p.join()
        # 清理显存和内存
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass


if __name__ == "__main__":
    # multiprocessing.freeze_support()  # Windows 多进程兼容
    from types import SimpleNamespace
    # import pycuda.autoinit
    # print(torch.cuda.is_available())
    for i in os.listdir(r"C:\Users\Administrator\Desktop\8_22_test"):
        try:
            # print(i)
            # if fr'all_out_png_{i}' in os.listdir(r"D:\3d\version3_4"):
            #     continue
            input_dir=os.path.join(r'C:\Users\Administrator\Desktop\8_22_test', i)
            # print(input_dir)
            if  'HC'!=i :
                continue
            # args = parse_args(input_dir=input_dir,output_dir=fr'{os.path.basename(i)}_Gray_date',save_dir=fr'all_out_png_{os.path.basename(i)}',data_dir=fr'{os.path.basename(i)}_Gray_date/{os.path.basename(i)}',save_svs_dir=fr'{os.path.basename(i)}.svs',label_dir='1.bmp',macro_dir='2.bmp')
            # config = load_config(args.config)
            args = SimpleNamespace(input_dir=input_dir,output_dir=fr'{os.path.basename(i)}_Gray_date',save_dir=fr'all_out_png_{os.path.basename(i)}',data_dir=fr'{os.path.basename(i)}_Gray_date/{os.path.basename(i)}',save_svs_dir=fr'{os.path.basename(i)}.svs',label_dir='1.bmp',macro_dir='2.bmp')
            # 合并参数
            params = {
                "input_dir": args.input_dir,
                "output_dir": args.output_dir,
                "save_dir": args.save_dir,
                "data_dir": args.data_dir,
                "save_svs_dir": args.save_svs_dir,
                "label_dir": args.label_dir,
                "macro_dir": args.macro_dir,
            }

            print("最终参数:", params)
            # # #
            start1 = time.time()
            run_step(slide_main, params)
            slide_time=time.time()-start1

            start = time.time()
            run_step(merge_main, params)
            merge_time=time.time()-start

            start = time.time()
            run_step(crop_main, params)
            crop_time=time.time()-start

            start = time.time()
            run_step(change_main, params)
            change_time=time.time()-start
            # break
            #
            start = time.time()
            copy_count = copy_main(params)
            #
            gc.collect()
            torch.cuda.empty_cache()

            if copy_count > 0:
                # print('tete')
                dapi_count = max_main(params)
                print(f"成功处理了 {dapi_count} 个 DAPI 图像")
                max_time=time.time()-start

            start = time.time()
            svs_main(task_function, params)
            svs_time=time.time()-start
            #
            #
            #
            #
            start = time.time()
            #
            gc.collect()
            torch.cuda.empty_cache()
            run_step(clear_dapi, params)
            # #
            clear_dapi_time=time.time()-start
            print(f"清理DAPI耗时: {clear_dapi_time:.2f} 秒")
            #
            start = time.time()
            run_step(multipage, params)
            multipage_time=time.time()-start
            #
            total_time = time.time() - start1-svs_time
            x = f"耗时_{i}_{total_time:.2f}秒.txt"
            with open(x, 'w', encoding='utf-8') as f:
                f.write(f"slide_time耗时: {slide_time:.2f} 秒\n"
                        f"merge耗时: {merge_time:.2f} 秒\n"
                        f"crop耗时: {crop_time:.2f} 秒\n"
                        f"融合耗时: {change_time:.2f} 秒\n"
                        f"max耗时: {max_time:.2f} 秒\n"
                        f"svs耗时: {svs_time:.2f} 秒\n"
                       f"clear_dapi耗时: {clear_dapi_time:.2f} 秒\n"
                        f"multipage耗时: {multipage_time:.2f} 秒\n")
            print(f"总耗时: {time.time() - start:.2f} 秒")
        except Exception as e:
            print(e)
            with open('error.txt', 'a', encoding='utf-8') as f:
                f.write(f'error:{i}_{e}\n')
            continue
