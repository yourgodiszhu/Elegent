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

from numba.cuda.printimpl import print_item

from slide最终优化版本_arg import main as slide_main
# from merge图像配准_new_4channel_gray_png_yh_version1_arg1 import main as merge_main
from merge图像配准_new_4channel_gray_png_yh_version1_arg1优化 import main as merge_main
from crop_4channel_gray_png_yh_single_arg import main_proc as crop_main
# from 改变配准算法_4张_model_4channel_gray_png_single存在内存中_arg1 import main as change_main
# from 改变配准算法_4张_model_4channel_gray_png_single存在内存中_arg1_全局参数 import main as change_main
# from version_1.改变配准算法_4张_model_4channel_gray_png_single存在内存中_arg import main as change_main
from 改变配准算法_4张_model_4channel_gray_png_single存在内存中_arg1_全局参数_直接覆盖  import main as change_main
# from 改变配准算法_4张_model_4channel_gray_png_single存在内存中_arg1_全局参数_single  import process_stage2_results as change_results
from creat_max_DAPI_arg import create_max_projection_parallel as max_main
from creat_max_DAPI_arg import find_and_copy_moving_png_parallel as copy_main
from 保存为svs_arg import task_function
from 保存为svs_arg import threaded_executor as svs_main
# from multipage_tiff_test1_arg import main as multipage
from multipage_tiff_test1_arg分块写入_保存resize图片到磁盘再读取 import main as multipage
from 最清晰的DAPI_arg import get_img_path as clear_dapi
from 处理重影patch_多进程_粗 import main as patch_main
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


def worker(func, params, queue):
    """子进程工作函数"""
    try:
        result = func(*params) if isinstance(params, (tuple, list)) else func(params)
        queue.put(('success', result))
    except Exception as e:
        print(e)
        queue.put(('error', str(e)))


def run_step1(func, params, task_function=None):
    """改进版：可捕获子进程返回值"""
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.cuda.synchronize()

    # 创建进程间通信队列
    queue = multiprocessing.Queue()

    # 启动子进程
    if task_function is None:
        p = multiprocessing.Process(
            target=worker,
            args=(func, params if isinstance(params, (tuple, list)) else (params,), queue)
        )
    else:
        p = multiprocessing.Process(
            target=worker,
            args=(func, (params, task_function), queue)
        )

    p.start()
    p.join()

    # 获取结果
    status, data = queue.get()
    if status == 'error':
        raise RuntimeError(f"子进程执行失败: {data}")

    # 清理资源
    gc.collect()
    torch.cuda.empty_cache()

    return data  # 返回子进程的结果
# log_file = "main_pipeline.log"
# import sys
#
# # 创建文件夹保存日志（如果需要）
# # os.makedirs(os.path.dirname(log_file), exist_ok=True)
#
# # 重定向 stdout 和 stderr
# class Logger(object):
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a", encoding="utf-8")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.flush()
#
#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()
#
# sys.stdout = Logger(log_file)
# sys.stderr = sys.stdout  # 错误信息也保存

print("日志开始")  # 之后所有 print 都会写入 log 文件
if __name__ == "__main__":
    from progress_manager import ProgressManager
    # 初始化进度条系统
    module_weights = {
        "slide_main": 0.08,
        "merge_main": 0.12,
        "crop_main": 0.06,
        "change_main": 0.358,
        "patch_main": 0.19,
        "copy_main": 0.001,
        "clear_dapi": 0.001,
        "multipage": 0.19
    }

    pm = ProgressManager(module_weights)
    pm.start()
    pm = ProgressManager(module_weights)
    pm.start()
    # multiprocessing.freeze_support()  # Windows 多进程兼容
    from types import SimpleNamespace
    # import pycuda.autoinit
    # print(torch.cuda.is_available())
    for i in os.listdir(r"F:\10-23\1.侯健 SWR02912978  3张"):
        try:
            # print(i)
            # if fr'all_out_png_{i}'  not in os.listdir(r"F:\3d\version3_9_支持中文_调整曝光"):
            #     continue
            input_dir=os.path.join(r'F:\10-23\1.侯健 SWR02912978  3张', i)
            # print(input_dir)
            # if  '3412'!=i:
            #     continue
            # args = parse_args(input_dir=input_dir,output_dir=fr'{os.path.basename(i)}_Gray_date',save_dir=fr'all_out_png_{os.path.basename(i)}',data_dir=fr'{os.path.basename(i)}_Gray_date/{os.path.basename(i)}',save_svs_dir=fr'{os.path.basename(i)}.svs',label_dir='1.bmp',macro_dir='2.bmp')
            # config = load_config(args.config)
            args = SimpleNamespace(input_dir=input_dir,output_dir=fr'{os.path.basename(i)}_Gray_date',save_dir=fr'all_out_png_{os.path.basename(i)}',data_dir=fr'{os.path.basename(i)}_Gray_date/{os.path.basename(i)}',save_svs_dir=fr'{os.path.basename(i)}.svs',label_dir='1.bmp',macro_dir='2.bmp',over_lab=(400,400),window_size=(2048,2048),merge_speed=1)
            # 合并参数.
            import re
            txt_path = rf"F:\jsons\{i}.txt"  # 你的 TXT 文件路径
            json_path = rf"F:\jsons\{i}.json"  # 输出 JSON 文件路径

            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            #
            # # 修正反斜杠
            content = content.replace('\\', '\\\\')
            #
            # # 删除数组中最后一个对象后的多余逗号
            content = re.sub(r',\s*(\]\s*)$', r'\1', content, flags=re.MULTILINE)
            #
            # 尝试解析 JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print("JSON 解析错误:", e)
                exit(1)

            # 保存合法 JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            print(f"已生成合法 JSON: {json_path}")
            # json_path = r"C:\Users\Administrator\Documents\WXWork\1688854579669486\Cache\File\2025-10\传参示例.json"
            # if  os.path.exists(json_path):
            #     raise FileNotFoundError(f"{json_path} 存在！")
            with open(json_path, 'r', encoding='utf-8') as f:
                color_dict = json.load(f)
            params = {
                "input_dir": args.input_dir,
                "output_dir": args.output_dir,
                "save_dir": args.save_dir,
                "data_dir": args.data_dir,
                "save_svs_dir": args.save_svs_dir,
                "label_dir": args.label_dir,
                "macro_dir": args.macro_dir,
                'over_lab':args.over_lab,
                'window_size':args.window_size,
                'merge_speed':args.merge_speed,
                'config':color_dict,
            }
            #
            print("最终参数:", params)

            start1 = time.time()
            run_step(slide_main, params)
            slide_time=time.time()-start1
            pm.update_module_done("slide_main")
            # exit()

            start = time.time()
            run_step(merge_main, params)
            merge_time=time.time()-start
            pm.update_module_done("merge_main")
            # exit()
            start = time.time()
            run_step(crop_main, params)
            crop_time=time.time()-start
            pm.update_module_done("crop_main")
            # exit()

            start = time.time()
            run_step(change_main, params)
            change_time=time.time()-start
            pm.update_module_done("change_main")
            # exit()

            start = time.time()
            run_step(patch_main, params)
            patch_time=time.time()-start
            pm.update_module_done("patch_main")
            # exit()
            #
            # break
            #
            start = time.time()
            copy_count = copy_main(params)
            copy_time=time.time()-start
            pm.update_module_done("copy_main")
            # exit()
            #
            # gc.collect()
            # torch.cuda.empty_cache()
            # #
            # # if copy_count > 0:
            # #     print('tete')
            # #     start = time.time()
            # #     dapi_count = max_main(params)
            # #     print(f"成功处理了 {dapi_count} 个 DAPI 图像")
            # #     max_time=time.time()-start
            #
            # # start = time.time()
            # # svs_main(task_function, params)
            # # svs_time=time.time()-start
            #
            #
            #
            #
            start = time.time()
            # # #
            gc.collect()
            torch.cuda.empty_cache()
            run_step(clear_dapi, params)
            # #
            clear_dapi_time=time.time()-start
            pm.update_module_done("clear_dapi")

            print(f"清理DAPI耗时: {clear_dapi_time:.2f} 秒")
            # #
            start = time.time()
            run_step(multipage, params)
            multipage_time=time.time()-start
            # #
            pm.update_module_done("multipage")

            # total_time = time.time() - start1-svs_time-max_time
            total_time = time.time() - start1
            #
            x = f"耗时_{i}_{total_time:.2f}秒.txt"
            with open(x, 'w', encoding='utf-8') as f:
                f.write(f"slide_time耗时: {slide_time:.2f} 秒\n"
                        f"merge耗时: {merge_time:.2f} 秒\n"
                        f"crop耗时: {crop_time:.2f} 秒\n"
                        f"融合耗时: {change_time:.2f} 秒\n"
                        # f"max耗时: {max_time:.2f} 秒\n"
                        # f"svs耗时: {svs_time:.2f} 秒\n"
                        f"copy耗时: {copy_time:.2f} 秒\n"
                        f"patch耗时: {patch_time:.2f} 秒\n"
                       f"clear_dapi耗时: {clear_dapi_time:.2f} 秒\n"
                        f"multipage耗时: {multipage_time:.2f} 秒\n")
            pm.close()

            print(f"总耗时: {time.time() - start:.2f} 秒")
        except Exception as e:
            print(e)
            with open('error.txt', 'a', encoding='utf-8') as f:
                f.write(f'error:{i}_{e}\n')
            continue
