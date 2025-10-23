from tqdm import tqdm
from multiprocessing import Manager
import time

class ProgressManager:
    def __init__(self, modules):
        """
        modules: dict 模块名称 -> 权重 (权重和自动归一化)
        """
        total = sum(modules.values())
        self.weights = {k: v / total for k, v in modules.items()}
        self.manager = Manager()
        self.progress = self.manager.dict({k: 0.0 for k in self.weights})
        self.done = self.manager.dict({k: False for k in self.weights})
        self.total_bar = None

    def start(self):
        self.total_bar = tqdm(total=100, desc="总进度", ncols=90)

    def update_module_done(self, module_name):
        """标记模块完成"""
        self.progress[module_name] = 1.0
        self.done[module_name] = True
        self.refresh()

    def refresh(self):
        """刷新总进度"""
        total = sum(self.progress[k] * w for k, w in self.weights.items())
        if self.total_bar:
            self.total_bar.n = int(total * 100)
            self.total_bar.refresh()

    def close(self):
        """关闭"""
        if self.total_bar:
            self.total_bar.n = 100
            self.total_bar.refresh()
            self.total_bar.close()
