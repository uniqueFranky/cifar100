"""
性能监控工具
"""

import time
import psutil
import torch
import numpy as np
from collections import defaultdict


class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
    
    def start_timer(self):
        """开始计时"""
        self.start_time = time.time()
    
    def end_timer(self, metric_name):
        """结束计时并记录"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.metrics[metric_name].append(elapsed)
            self.start_time = None
            return elapsed
        return 0
    
    def record_memory(self, metric_name, device=None):
        """记录内存使用"""
        # GPU内存
        if torch.cuda.is_available() and device is not None:
            if isinstance(device, int):
                device = torch.device(f'cuda:{device}')
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
            self.metrics[f'{metric_name}_gpu_memory'].append(gpu_memory)
        
        # CPU内存
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        self.metrics[f'{metric_name}_cpu_memory'].append(cpu_memory)
    
    def get_average(self, metric_name):
        """获取平均值"""
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return np.mean(self.metrics[metric_name])
        return 0
    
    def get_latest(self, metric_name):
        """获取最新值"""
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return self.metrics[metric_name][-1]
        return 0
    
    def reset(self):
        """重置所有指标"""
        self.metrics.clear()
        self.start_time = None
    
    def summary(self):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("性能统计摘要")
        print("="*60)
        
        for metric_name in sorted(self.metrics.keys()):
            values = self.metrics[metric_name]
            if len(values) > 0:
                avg = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                print(f"{metric_name}:")
                print(f"  平均: {avg:.4f}, 标准差: {std:.4f}")
                print(f"  最小: {min_val:.4f}, 最大: {max_val:.4f}")
