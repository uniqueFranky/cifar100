"""
训练器模块
"""

from .single_gpu import SingleGPUTrainer
from .data_parallel import DataParallelTrainer
from .ddp_trainer import DDPTrainer

__all__ = ['SingleGPUTrainer', 'DataParallelTrainer', 'DDPTrainer']
