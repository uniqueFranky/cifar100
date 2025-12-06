"""
训练器模块
"""

from .single_gpu import SingleGPUTrainer
from .data_parallel import DataParallelTrainer
from .ddp_trainer import DDPTrainer
from .pipeline_parallel import PipelineParallelTrainer
from .model_parallel import ModelParallelTrainer

__all__ = ['SingleGPUTrainer', 'DataParallelTrainer', 'DDPTrainer', 'ModelParallelTrainer', 'PipelineParallelTrainer']
