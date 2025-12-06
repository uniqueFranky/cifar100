"""
配置管理
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """训练配置类"""
    # 训练模式
    mode: str = 'single'  # single, dp, ddp
    
    # GPU设置
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_gpus: int = 1
    
    # 数据加载
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # 训练超参数
    batch_size: int = 128
    epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # 学习率调度
    lr_schedule: str = 'step'
    lr_step_size: int = 50
    lr_gamma: float = 0.1
    
    # 数据集
    dataset: str = 'cifar100'
    data_root: str = './data'
    num_classes: int = 100
    
    # 模型
    model: str = 'resnet18'
    
    # 保存和日志
    save_dir: str = './checkpoints'
    log_interval: int = 100
    save_interval: int = 10
    
    # 其他
    seed: int = 42
    resume: str = None
    evaluate: bool = False
    
    # DDP
    dist_backend: str = 'nccl'
    dist_url: str = 'env://'
    
    # 自动计算的属性
    effective_batch_size: int = 128  # batch_size * num_gpus
    
    def __post_init__(self):
        """初始化后处理"""
        # 计算有效batch size
        if self.mode in ['dp', 'ddp']:
            self.effective_batch_size = self.batch_size * self.num_gpus
        else:
            self.effective_batch_size = self.batch_size
        
        # 根据数据集设置类别数
        if self.dataset == 'cifar10':
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.num_classes = 100
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置随机种子
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


def get_config(args):
    """从命令行参数创建配置"""
    config = Config(
        mode=args.mode,
        gpu_ids=args.gpu_ids,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        dataset=args.dataset,
        data_root=args.data_root,
        model=args.model,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        resume=args.resume,
        evaluate=args.evaluate,
        dist_backend=args.dist_backend,
        dist_url=args.dist_url,
    )
    return config
