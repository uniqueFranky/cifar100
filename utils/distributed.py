"""
分布式训练工具
"""

import os
import torch
import torch.distributed as dist


def setup_distributed(rank, world_size, backend='nccl', device=None):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        backend: 分布式后端
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=device)
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(device)


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """
    在所有进程间归约tensor
    
    Args:
        tensor: 要归约的tensor
        world_size: 总进程数
    
    Returns:
        归约后的tensor
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def is_main_process(rank=None):
    """判断是否为主进程"""
    if rank is not None:
        return rank == 0
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """获取当前进程的rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """获取总进程数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
