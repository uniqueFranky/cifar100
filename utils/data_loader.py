"""
数据加载工具
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_transforms(dataset='cifar100'):
    """获取数据增强transforms"""
    if dataset in ['cifar10', 'cifar100']:
        # CIFAR数据集的均值和标准差
        if dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        else:  # cifar100
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError(f"未知数据集: {dataset}")
    
    return transform_train, transform_test


def get_dataset(config):
    """获取数据集"""
    transform_train, transform_test = get_transforms(config.dataset)
    
    if config.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=True,
            download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=False,
            download=True, transform=transform_test
        )
    elif config.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_root, train=True,
            download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=config.data_root, train=False,
            download=True, transform=transform_test
        )
    else:
        raise ValueError(f"未知数据集: {config.dataset}")
    
    return trainset, testset


def get_dataloader(config, distributed=False, rank=0, world_size=1):
    """
    获取数据加载器
    
    Args:
        config: 配置对象
        distributed: 是否使用分布式
        rank: 当前进程的rank
        world_size: 总进程数
    """
    trainset, testset = get_dataset(config)
    
    # 创建sampler
    if distributed:
        train_sampler = DistributedSampler(
            trainset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            testset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True
    
    # 创建DataLoader
    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )
    
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )
    
    return trainloader, testloader, train_sampler
