"""
主入口文件 - 支持多种训练模式
使用方式:
    单GPU: python main.py --mode single --gpu_ids 0
    DataParallel: python main.py --mode dp --gpu_ids 0,1,2,3
    DDP: python main.py --mode ddp --gpu_ids 0,1,2,3
"""

import argparse
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from trainers.single_gpu import SingleGPUTrainer
from trainers.data_parallel import DataParallelTrainer
from trainers.ddp_trainer import DDPTrainer
from trainers.pipeline_parallel import PipelineParallelTrainer
from trainers.model_parallel import ModelParallelTrainer
from trainers.hybrid_parallel import HybridParallelTrainer



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多GPU训练框架 - CIFAR100')

    # 训练模式
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'dp', 'ddp', 'mp', 'hp', 'pp'],
                       help='训练模式: single(单GPU), dp(DataParallel), ddp(DistributedDataParallel), mp(ModelParallel), hp(HybridParallel)')
    
    # GPU设置
    parser.add_argument('--gpu-ids', type=str, default='0',
                       help='使用的GPU ID，用逗号分隔，如: 0,1,2,3')
    
    # 数据加载设置
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader的worker数量')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                       help='每个worker预取的batch数量')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                       help='是否使用pin_memory加速数据传输')
    parser.add_argument('--persistent-workers', action='store_true', default=True,
                       help='是否保持worker进程')
    
    # 训练超参数
    parser.add_argument('--batch-size', type=int, default=128,
                       help='每个GPU的batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='权重衰减')
    
    # 学习率调度
    parser.add_argument('--lr-schedule', type=str, default='step',
                       choices=['step', 'cosine', 'multistep'],
                       help='学习率调度策略')
    parser.add_argument('--lr-step_size', type=int, default=50,
                       help='StepLR的步长')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                       help='学习率衰减因子')
    
    # 数据集设置
    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['cifar10', 'cifar100'],
                       help='数据集选择')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='数据集根目录')
    
    # 模型设置
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='模型架构')
    
    # 保存和日志
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='日志打印间隔(batch)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='模型保存间隔(epoch)')
    parser.add_argument('--final-checkpoint-path', type=str, default=None,
                       help='最终checkpoint的自定义保存路径(文件名)')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    parser.add_argument('--evaluate', action='store_true',
                       help='只进行评估，不训练')
    
    # DDP特定参数
    parser.add_argument('--dist-backend', type=str, default='nccl',
                       help='分布式后端: nccl, gloo')
    parser.add_argument('--dist-url', type=str, default='env://',
                       help='分布式训练URL')

    parser.add_argument('--chunks', type=int, default=16)

    args = parser.parse_args()
    
    # 解析GPU IDs
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    args.num_gpus = len(args.gpu_ids)
    
    return args


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 获取配置
    config = get_config(args)
    
    # 打印配置
    print("=" * 80)
    print("训练配置:")
    print("=" * 80)
    for key, value in vars(config).items():
        print(f"  {key:25s}: {value}")
    print("=" * 80)
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法使用GPU训练")
        return
    
    available_gpus = torch.cuda.device_count()
    print(f"\n可用GPU数量: {available_gpus}")
    
    for gpu_id in config.gpu_ids:
        if gpu_id >= available_gpus:
            print(f"错误: GPU {gpu_id} 不存在")
            return
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # 根据模式选择训练器
    print(f"\n使用训练模式: {config.mode.upper()}")
    
    if config.mode == 'single':
        if config.num_gpus > 1:
            print(f"警告: 指定了{config.num_gpus}个GPU，但单GPU模式只使用第一个GPU")
        trainer = SingleGPUTrainer(config)
        trainer.train()
        
    elif config.mode == 'dp':
        if config.num_gpus < 2:
            print("警告: DataParallel模式建议使用至少2个GPU")
        trainer = DataParallelTrainer(config)
        trainer.train()

    elif config.mode == 'ddp':
        if config.num_gpus < 2:
            print("警告: DDP模式建议使用至少2个GPU")
        # DDP使用多进程，需要特殊启动
        trainer = DDPTrainer(config)
        trainer.launch()
    
    elif config.mode == 'pp':
        if config.num_gpus < 2:
            print("警告: PipelineParallel模式建议使用至少2个GPU")
        # === 修改这里: 使用 launch() 启动多进程 ===
        trainer = PipelineParallelTrainer(config)
        trainer.launch()
    elif config.mode == 'mp':
        if config.num_gpus < 2:
            print("警告: ModelParallel模式建议使用至少2个GPU")
        trainer = ModelParallelTrainer(config)
        trainer.train()
    elif config.mode == 'hp':
        if config.num_gpus != 4:
            print("警告: HybridParallel模式建议使用4个GPU")
        trainer = HybridParallelTrainer(config)
        trainer.launch()
    
    else:
        print(f"错误: 未知的训练模式 '{config.mode}'")
        return

    
    print("\n训练完成!")


if __name__ == '__main__':
    main()
