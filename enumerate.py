"""
批量训练脚本 - 在完全干净的环境中运行训练
不设置任何自定义环境变量，让系统和PyTorch使用默认配置

使用方式:
    python batch_train.py --config quick    # 快速测试配置
    python batch_train.py --config full     # 完整实验配置
    python batch_train.py --config custom   # 自定义配置
"""

import subprocess
import os
import sys
import time
import json
from datetime import datetime
from itertools import product
from typing import List, Dict, Any
import argparse


class TrainingScheduler:
    """训练任务调度器"""
    
    def __init__(self, base_checkpoint_dir: str = './checkpoints', log_dir: str = './training_logs'):
        self.base_checkpoint_dir = base_checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(base_checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.results = []
        self.current_task = 0
        self.total_tasks = 0
    
    def generate_experiment_name(self, params: Dict[str, Any]) -> str:
        """
        根据参数生成实验名称
        格式: mode_model_bs{batch}_ep{epochs}_gpu{num_gpus}_nw{num_workers}_pf{prefetch}
        """
        name_parts = [
            params['mode'],
            params['model'],
            f"bs{params['batch_size']}",
            f"ep{params['epochs']}",
            f"gpu{params['num_gpus']}"
        ]
        
        # 添加num_workers和prefetch_factor（如果不是默认值）
        if params.get('num_workers') is not None and params['num_workers'] != 4:
            name_parts.append(f"nw{params['num_workers']}")
        
        if params.get('prefetch_factor') is not None and params['prefetch_factor'] != 2:
            name_parts.append(f"pf{params['prefetch_factor']}")
        
        # 添加数据集信息（如果不是默认的cifar100）
        if params.get('dataset') and params['dataset'] != 'cifar100':
            name_parts.append(params['dataset'])
        
        return '_'.join(name_parts)
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        验证参数组合是否有效
        
        Returns:
            True if valid, False otherwise
        """
        mode = params['mode']
        num_gpus = len(params['gpu_ids'])
        
        # single模式只能使用1个GPU
        if mode == 'single' and num_gpus != 1:
            return False
        
        # dp, ddp, mp, hybrid模式需要至少2个GPU
        if mode in ['dp', 'ddp', 'mp', 'hybrid'] and num_gpus < 2:
            return False
        
        if mode == 'hybrid':
            # 混合并行需要GPU数量是模型并行组大小的整数倍
            mp_group_size = params.get('mp_group_size', 2)
            if num_gpus <= mp_group_size or num_gpus % mp_group_size != 0:
                return False
        
        return True
    
    def build_command(self, params: Dict[str, Any], checkpoint_path: str) -> List[str]:
        """构建训练命令"""
        mode = params['mode']
        
        # DDP和混合并行模式需要使用torchrun
        if mode in ['ddp', 'hybrid']:
            num_gpus = len(params['gpu_ids'])
            
            # 使用 torchrun (PyTorch 1.10+)
            cmd = [
                'torchrun',
                '--standalone',
                '--nnodes=1',
                f'--nproc_per_node={num_gpus}',
                'main.py'
            ]
        else:
            # 其他模式使用普通python命令
            cmd = ['python', '-u', 'main.py']
        
        # 添加所有参数
        for key, value in params.items():
            if key == 'gpu_ids':
                # GPU IDs特殊处理
                cmd.extend(['--gpu-ids', ','.join(map(str, value))])
            elif key == 'num_gpus':
                # 这个是计算得出的，不需要传递
                continue
            elif isinstance(value, bool):
                if value:
                    cmd.append(f'--{key.replace("_", "-")}')
            elif value is not None:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])
        
        # 添加最终checkpoint路径
        cmd.extend(['--final-checkpoint-path', checkpoint_path])
        
        return cmd
    
    def run_training(self, params: Dict[str, Any], exp_name: str) -> Dict[str, Any]:
        """运行单次训练 - 在完全干净的环境中"""
        self.current_task += 1
        
        print("\n" + "="*100)
        print(f"Task [{self.current_task}/{self.total_tasks}]: {exp_name}")
        print(f"Mode: {params['mode'].upper()}, GPUs: {params['gpu_ids']}")
        print("="*100)
        
        # 生成checkpoint路径
        checkpoint_path = os.path.join(self.base_checkpoint_dir, f"{exp_name}.pth")
        
        # 生成日志文件路径
        log_file = os.path.join(self.log_dir, f"{exp_name}.log")
        
        # 构建命令
        cmd = self.build_command(params, checkpoint_path)
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {log_file}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Running in clean environment (no custom env vars)")
        print("-"*100)
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行训练
        try:
            with open(log_file, 'w', buffering=1) as f:
                # 写入实验信息到日志
                f.write("="*100 + "\n")
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Mode: {params['mode'].upper()}\n")
                f.write(f"GPUs: {params['gpu_ids']}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Running in clean environment\n")
                f.write("="*100 + "\n\n")
                f.flush()
                
                # 不传递env参数，让subprocess使用完全干净的环境
                # 这样会继承父进程的环境，但不做任何修改
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    # env=None,  # 不设置，使用默认继承
                    # 创建新的进程组
                    preexec_fn=None if sys.platform == 'win32' else os.setpgrp
                )
                
                # 实时输出并保存到日志
                try:
                    for line in process.stdout:
                        print(line, end='')
                        f.write(line)
                        f.flush()
                except KeyboardInterrupt:
                    print("\n⚠ Interrupted by user, terminating process...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        print("⚠ Process didn't terminate, killing...")
                        process.kill()
                        process.wait()
                    raise
                
                process.wait()
                return_code = process.returncode
            
            elapsed_time = time.time() - start_time
            
            # 检查是否成功
            success = return_code == 0 and os.path.exists(checkpoint_path)
            
            result = {
                'experiment_name': exp_name,
                'params': params,
                'checkpoint_path': checkpoint_path,
                'log_file': log_file,
                'success': success,
                'return_code': return_code,
                'elapsed_time': elapsed_time,
                'elapsed_time_str': f"{elapsed_time/3600:.2f}h",
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                print(f"\n✓ Training completed successfully! Time: {elapsed_time/3600:.2f} hours")
            else:
                print(f"\n✗ Training failed! Return code: {return_code}")
                print(f"   Check log file: {log_file}")
            
            return result
            
        except KeyboardInterrupt:
            print(f"\n✗ Training interrupted by user")
            elapsed_time = time.time() - start_time
            
            return {
                'experiment_name': exp_name,
                'params': params,
                'checkpoint_path': checkpoint_path,
                'log_file': log_file,
                'success': False,
                'error': 'Interrupted by user',
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"\n✗ Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            elapsed_time = time.time() - start_time
            
            return {
                'experiment_name': exp_name,
                'params': params,
                'checkpoint_path': checkpoint_path,
                'log_file': log_file,
                'success': False,
                'error': str(e),
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_experiments(self, param_grid: Dict[str, List[Any]]):
        """运行所有实验"""
        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        all_combinations = list(product(*values))
        
        # 过滤掉无效的参数组合
        valid_combinations = []
        for combination in all_combinations:
            params = dict(zip(keys, combination))
            params['num_gpus'] = len(params['gpu_ids'])
            
            if self.validate_params(params):
                valid_combinations.append(params)
            else:
                print(f"⚠ Skipping invalid combination: mode={params['mode']}, gpu_ids={params['gpu_ids']}")
        
        self.total_tasks = len(valid_combinations)
        
        print("\n" + "="*100)
        print(f"Preparing to run {self.total_tasks} training tasks")
        print(f"All tasks will run in clean environment (no custom env vars)")
        print("="*100)
        
        # 打印参数网格
        print("\nParameter Grid:")
        for key, vals in param_grid.items():
            print(f"  {key}: {vals}")
        
        print(f"\nValid combinations: {self.total_tasks} out of {len(all_combinations)}")
        
        # 运行所有有效组合
        for params in valid_combinations:
            # 生成实验名称
            exp_name = self.generate_experiment_name(params)
            
            # 检查是否已存在
            checkpoint_path = os.path.join(self.base_checkpoint_dir, f"{exp_name}.pth")
            if os.path.exists(checkpoint_path):
                print(f"\n⚠ Skipping existing experiment: {exp_name}")
                self.current_task += 1
                continue
            
            # 运行训练
            result = self.run_training(params, exp_name)
            self.results.append(result)
            
            # 保存中间结果
            self.save_results()
            
            # 在并行训练模式后添加短暂延迟，确保资源完全释放
            if params['mode'] in ['dp', 'ddp', 'mp', 'hybrid']:
                delay = 10 if params['mode'] in ['ddp', 'hybrid'] else 5
                print(f"\n⏳ Waiting {delay}s for resources to be released...")
                time.sleep(delay)
        
        # 打印最终总结
        self.print_summary()
    
    def save_results(self):
        """保存结果到JSON文件"""
        results_file = os.path.join(self.log_dir, 'training_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_tasks': self.total_tasks,
                'completed_tasks': self.current_task,
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """打印训练总结"""
        print("\n" + "="*100)
        print(" "*40 + "Training Summary")
        print("="*100)
        
        successful = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - successful
        total_time = sum(r['elapsed_time'] for r in self.results)
        
        print(f"\nTotal Tasks: {self.total_tasks}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total Time: {total_time/3600:.2f} hours")
        
        # 按模式分组统计
        mode_stats = {}
        for result in self.results:
            mode = result['params']['mode']
            if mode not in mode_stats:
                mode_stats[mode] = {'success': 0, 'failed': 0, 'total_time': 0}
            
            if result['success']:
                mode_stats[mode]['success'] += 1
            else:
                mode_stats[mode]['failed'] += 1
            mode_stats[mode]['total_time'] += result['elapsed_time']
        
        print("\nStatistics by Mode:")
        print("-"*60)
        for mode, stats in mode_stats.items():
            print(f"{mode.upper():<10} Success: {stats['success']:<3} Failed: {stats['failed']:<3} "
                  f"Time: {stats['total_time']/3600:.2f}h")
        
        print("\n" + "-"*100)
        print(f"{'Experiment Name':<60} {'Status':<10} {'Time':<15}")
        print("-"*100)
        
        for result in self.results:
            status = "✓ Success" if result['success'] else "✗ Failed"
            time_str = result['elapsed_time_str']
            print(f"{result['experiment_name']:<60} {status:<10} {time_str:<15}")
        
        print("="*100)
        
        # 保存总结到文件
        summary_file = os.path.join(self.log_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write(" "*40 + "Training Summary\n")
            f.write("="*100 + "\n\n")
            f.write(f"Total Tasks: {self.total_tasks}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Total Time: {total_time/3600:.2f} hours\n\n")
            
            f.write("Statistics by Mode:\n")
            f.write("-"*60 + "\n")
            for mode, stats in mode_stats.items():
                f.write(f"{mode.upper():<10} Success: {stats['success']:<3} Failed: {stats['failed']:<3} "
                       f"Time: {stats['total_time']/3600:.2f}h\n")
            
            f.write("\n" + "-"*100 + "\n")
            f.write(f"{'Experiment Name':<60} {'Status':<10} {'Time':<15}\n")
            f.write("-"*100 + "\n")
            for result in self.results:
                status = "Success" if result['success'] else "Failed"
                time_str = result['elapsed_time_str']
                f.write(f"{result['experiment_name']:<60} {status:<10} {time_str:<15}\n")
        
        print(f"\n✓ Summary saved to: {summary_file}")


def get_quick_config():
    """快速测试配置 - 少量epoch用于测试"""
    return {
        'mode': ['single', 'dp', 'mp'],
        'model': ['resnet18'],
        'batch_size': [128],
        'epochs': [10],
        'dataset': ['cifar100'],
        'gpu_ids': [[0], [0, 1]],
        'num_workers': [4],
        'prefetch_factor': [2]
    }


def get_full_config():
    """完整实验配置 - 对比不同训练模式、模型和batch size"""
    return {
        'mode': ['single', 'dp', 'ddp', 'mp'],
        'model': ['resnet18', 'resnet34', 'resnet50'],
        'batch_size': [64, 128, 256],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0], [0, 1], [0, 1, 2, 3]],
        'num_workers': [0, 1, 2, 4],
        'prefetch_factor': [1]
    }


def get_mode_comparison_config():
    """训练模式对比配置 - 相同参数不同模式"""
    return {
        'mode': ['single', 'dp', 'ddp', 'mp'],
        'model': ['resnet18'],
        'batch_size': [128],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0], [0, 1], [0, 1, 2, 3]],
        'num_workers': [4],
        'prefetch_factor': [2]
    }


def get_batch_size_comparison_config():
    """Batch Size对比配置"""
    return {
        'mode': ['single'],
        'model': ['resnet18'],
        'batch_size': [32, 64, 128, 256, 512],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0]],
        'num_workers': [4],
        'prefetch_factor': [2]
    }


def get_model_comparison_config():
    """模型架构对比配置"""
    return {
        'mode': ['single'],
        'model': ['resnet18', 'resnet34', 'resnet50'],
        'batch_size': [128],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0]],
        'num_workers': [4],
        'prefetch_factor': [2]
    }


def get_gpu_scaling_config():
    """GPU扩展性测试配置 - 测试不同GPU数量的性能"""
    return {
        'mode': ['single', 'dp', 'ddp'],
        'model': ['resnet18'],
        'batch_size': [128],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0], [0, 1], [0, 1, 2, 3]],
        'num_workers': [4],
        'prefetch_factor': [2]
    }


def get_parallel_methods_config():
    """并行方法对比配置 - 对比DP、DDP、MP在相同GPU数量下的表现"""
    return {
        'mode': ['dp', 'ddp', 'mp'],
        'model': ['resnet18'],
        'batch_size': [128],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0, 1]],
        'num_workers': [4],
        'prefetch_factor': [2]
    }


def get_dataloader_tuning_config():
    """DataLoader参数调优配置 - 测试不同num_workers和prefetch_factor的影响"""
    return {
        'mode': ['single'],
        'model': ['resnet18'],
        'batch_size': [128],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0]],
        'num_workers': [0, 2, 4, 8, 16],
        'prefetch_factor': [2, 4, 8]
    }


def get_dataloader_workers_config():
    """DataLoader workers对比配置 - 只测试num_workers的影响"""
    return {
        'mode': ['single'],
        'model': ['resnet18'],
        'batch_size': [128],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0]],
        'num_workers': [0, 2, 4, 8, 16],
        'prefetch_factor': [2]
    }


def get_dataloader_prefetch_config():
    """DataLoader prefetch对比配置 - 只测试prefetch_factor的影响"""
    return {
        'mode': ['single'],
        'model': ['resnet18'],
        'batch_size': [128],
        'epochs': [100],
        'dataset': ['cifar100'],
        'gpu_ids': [[0]],
        'num_workers': [4],
        'prefetch_factor': [1, 2, 4, 8, 16]
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Batch Training Script')
    
    parser.add_argument('--config', type=str, default='quick',
                       choices=['quick', 'full', 'mode_comp', 'batch_comp', 
                               'model_comp', 'gpu_scaling', 'parallel_comp',
                               'dataloader_tuning', 'dataloader_workers', 'dataloader_prefetch',
                               'custom'],
                       help='Predefined config')
    
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint save directory')
    
    parser.add_argument('--log-dir', type=str, default='./training_logs',
                       help='Log save directory')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 选择配置
    if args.config == 'quick':
        param_grid = get_quick_config()
        print("\nUsing config: Quick Test (10 epochs)")
    elif args.config == 'full':
        param_grid = get_full_config()
        print("\nUsing config: Full Experiments")
    elif args.config == 'mode_comp':
        param_grid = get_mode_comparison_config()
        print("\nUsing config: Training Mode Comparison")
    elif args.config == 'batch_comp':
        param_grid = get_batch_size_comparison_config()
        print("\nUsing config: Batch Size Comparison")
    elif args.config == 'model_comp':
        param_grid = get_model_comparison_config()
        print("\nUsing config: Model Architecture Comparison")
    elif args.config == 'gpu_scaling':
        param_grid = get_gpu_scaling_config()
        print("\nUsing config: GPU Scaling Test")
    elif args.config == 'parallel_comp':
        param_grid = get_parallel_methods_config()
        print("\nUsing config: Parallel Methods Comparison (DP vs DDP vs MP)")
    elif args.config == 'dataloader_tuning':
        param_grid = get_dataloader_tuning_config()
        print("\nUsing config: DataLoader Parameter Tuning")
    elif args.config == 'dataloader_workers':
        param_grid = get_dataloader_workers_config()
        print("\nUsing config: DataLoader Workers Comparison")
    elif args.config == 'dataloader_prefetch':
        param_grid = get_dataloader_prefetch_config()
        print("\nUsing config: DataLoader Prefetch Factor Comparison")
    elif args.config == 'custom':
        param_grid = {
            'mode': ['single', 'dp', 'mp'],
            'model': ['resnet18'],
            'batch_size': [128, 256],
            'epochs': [100],
            'dataset': ['cifar100'],
            'gpu_ids': [[0], [0, 1]],
            'num_workers': [4, 8],
            'prefetch_factor': [2, 4]
        }
        print("\nUsing config: Custom")
    
    # 创建调度器并运行
    scheduler = TrainingScheduler(
        base_checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    scheduler.run_experiments(param_grid)
    
    print("\nAll training tasks completed!")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print(f"Logs saved in: {args.log_dir}")


if __name__ == '__main__':
    main()