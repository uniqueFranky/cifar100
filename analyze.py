"""
实验结果分析脚本
分析不同训练配置下的性能指标，包括耗时、准确率、内存占用等
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd

# 设置matplotlib支持中文显示（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置全局字体大小
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, experiments_dir='./experiments'):
        self.experiments_dir = Path(experiments_dir)
        self.results = {}
        self.checkpoints_data = {}
        
    def load_all_experiments(self):
        """加载所有实验结果"""
        print("Loading experiment results...")
        
        for group_dir in sorted(self.experiments_dir.iterdir()):
            if not group_dir.is_dir() or group_dir.name.startswith('.'):
                continue
                
            group_name = group_dir.name
            print(f"\nProcessing {group_name}...")
            
            # 加载results.json
            results_file = group_dir / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    self.results[group_name] = json.load(f)
            
            # 加载所有checkpoint文件
            checkpoints_dir = group_dir / 'checkpoints'
            if checkpoints_dir.exists():
                self.checkpoints_data[group_name] = {}
                for ckpt_file in checkpoints_dir.glob('*.pth'):
                    try:
                        ckpt = torch.load(ckpt_file, map_location='cpu')
                        self.checkpoints_data[group_name][ckpt_file.stem] = ckpt
                        print(f"  Loaded {ckpt_file.name}")
                    except Exception as e:
                        print(f"  Error loading {ckpt_file.name}: {e}")
        
        print(f"\nTotal groups loaded: {len(self.results)}")
        
    def extract_metrics(self):
        """从checkpoint中提取关键指标"""
        metrics_data = {}
        
        for group_name, checkpoints in self.checkpoints_data.items():
            metrics_data[group_name] = {}
            
            for ckpt_name, ckpt in checkpoints.items():
                history = ckpt.get('history', {})
                config = ckpt.get('config', {})
                
                # 提取配置信息
                training_mode = self._extract_training_mode(ckpt_name)
                model_name = config.get('model', 'unknown')
                batch_size = config.get('batch_size', 0)
                num_gpus = config.get('num_gpus', 1)
                num_workers = config.get('num_workers', 0)
                chunks = config.get('chunks', 0)
                
                # 提取性能指标
                final_train_acc = history.get('train_acc', [0])[-1] if history.get('train_acc') else 0
                final_test_acc = history.get('test_acc', [0])[-1] if history.get('test_acc') else 0
                best_test_acc = max(history.get('test_acc', [0])) if history.get('test_acc') else 0
                
                # 计算平均epoch时间
                epoch_times = history.get('epoch_time', [])
                avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
                total_time = sum(epoch_times) if epoch_times else 0
                
                # 提取GPU内存使用（处理不同格式）
                gpu_memory = history.get('gpu_memory_per_device', [])
                max_gpu_memory = self._extract_max_gpu_memory(gpu_memory)
                avg_gpu_memory = self._extract_avg_gpu_memory(gpu_memory)
                
                # 计算收敛速度（达到某个准确率阈值的epoch数）
                convergence_epoch = self._calculate_convergence_epoch(
                    history.get('test_acc', []), threshold=60.0
                )
                
                metrics_data[group_name][ckpt_name] = {
                    'training_mode': training_mode,
                    'model': model_name,
                    'batch_size': batch_size,
                    'num_gpus': num_gpus,
                    'num_workers': num_workers,
                    'chunks': chunks,
                    'final_train_acc': final_train_acc,
                    'final_test_acc': final_test_acc,
                    'best_test_acc': best_test_acc,
                    'avg_epoch_time': avg_epoch_time,
                    'total_time': total_time,
                    'max_gpu_memory_allocated': max_gpu_memory['allocated'],
                    'max_gpu_memory_reserved': max_gpu_memory['reserved'],
                    'avg_gpu_memory_allocated': avg_gpu_memory['allocated'],
                    'avg_gpu_memory_reserved': avg_gpu_memory['reserved'],
                    'convergence_epoch': convergence_epoch,
                    'history': history,
                    'config': config
                }
        
        return metrics_data
    
    def _extract_training_mode(self, ckpt_name):
        """从checkpoint名称中提取训练模式"""
        if ckpt_name.startswith('single_'):
            return 'Single-GPU'
        elif ckpt_name.startswith('ddp_'):
            return 'DDP'
        elif ckpt_name.startswith('dp_'):
            return 'DP'
        elif ckpt_name.startswith('mp_'):
            return 'MP'
        elif ckpt_name.startswith('pp_'):
            return 'PP'
        elif ckpt_name.startswith('hp_'):
            return 'HP'
        else:
            return 'Unknown'
    
    def _extract_max_gpu_memory(self, gpu_memory_list):
        """提取最大GPU内存使用"""
        max_allocated = 0
        max_reserved = 0
        
        if not gpu_memory_list:
            return {'allocated': 0, 'reserved': 0}
        
        for epoch_mem in gpu_memory_list:
            if isinstance(epoch_mem, list):
                # 格式: [{'device_id': 0, 'allocated': x, 'reserved': y}, ...]
                for device_mem in epoch_mem:
                    if isinstance(device_mem, dict):
                        max_allocated = max(max_allocated, device_mem.get('allocated', 0))
                        max_reserved = max(max_reserved, device_mem.get('reserved', 0))
            elif isinstance(epoch_mem, dict):
                # 旧格式兼容
                max_allocated = max(max_allocated, epoch_mem.get('allocated', 0))
                max_reserved = max(max_reserved, epoch_mem.get('reserved', 0))
        
        return {'allocated': max_allocated, 'reserved': max_reserved}
    
    def _extract_avg_gpu_memory(self, gpu_memory_list):
        """提取平均GPU内存使用"""
        allocated_list = []
        reserved_list = []
        
        if not gpu_memory_list:
            return {'allocated': 0, 'reserved': 0}
        
        for epoch_mem in gpu_memory_list:
            if isinstance(epoch_mem, list):
                # 对于多GPU，计算所有GPU的平均值
                epoch_allocated = []
                epoch_reserved = []
                for device_mem in epoch_mem:
                    if isinstance(device_mem, dict):
                        epoch_allocated.append(device_mem.get('allocated', 0))
                        epoch_reserved.append(device_mem.get('reserved', 0))
                if epoch_allocated:
                    allocated_list.append(np.mean(epoch_allocated))
                    reserved_list.append(np.mean(epoch_reserved))
            elif isinstance(epoch_mem, dict):
                allocated_list.append(epoch_mem.get('allocated', 0))
                reserved_list.append(epoch_mem.get('reserved', 0))
        
        return {
            'allocated': np.mean(allocated_list) if allocated_list else 0,
            'reserved': np.mean(reserved_list) if reserved_list else 0
        }
    
    def _calculate_convergence_epoch(self, test_acc_list, threshold=60.0):
        """计算达到指定准确率阈值的epoch数"""
        for epoch, acc in enumerate(test_acc_list):
            if acc >= threshold:
                return epoch + 1
        return len(test_acc_list) if test_acc_list else 0
    
    def analyze_group_1(self, metrics_data, output_dir):
        """分析组1: 训练模式和硬件配置比较"""
        print("\n" + "="*60)
        print("Analyzing Group 1: Training Mode and Hardware Configuration")
        print("="*60)
        
        group_name = 'group_1_training_mode_and_hardware_configuration_comparison'
        if group_name not in metrics_data:
            print(f"Group {group_name} not found!")
            return
        
        data = metrics_data[group_name]
        
        # 创建DataFrame
        df_list = []
        for ckpt_name, metrics in data.items():
            df_list.append({
                'Name': ckpt_name,
                'Training Mode': metrics['training_mode'],
                'GPUs': metrics['num_gpus'],
                'Best Test Acc (%)': metrics['best_test_acc'],
                'Avg Epoch Time (s)': metrics['avg_epoch_time'],
                'Total Time (s)': metrics['total_time'],
                'Max GPU Memory (GB)': metrics['max_gpu_memory_allocated'],
                'Convergence Epoch': metrics['convergence_epoch']
            })
        df = pd.DataFrame(df_list)
        
        # 打印统计表
        print("\n" + df.to_string(index=False))
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Group 1: Training Mode and Hardware Configuration Comparison', 
                     fontsize=14, fontweight='bold')
        
        # 1. 训练模式对比 - 准确率
        ax = axes[0, 0]
        df_sorted = df.sort_values('Training Mode')
        colors = sns.color_palette("husl", len(df_sorted))
        bars = ax.bar(range(len(df_sorted)), df_sorted['Best Test Acc (%)'], color=colors)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Best Test Accuracy (%)')
        ax.set_title('Test Accuracy Comparison')
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([f"{row['Training Mode']}\n{row['GPUs']}GPU" 
                            for _, row in df_sorted.iterrows()], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
        
        # 2. 训练时间对比
        ax = axes[0, 1]
        bars = ax.bar(range(len(df_sorted)), df_sorted['Avg Epoch Time (s)'], color=colors)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Average Epoch Time (s)')
        ax.set_title('Training Speed Comparison')
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([f"{row['Training Mode']}\n{row['GPUs']}GPU" 
                            for _, row in df_sorted.iterrows()], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        
        # 3. GPU内存使用对比
        ax = axes[0, 2]
        bars = ax.bar(range(len(df_sorted)), df_sorted['Max GPU Memory (GB)'], color=colors)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Max GPU Memory per Device (GB)')
        ax.set_title('GPU Memory Usage Comparison')
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([f"{row['Training Mode']}\n{row['GPUs']}GPU" 
                            for _, row in df_sorted.iterrows()], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}GB', ha='center', va='bottom', fontsize=8)
        
        # 4. 收敛速度对比
        ax = axes[1, 0]
        bars = ax.bar(range(len(df_sorted)), df_sorted['Convergence Epoch'], color=colors)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Epochs to 60% Accuracy')
        ax.set_title('Convergence Speed Comparison')
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([f"{row['Training Mode']}\n{row['GPUs']}GPU" 
                            for _, row in df_sorted.iterrows()], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 5. 训练曲线对比
        ax = axes[1, 1]
        for ckpt_name, metrics in data.items():
            history = metrics['history']
            test_acc = history.get('test_acc', [])
            if test_acc:
                label = f"{metrics['training_mode']}-{metrics['num_gpus']}GPU"
                ax.plot(range(1, len(test_acc)+1), test_acc, marker='o', 
                       markersize=3, label=label, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Training Curves Comparison')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
        
        # 6. 效率分析 (准确率/时间)
        ax = axes[1, 2]
        efficiency = df_sorted['Best Test Acc (%)'] / df_sorted['Avg Epoch Time (s)']
        bars = ax.bar(range(len(df_sorted)), efficiency, color=colors)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Accuracy / Time (%/s)')
        ax.set_title('Training Efficiency (Higher is Better)')
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([f"{row['Training Mode']}\n{row['GPUs']}GPU" 
                            for _, row in df_sorted.iterrows()], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = output_dir / 'group1_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {output_path}")
        plt.close()
        
    def analyze_group_2(self, metrics_data, output_dir):
        """分析组2: 模型复杂度影响"""
        print("\n" + "="*60)
        print("Analyzing Group 2: Model Complexity Impact")
        print("="*60)
        
        group_name = 'group_2_model_complexity_impact_across_training_modes'
        if group_name not in metrics_data:
            print(f"Group {group_name} not found!")
            return
        
        data = metrics_data[group_name]
        
        # 创建DataFrame
        df_list = []
        for ckpt_name, metrics in data.items():
            df_list.append({
                'Name': ckpt_name,
                'Training Mode': metrics['training_mode'],
                'Model': metrics['model'],
                'Best Test Acc (%)': metrics['best_test_acc'],
                'Avg Epoch Time (s)': metrics['avg_epoch_time'],
                'Max GPU Memory (GB)': metrics['max_gpu_memory_allocated']
            })
        df = pd.DataFrame(df_list)
        
        print("\n" + df.to_string(index=False))
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Group 2: Model Complexity Impact Across Training Modes', 
                     fontsize=14, fontweight='bold')
        
        # 按训练模式分组
        for mode in df['Training Mode'].unique():
            df_mode = df[df['Training Mode'] == mode].sort_values('Model')
            
            # 1. 准确率对比
            ax = axes[0, 0]
            ax.plot(df_mode['Model'], df_mode['Best Test Acc (%)'], 
                   marker='o', label=mode, linewidth=2, markersize=8)
            
            # 2. 训练时间对比
            ax = axes[0, 1]
            ax.plot(df_mode['Model'], df_mode['Avg Epoch Time (s)'], 
                   marker='s', label=mode, linewidth=2, markersize=8)
            
            # 3. GPU内存对比
            ax = axes[1, 0]
            ax.plot(df_mode['Model'], df_mode['Max GPU Memory (GB)'], 
                   marker='^', label=mode, linewidth=2, markersize=8)
        
        # 设置图表属性
        axes[0, 0].set_xlabel('Model Architecture')
        axes[0, 0].set_ylabel('Best Test Accuracy (%)')
        axes[0, 0].set_title('Accuracy vs Model Complexity')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].set_xlabel('Model Architecture')
        axes[0, 1].set_ylabel('Average Epoch Time (s)')
        axes[0, 1].set_title('Training Time vs Model Complexity')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].set_xlabel('Model Architecture')
        axes[1, 0].set_ylabel('Max GPU Memory (GB)')
        axes[1, 0].set_title('GPU Memory vs Model Complexity')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. 综合对比热图
        ax = axes[1, 1]
        pivot_data = df.pivot(index='Model', columns='Training Mode', 
                             values='Best Test Acc (%)')
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', 
                   ax=ax, cbar_kws={'label': 'Test Accuracy (%)'})
        ax.set_title('Accuracy Heatmap: Model vs Training Mode')
        ax.set_xlabel('Training Mode')
        ax.set_ylabel('Model Architecture')
        
        plt.tight_layout()
        output_path = output_dir / 'group2_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {output_path}")
        plt.close()
        
    def analyze_group_3(self, metrics_data, output_dir):
        """分析组3: DataLoader优化"""
        print("\n" + "="*60)
        print("Analyzing Group 3: DataLoader Optimization")
        print("="*60)
        
        group_name = 'group_3_dataloader_optimization_across_training_modes'
        if group_name not in metrics_data:
            print(f"Group {group_name} not found!")
            return
        
        data = metrics_data[group_name]
        
        # 创建DataFrame
        df_list = []
        for ckpt_name, metrics in data.items():
            df_list.append({
                'Name': ckpt_name,
                'Training Mode': metrics['training_mode'],
                'Num Workers': metrics['num_workers'],
                'Best Test Acc (%)': metrics['best_test_acc'],
                'Avg Epoch Time (s)': metrics['avg_epoch_time']
            })
        df = pd.DataFrame(df_list)
        
        print("\n" + df.to_string(index=False))
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Group 3: DataLoader Optimization (num_workers Impact)', 
                     fontsize=14, fontweight='bold')
        
        # 按训练模式分组
        for mode in df['Training Mode'].unique():
            df_mode = df[df['Training Mode'] == mode].sort_values('Num Workers')
            
            # 1. 训练时间 vs num_workers
            ax = axes[0]
            ax.plot(df_mode['Num Workers'], df_mode['Avg Epoch Time (s)'], 
                   marker='o', label=mode, linewidth=2, markersize=8)
            
            # 2. 准确率 vs num_workers
            ax = axes[1]
            ax.plot(df_mode['Num Workers'], df_mode['Best Test Acc (%)'], 
                   marker='s', label=mode, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Number of Workers')
        axes[0].set_ylabel('Average Epoch Time (s)')
        axes[0].set_title('Training Time vs num_workers')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_xscale('symlog')  # 使用对称对数刻度以处理0值
        
        axes[1].set_xlabel('Number of Workers')
        axes[1].set_ylabel('Best Test Accuracy (%)')
        axes[1].set_title('Accuracy vs num_workers')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xscale('symlog')
        
        plt.tight_layout()
        output_path = output_dir / 'group3_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {output_path}")
        plt.close()
        
    def analyze_group_4(self, metrics_data, output_dir):
        """分析组4: Batch Size影响"""
        print("\n" + "="*60)
        print("Analyzing Group 4: Batch Size Impact")
        print("="*60)
        
        group_name = 'group_4_batch_size_impact_across_training_modes_and_hardware'
        if group_name not in metrics_data:
            print(f"Group {group_name} not found!")
            return
        
        data = metrics_data[group_name]
        
        # 创建DataFrame
        df_list = []
        for ckpt_name, metrics in data.items():
            df_list.append({
                'Name': ckpt_name,
                'Training Mode': metrics['training_mode'],
                'Batch Size': metrics['batch_size'],
                'Best Test Acc (%)': metrics['best_test_acc'],
                'Avg Epoch Time (s)': metrics['avg_epoch_time'],
                'Max GPU Memory (GB)': metrics['max_gpu_memory_allocated']
            })
        df = pd.DataFrame(df_list)
        
        print("\n" + df.to_string(index=False))
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Group 4: Batch Size Impact Across Training Modes', 
                     fontsize=14, fontweight='bold')
        
        # 按训练模式分组
        for mode in df['Training Mode'].unique():
            df_mode = df[df['Training Mode'] == mode].sort_values('Batch Size')
            
            # 1. 准确率 vs batch size
            ax = axes[0, 0]
            ax.plot(df_mode['Batch Size'], df_mode['Best Test Acc (%)'], 
                   marker='o', label=mode, linewidth=2, markersize=8)
            
            # 2. 训练时间 vs batch size
            ax = axes[0, 1]
            ax.plot(df_mode['Batch Size'], df_mode['Avg Epoch Time (s)'], 
                   marker='s', label=mode, linewidth=2, markersize=8)
            
            # 3. GPU内存 vs batch size
            ax = axes[1, 0]
            ax.plot(df_mode['Batch Size'], df_mode['Max GPU Memory (GB)'], 
                   marker='^', label=mode, linewidth=2, markersize=8)
        
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Best Test Accuracy (%)')
        axes[0, 0].set_title('Accuracy vs Batch Size')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Average Epoch Time (s)')
        axes[0, 1].set_title('Training Time vs Batch Size')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Max GPU Memory (GB)')
        axes[1, 0].set_title('GPU Memory vs Batch Size')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. 吞吐量分析 (samples/second)
        ax = axes[1, 1]
        for mode in df['Training Mode'].unique():
            df_mode = df[df['Training Mode'] == mode].sort_values('Batch Size')
            # 假设每个epoch有50000个样本（CIFAR-100训练集大小）
            throughput = 50000 / df_mode['Avg Epoch Time (s)']
            ax.plot(df_mode['Batch Size'], throughput, 
                   marker='D', label=mode, linewidth=2, markersize=8)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (samples/s)')
        ax.set_title('Training Throughput vs Batch Size')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'group4_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {output_path}")
        plt.close()
        
    def analyze_group_5(self, metrics_data, output_dir):
        """分析组5: Pipeline Parallel优化 - 包含详细的loss和准确率曲线"""
        print("\n" + "="*60)
        print("Analyzing Group 5: Pipeline Parallel Optimization with Loss Curves")
        print("="*60)
        
        group_name = 'group_5_pipeline_parallel_chunks_parameter_optimization'
        if group_name not in metrics_data:
            print(f"Group {group_name} not found!")
            return
        
        data = metrics_data[group_name]
        
        if not data:
            print("No data available for Group 5 (checkpoints may be empty)")
            return
        
        # 创建DataFrame和收集训练历史
        df_list = []
        chunks_history = {}
        
        for ckpt_name, metrics in data.items():
            df_list.append({
                'Name': ckpt_name,
                'Chunks': metrics['chunks'],
                'Best Test Acc (%)': metrics['best_test_acc'],
                'Final Test Acc (%)': metrics['final_test_acc'],
                'Avg Epoch Time (s)': metrics['avg_epoch_time'],
                'Max GPU Memory (GB)': metrics['max_gpu_memory_allocated'],
                'Convergence Epoch': metrics['convergence_epoch']
            })
            
            # 收集训练历史数据
            history = metrics.get('history', {})
            if history and all(key in history for key in ['train_loss', 'test_loss', 'train_acc', 'test_acc']):
                chunks_history[metrics['chunks']] = {
                    'train_loss': history['train_loss'],
                    'test_loss': history['test_loss'],
                    'train_acc': history['train_acc'],
                    'test_acc': history['test_acc'],
                    'epoch_time': history.get('epoch_time', []),
                    'name': ckpt_name
                }
        
        df = pd.DataFrame(df_list).sort_values('Chunks')
        print("\n" + df.to_string(index=False))
        
        # 创建综合分析图 - 3x3布局
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Group 5: Pipeline Parallel Chunks Parameter Optimization - Complete Analysis', 
                    fontsize=16, fontweight='bold')
        
        colors = sns.color_palette("husl", len(chunks_history) if chunks_history else len(df))
        
        # 第一行：训练曲线
        if chunks_history:
            # 1. 训练Loss曲线
            ax = axes[0, 0]
            for i, (chunks, data) in enumerate(sorted(chunks_history.items())):
                epochs = range(1, len(data['train_loss']) + 1)
                ax.plot(epochs, data['train_loss'], 
                    label=f'Chunks={chunks}', 
                    color=colors[i], 
                    linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Training Loss')
            ax.set_title('Training Loss Curves')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 2. 测试Loss曲线
            ax = axes[0, 1]
            for i, (chunks, data) in enumerate(sorted(chunks_history.items())):
                epochs = range(1, len(data['test_loss']) + 1)
                ax.plot(epochs, data['test_loss'], 
                    label=f'Chunks={chunks}', 
                    color=colors[i], 
                    linewidth=2, marker='s', markersize=3)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Test Loss')
            ax.set_title('Test Loss Curves')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 3. 测试准确率曲线
            ax = axes[0, 2]
            for i, (chunks, data) in enumerate(sorted(chunks_history.items())):
                epochs = range(1, len(data['test_acc']) + 1)
                ax.plot(epochs, data['test_acc'], 
                    label=f'Chunks={chunks}', 
                    color=colors[i], 
                    linewidth=2, marker='^', markersize=3)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Test Accuracy (%)')
            ax.set_title('Test Accuracy Curves')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            # 如果没有训练历史数据，显示提示信息
            for i in range(3):
                axes[0, i].text(0.5, 0.5, 'No training history available', 
                            ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].set_title(['Training Loss', 'Test Loss', 'Test Accuracy'][i])
        
        # 第二行：性能指标对比
        # 4. 最佳准确率对比
        ax = axes[1, 0]
        bars = ax.bar(range(len(df)), df['Best Test Acc (%)'], color=colors[:len(df)])
        ax.set_xlabel('Chunks')
        ax.set_ylabel('Best Test Accuracy (%)')
        ax.set_title('Best Accuracy vs Chunks')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Chunks'])
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
        
        # 5. 训练时间对比
        ax = axes[1, 1]
        bars = ax.bar(range(len(df)), df['Avg Epoch Time (s)'], color=colors[:len(df)])
        ax.set_xlabel('Chunks')
        ax.set_ylabel('Average Epoch Time (s)')
        ax.set_title('Training Time vs Chunks')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Chunks'])
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        
        # 6. GPU内存使用对比
        ax = axes[1, 2]
        bars = ax.bar(range(len(df)), df['Max GPU Memory (GB)'], color=colors[:len(df)])
        ax.set_xlabel('Chunks')
        ax.set_ylabel('Max GPU Memory (GB)')
        ax.set_title('GPU Memory vs Chunks')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Chunks'])
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}GB', ha='center', va='bottom', fontsize=8)
        
        # 第三行：综合分析
        # 7. 收敛速度对比
        ax = axes[2, 0]
        bars = ax.bar(range(len(df)), df['Convergence Epoch'], color=colors[:len(df)])
        ax.set_xlabel('Chunks')
        ax.set_ylabel('Epochs to 60% Accuracy')
        ax.set_title('Convergence Speed vs Chunks')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Chunks'])
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 8. 训练效率分析 (准确率/时间)
        ax = axes[2, 1]
        efficiency = df['Best Test Acc (%)'] / df['Avg Epoch Time (s)']
        bars = ax.bar(range(len(df)), efficiency, color=colors[:len(df)])
        ax.set_xlabel('Chunks')
        ax.set_ylabel('Efficiency (Accuracy/Time)')
        ax.set_title('Training Efficiency vs Chunks')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Chunks'])
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 9. 综合性能雷达图或对比表格
        ax = axes[2, 2]
        if len(df) > 0:
            # 创建综合性能对比表格
            ax.axis('tight')
            ax.axis('off')
            
            # 准备表格数据
            table_data = []
            headers = ['Chunks', 'Best Acc (%)', 'Time (s)', 'Memory (GB)', 'Efficiency']
            
            for _, row in df.iterrows():
                eff = row['Best Test Acc (%)'] / row['Avg Epoch Time (s)']
                table_data.append([
                    f"{row['Chunks']}",
                    f"{row['Best Test Acc (%)']:.2f}",
                    f"{row['Avg Epoch Time (s)']:.1f}",
                    f"{row['Max GPU Memory (GB)']:.2f}",
                    f"{eff:.2f}"
                ])
            
            # 创建表格
            table = ax.table(cellText=table_data,
                            colLabels=headers,
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # 设置表格样式
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#E6E6FA')
                table[(0, i)].set_text_props(weight='bold')
            
            # 高亮最佳值
            best_acc_row = df['Best Test Acc (%)'].idxmax() + 1
            best_time_row = df['Avg Epoch Time (s)'].idxmin() + 1
            best_memory_row = df['Max GPU Memory (GB)'].idxmin() + 1
            best_eff_row = efficiency.idxmax() + 1
            
            table[(best_acc_row, 1)].set_facecolor('#90EE90')  # 浅绿色
            table[(best_time_row, 2)].set_facecolor('#90EE90')
            table[(best_memory_row, 3)].set_facecolor('#90EE90')
            table[(best_eff_row, 4)].set_facecolor('#90EE90')
            
            ax.set_title('Performance Summary\n(Green = Best)', fontsize=10, pad=20)
        
        plt.tight_layout()
        output_path = output_dir / 'group5_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved Group 5 complete analysis to {output_path}")
        plt.close()


    
    def generate_overall_summary(self, metrics_data, output_dir):
        """生成总体对比分析"""
        print("\n" + "="*60)
        print("Generating Overall Summary")
        print("="*60)
        
        # 收集所有实验的关键指标
        all_experiments = []
        for group_name, experiments in metrics_data.items():
            for exp_name, metrics in experiments.items():
                all_experiments.append({
                    'Group': group_name.replace('_', ' ').title(),
                    'Experiment': exp_name,
                    'Training Mode': metrics['training_mode'],
                    'Model': metrics['model'],
                    'GPUs': metrics['num_gpus'],
                    'Batch Size': metrics['batch_size'],
                    'Workers': metrics['num_workers'],
                    'Best Acc (%)': metrics['best_test_acc'],
                    'Avg Time (s)': metrics['avg_epoch_time'],
                    'Max Memory (GB)': metrics['max_gpu_memory_allocated']
                })
        
        df_all = pd.DataFrame(all_experiments)
        
        # 创建综合对比图
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Overall Performance Summary Across All Experiments', 
                     fontsize=16, fontweight='bold')
        
        # 1. 训练模式对比 - 准确率
        ax1 = fig.add_subplot(gs[0, 0])
        mode_acc = df_all.groupby('Training Mode')['Best Acc (%)'].agg(['mean', 'std'])
        mode_acc = mode_acc.sort_values('mean', ascending=False)
        bars = ax1.bar(range(len(mode_acc)), mode_acc['mean'], 
                      yerr=mode_acc['std'], capsize=5,
                      color=sns.color_palette("husl", len(mode_acc)))
        ax1.set_xlabel('Training Mode')
        ax1.set_ylabel('Average Best Accuracy (%)')
        ax1.set_title('Average Accuracy by Training Mode')
        ax1.set_xticks(range(len(mode_acc)))
        ax1.set_xticklabels(mode_acc.index, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 训练模式对比 - 速度
        ax2 = fig.add_subplot(gs[0, 1])
        mode_time = df_all.groupby('Training Mode')['Avg Time (s)'].agg(['mean', 'std'])
        mode_time = mode_time.sort_values('mean')
        bars = ax2.bar(range(len(mode_time)), mode_time['mean'], 
                      yerr=mode_time['std'], capsize=5,
                      color=sns.color_palette("husl", len(mode_time)))
        ax2.set_xlabel('Training Mode')
        ax2.set_ylabel('Average Epoch Time (s)')
        ax2.set_title('Average Training Time by Mode')
        ax2.set_xticks(range(len(mode_time)))
        ax2.set_xticklabels(mode_time.index, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. 训练模式对比 - 内存
        ax3 = fig.add_subplot(gs[0, 2])
        mode_mem = df_all.groupby('Training Mode')['Max Memory (GB)'].agg(['mean', 'std'])
        mode_mem = mode_mem.sort_values('mean')
        bars = ax3.bar(range(len(mode_mem)), mode_mem['mean'], 
                      yerr=mode_mem['std'], capsize=5,
                      color=sns.color_palette("husl", len(mode_mem)))
        ax3.set_xlabel('Training Mode')
        ax3.set_ylabel('Average Max Memory (GB)')
        ax3.set_title('Average GPU Memory by Mode')
        ax3.set_xticks(range(len(mode_mem)))
        ax3.set_xticklabels(mode_mem.index, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. GPU数量影响
        ax4 = fig.add_subplot(gs[1, 0])
        gpu_metrics = df_all.groupby('GPUs').agg({
            'Best Acc (%)': 'mean',
            'Avg Time (s)': 'mean'
        })
        ax4_twin = ax4.twinx()
        x = range(len(gpu_metrics))
        line1 = ax4.plot(x, gpu_metrics['Best Acc (%)'], 'o-', 
                        color='blue', label='Accuracy', linewidth=2, markersize=8)
        line2 = ax4_twin.plot(x, gpu_metrics['Avg Time (s)'], 's-', 
                             color='red', label='Time', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of GPUs')
        ax4.set_ylabel('Average Accuracy (%)', color='blue')
        ax4_twin.set_ylabel('Average Time (s)', color='red')
        ax4.set_title('Performance vs Number of GPUs')
        ax4.set_xticks(x)
        ax4.set_xticklabels(gpu_metrics.index)
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        ax4.grid(alpha=0.3)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        # 5. Batch Size影响
        ax5 = fig.add_subplot(gs[1, 1])
        bs_metrics = df_all[df_all['Batch Size'] > 0].groupby('Batch Size').agg({
            'Best Acc (%)': 'mean',
            'Avg Time (s)': 'mean'
        })
        ax5_twin = ax5.twinx()
        x = range(len(bs_metrics))
        line1 = ax5.plot(x, bs_metrics['Best Acc (%)'], 'o-', 
                        color='green', label='Accuracy', linewidth=2, markersize=8)
        line2 = ax5_twin.plot(x, bs_metrics['Avg Time (s)'], 's-', 
                             color='orange', label='Time', linewidth=2, markersize=8)
        ax5.set_xlabel('Batch Size')
        ax5.set_ylabel('Average Accuracy (%)', color='green')
        ax5_twin.set_ylabel('Average Time (s)', color='orange')
        ax5.set_title('Performance vs Batch Size')
        ax5.set_xticks(x)
        ax5.set_xticklabels(bs_metrics.index)
        ax5.tick_params(axis='y', labelcolor='green')
        ax5_twin.tick_params(axis='y', labelcolor='orange')
        ax5.grid(alpha=0.3)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left')
        
        # 6. Workers影响
        ax6 = fig.add_subplot(gs[1, 2])
        worker_metrics = df_all[df_all['Workers'] >= 0].groupby('Workers')['Avg Time (s)'].mean()
        ax6.plot(worker_metrics.index, worker_metrics.values, 'o-', 
                linewidth=2, markersize=8, color='purple')
        ax6.set_xlabel('Number of Workers')
        ax6.set_ylabel('Average Epoch Time (s)')
        ax6.set_title('Training Time vs DataLoader Workers')
        ax6.grid(alpha=0.3)
        ax6.set_xscale('symlog')
        
        # 7. 模型复杂度影响
        ax7 = fig.add_subplot(gs[2, 0])
        model_metrics = df_all.groupby('Model').agg({
            'Best Acc (%)': 'mean',
            'Avg Time (s)': 'mean',
            'Max Memory (GB)': 'mean'
        })
        x = range(len(model_metrics))
        width = 0.25
        ax7.bar([i - width for i in x], model_metrics['Best Acc (%)'], 
               width, label='Accuracy (%)', color='skyblue')
        ax7_twin = ax7.twinx()
        ax7_twin.bar(x, model_metrics['Avg Time (s)'], 
                    width, label='Time (s)', color='lightcoral')
        ax7_twin.bar([i + width for i in x], model_metrics['Max Memory (GB)'], 
                    width, label='Memory (GB)', color='lightgreen')
        ax7.set_xlabel('Model Architecture')
        ax7.set_ylabel('Accuracy (%)', color='skyblue')
        ax7_twin.set_ylabel('Time (s) / Memory (GB)')
        ax7.set_title('Model Complexity Impact')
        ax7.set_xticks(x)
        ax7.set_xticklabels(model_metrics.index, rotation=45, ha='right')
        ax7.legend(loc='upper left')
        ax7_twin.legend(loc='upper right')
        ax7.grid(axis='y', alpha=0.3)
        
        # 8. Top 10最佳配置
        ax8 = fig.add_subplot(gs[2, 1:])
        top10 = df_all.nlargest(10, 'Best Acc (%)')
        y_pos = range(len(top10))
        colors = sns.color_palette("RdYlGn", len(top10))
        bars = ax8.barh(y_pos, top10['Best Acc (%)'], color=colors)
        ax8.set_yticks(y_pos)
        labels = [f"{row['Training Mode']}-{row['Model']}-{row['GPUs']}GPU-BS{row['Batch Size']}" 
                 for _, row in top10.iterrows()]
        ax8.set_yticklabels(labels, fontsize=8)
        ax8.set_xlabel('Best Test Accuracy (%)')
        ax8.set_title('Top 10 Best Configurations')
        ax8.grid(axis='x', alpha=0.3)
        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, top10['Best Acc (%)'])):
            ax8.text(acc, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.2f}%', ha='left', va='center', fontsize=8)
        
        plt.savefig(output_dir / 'overall_summary.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved overall summary to {output_dir / 'overall_summary.png'}")
        plt.close()
        
        # 保存统计表格
        summary_file = output_dir / 'summary_statistics.txt'
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OVERALL PERFORMANCE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. Training Mode Comparison:\n")
            f.write("-" * 80 + "\n")
            mode_summary = df_all.groupby('Training Mode').agg({
                'Best Acc (%)': ['mean', 'std', 'max'],
                'Avg Time (s)': ['mean', 'std', 'min'],
                'Max Memory (GB)': ['mean', 'std', 'min']
            })
            f.write(mode_summary.to_string())
            f.write("\n\n")
            
            f.write("2. Top 10 Best Configurations:\n")
            f.write("-" * 80 + "\n")
            f.write(top10[['Training Mode', 'Model', 'GPUs', 'Batch Size', 
                          'Best Acc (%)', 'Avg Time (s)', 'Max Memory (GB)']].to_string(index=False))
            f.write("\n\n")
            
            f.write("3. Fastest Configurations:\n")
            f.write("-" * 80 + "\n")
            fastest = df_all.nsmallest(10, 'Avg Time (s)')
            f.write(fastest[['Training Mode', 'Model', 'GPUs', 'Batch Size', 
                           'Best Acc (%)', 'Avg Time (s)']].to_string(index=False))
            f.write("\n\n")
            
            f.write("4. Most Memory Efficient:\n")
            f.write("-" * 80 + "\n")
            mem_efficient = df_all.nsmallest(10, 'Max Memory (GB)')
            f.write(mem_efficient[['Training Mode', 'Model', 'GPUs', 'Batch Size', 
                                  'Best Acc (%)', 'Max Memory (GB)']].to_string(index=False))
            f.write("\n")
        
        print(f"Saved summary statistics to {summary_file}")
        
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS ANALYSIS")
        print("="*80)
        
        # 创建输出目录
        output_dir = self.experiments_dir / 'analysis_results'
        output_dir.mkdir(exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        
        # 加载所有实验数据
        self.load_all_experiments()
        
        # 提取指标
        print("\nExtracting metrics from checkpoints...")
        metrics_data = self.extract_metrics()
        
        # 保存提取的指标为JSON
        metrics_file = output_dir / 'extracted_metrics.json'
        with open(metrics_file, 'w') as f:
            # 移除history以减小文件大小
            metrics_to_save = {}
            for group, experiments in metrics_data.items():
                metrics_to_save[group] = {}
                for exp, data in experiments.items():
                    data_copy = data.copy()
                    data_copy.pop('history', None)
                    data_copy.pop('config', None)
                    metrics_to_save[group][exp] = data_copy
            json.dump(metrics_to_save, f, indent=2)
        print(f"Saved extracted metrics to {metrics_file}")
        
        # 分析各个实验组
        self.analyze_group_1(metrics_data, output_dir)
        self.analyze_group_2(metrics_data, output_dir)
        self.analyze_group_3(metrics_data, output_dir)
        self.analyze_group_4(metrics_data, output_dir)
        self.analyze_group_5(metrics_data, output_dir)
        
        # 生成总体对比
        self.generate_overall_summary(metrics_data, output_dir)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - group1_analysis.png: Training mode and hardware comparison")
        print("  - group2_analysis.png: Model complexity impact")
        print("  - group3_analysis.png: DataLoader optimization")
        print("  - group4_analysis.png: Batch size impact")
        print("  - group5_analysis.png: Pipeline parallel optimization")
        print("  - overall_summary.png: Overall performance summary")
        print("  - summary_statistics.txt: Detailed statistics")
        print("  - extracted_metrics.json: All extracted metrics")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--experiments_dir', type=str, default='./experiments',
                       help='Path to experiments directory')
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = ExperimentAnalyzer(experiments_dir=args.experiments_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
