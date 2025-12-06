"""
训练性能分析脚本
分析checkpoint中记录的训练过程，生成性能报告和可视化图表
"""

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import seaborn as sns

# 设置matplotlib使用英文，避免中文字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def convert_to_serializable(obj: Any) -> Any:
    """
    将numpy类型转换为Python原生类型，以便JSON序列化
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, checkpoint_path: str, output_dir: str = './analysis_results'):
        """
        初始化分析器
        
        Args:
            checkpoint_path: checkpoint文件路径
            output_dir: 分析结果输出目录
        """
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取数据
        self.config = self.checkpoint.get('config', {})
        self.history = self.checkpoint.get('history', {})
        self.best_acc = self.checkpoint.get('best_acc', 0)
        self.final_epoch = self.checkpoint.get('epoch', 0)
        
        print(f"Successfully loaded {self.final_epoch + 1} epochs of training data")
    
    def compute_statistics(self) -> Dict:
        """计算统计指标"""
        stats = {}
        
        # 基本信息
        stats['training_info'] = {
            'total_epochs': len(self.history.get('train_loss', [])),
            'final_epoch': self.final_epoch + 1,
            'best_accuracy': float(self.best_acc),
            'mode': self.config.get('mode', 'unknown'),
            'num_gpus': self.config.get('num_gpus', 1),
            'batch_size': self.config.get('batch_size', 'unknown'),
            'effective_batch_size': self.config.get('effective_batch_size', 'unknown'),
        }
        
        # 时间统计
        epoch_times = self.history.get('epoch_time', [])
        if epoch_times:
            stats['time_stats'] = {
                'total_time_seconds': float(sum(epoch_times)),
                'total_time_minutes': float(sum(epoch_times) / 60),
                'total_time_hours': float(sum(epoch_times) / 3600),
                'avg_epoch_time': float(np.mean(epoch_times)),
                'std_epoch_time': float(np.std(epoch_times)),
                'min_epoch_time': float(np.min(epoch_times)),
                'max_epoch_time': float(np.max(epoch_times)),
                'median_epoch_time': float(np.median(epoch_times)),
            }
        
        # 准确率统计
        train_acc = self.history.get('train_acc', [])
        test_acc = self.history.get('test_acc', [])
        
        if train_acc and test_acc:
            stats['accuracy_stats'] = {
                'final_train_acc': float(train_acc[-1]),
                'final_test_acc': float(test_acc[-1]),
                'best_test_acc': float(max(test_acc)),
                'best_test_epoch': int(np.argmax(test_acc) + 1),
                'avg_train_acc': float(np.mean(train_acc)),
                'avg_test_acc': float(np.mean(test_acc)),
                'train_test_gap': float(train_acc[-1] - test_acc[-1]),  # 过拟合指标
            }
        
        # 损失统计
        train_loss = self.history.get('train_loss', [])
        test_loss = self.history.get('test_loss', [])
        
        if train_loss and test_loss:
            stats['loss_stats'] = {
                'final_train_loss': float(train_loss[-1]),
                'final_test_loss': float(test_loss[-1]),
                'min_train_loss': float(min(train_loss)),
                'min_test_loss': float(min(test_loss)),
                'avg_train_loss': float(np.mean(train_loss)),
                'avg_test_loss': float(np.mean(test_loss)),
            }
        
        # 收敛分析
        if test_acc:
            # 找到达到95%最佳准确率的epoch
            target_acc = 0.95 * max(test_acc)
            converge_epoch = next((i for i, acc in enumerate(test_acc) if acc >= target_acc), None)
            
            stats['convergence_stats'] = {
                'converge_epoch': int(converge_epoch + 1) if converge_epoch is not None else 'Not converged',
                'converge_time': float(sum(epoch_times[:converge_epoch+1])) if converge_epoch is not None else None,
                'improvement_last_10_epochs': float(test_acc[-1] - test_acc[-11]) if len(test_acc) > 10 else None,
            }
        
        # 吞吐量分析
        if epoch_times and self.config.get('batch_size'):
            # 假设数据集大小（CIFAR-10/100）
            dataset_size = 50000  # CIFAR训练集大小
            batches_per_epoch = dataset_size / self.config.get('effective_batch_size', self.config.get('batch_size', 128))
            
            stats['throughput_stats'] = {
                'avg_samples_per_second': float(dataset_size / np.mean(epoch_times)),
                'avg_batches_per_second': float(batches_per_epoch / np.mean(epoch_times)),
                'avg_time_per_batch': float(np.mean(epoch_times) / batches_per_epoch),
            }
        
        return stats
    
    def print_report(self, stats: Dict):
        """打印文本报告"""
        print("\n" + "="*80)
        print(" "*25 + "Performance Analysis Report")
        print("="*80)
        
        # 训练信息
        print("\n[Training Configuration]")
        info = stats['training_info']
        print(f"  Training Mode: {info['mode'].upper()}")
        print(f"  Number of GPUs: {info['num_gpus']}")
        print(f"  Batch Size: {info['batch_size']} (Effective: {info['effective_batch_size']})")
        print(f"  Total Epochs: {info['total_epochs']}")
        print(f"  Model: {self.config.get('model', 'unknown')}")
        print(f"  Dataset: {self.config.get('dataset', 'unknown')}")
        
        # 时间统计
        if 'time_stats' in stats:
            print("\n[Time Cost]")
            ts = stats['time_stats']
            print(f"  Total Training Time: {ts['total_time_hours']:.2f} hours ({ts['total_time_minutes']:.2f} minutes)")
            print(f"  Average per Epoch: {ts['avg_epoch_time']:.2f} ± {ts['std_epoch_time']:.2f} seconds")
            print(f"  Fastest Epoch: {ts['min_epoch_time']:.2f} seconds")
            print(f"  Slowest Epoch: {ts['max_epoch_time']:.2f} seconds")
            print(f"  Median: {ts['median_epoch_time']:.2f} seconds")
        
        # 吞吐量
        if 'throughput_stats' in stats:
            print("\n[Training Throughput]")
            tp = stats['throughput_stats']
            print(f"  Sample Processing Speed: {tp['avg_samples_per_second']:.2f} samples/s")
            print(f"  Batch Processing Speed: {tp['avg_batches_per_second']:.2f} batches/s")
            print(f"  Average Batch Time: {tp['avg_time_per_batch']*1000:.2f} ms/batch")
        
        # 准确率
        if 'accuracy_stats' in stats:
            print("\n[Accuracy Performance]")
            acc = stats['accuracy_stats']
            print(f"  Best Test Accuracy: {acc['best_test_acc']:.2f}% (Epoch {acc['best_test_epoch']})")
            print(f"  Final Train Accuracy: {acc['final_train_acc']:.2f}%")
            print(f"  Final Test Accuracy: {acc['final_test_acc']:.2f}%")
            print(f"  Train-Test Gap: {acc['train_test_gap']:.2f}% (Overfitting Indicator)")
            print(f"  Average Train Accuracy: {acc['avg_train_acc']:.2f}%")
            print(f"  Average Test Accuracy: {acc['avg_test_acc']:.2f}%")
        
        # 损失
        if 'loss_stats' in stats:
            print("\n[Loss Function]")
            loss = stats['loss_stats']
            print(f"  Final Train Loss: {loss['final_train_loss']:.4f}")
            print(f"  Final Test Loss: {loss['final_test_loss']:.4f}")
            print(f"  Min Train Loss: {loss['min_train_loss']:.4f}")
            print(f"  Min Test Loss: {loss['min_test_loss']:.4f}")
        
        # 收敛分析
        if 'convergence_stats' in stats:
            print("\n[Convergence Analysis]")
            conv = stats['convergence_stats']
            print(f"  Convergence Epoch (95% best): {conv['converge_epoch']}")
            if conv['converge_time']:
                print(f"  Convergence Time: {conv['converge_time']/60:.2f} minutes")
            if conv['improvement_last_10_epochs']:
                print(f"  Improvement in Last 10 Epochs: {conv['improvement_last_10_epochs']:.2f}%")
        
        print("\n" + "="*80)
    
    def save_report(self, stats: Dict):
        """保存JSON格式报告"""
        report_path = os.path.join(self.output_dir, 'performance_report.json')
        
        # 转换config为可序列化格式
        serializable_config = convert_to_serializable(self.config)
        serializable_stats = convert_to_serializable(stats)
        
        # 添加配置信息
        full_report = {
            'config': serializable_config,
            'statistics': serializable_stats,
            'checkpoint_path': self.checkpoint_path,
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        print(f"\nReport saved to: {report_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Process Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs, self.history['test_loss'], label='Test Loss', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
        ax2.plot(epochs, self.history['test_acc'], label='Test Accuracy', marker='s', markersize=3)
        ax2.axhline(y=self.best_acc, color='r', linestyle='--', label=f'Best: {self.best_acc:.2f}%')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 每Epoch时间
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['epoch_time'], marker='o', markersize=3, color='green')
        ax3.axhline(y=np.mean(self.history['epoch_time']), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(self.history["epoch_time"]):.2f}s')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Training Time per Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 过拟合分析（训练-测试准确率差距）
        ax4 = axes[1, 1]
        gap = np.array(self.history['train_acc']) - np.array(self.history['test_acc'])
        ax4.plot(epochs, gap, marker='o', markersize=3, color='orange')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Gap (%)')
        ax4.set_title('Overfitting Analysis (Train Acc - Test Acc)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
        
        plt.close()
    
    def plot_performance_metrics(self):
        """绘制性能指标"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Performance Metrics Analysis', fontsize=16, fontweight='bold')
        
        # 1. 时间分布直方图
        ax1 = axes[0]
        ax1.hist(self.history['epoch_time'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(self.history['epoch_time']), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(self.history["epoch_time"]):.2f}s')
        ax1.axvline(np.median(self.history['epoch_time']), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(self.history["epoch_time"]):.2f}s')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Epoch Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率提升速度
        ax2 = axes[1]
        test_acc = np.array(self.history['test_acc'])
        acc_improvement = np.diff(test_acc)
        epochs_for_diff = range(2, len(test_acc) + 1)
        
        colors = ['green' if x > 0 else 'red' for x in acc_improvement]
        ax2.bar(epochs_for_diff, acc_improvement, color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy Change (%)')
        ax2.set_title('Test Accuracy Change Rate')
        ax2.grid(True, alpha=0.3)
        
        # 3. 学习效率（准确率/时间）
        ax3 = axes[2]
        efficiency = np.array(self.history['test_acc']) / np.array(self.history['epoch_time'])
        epochs = range(1, len(efficiency) + 1)
        ax3.plot(epochs, efficiency, marker='o', markersize=3, color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy / Time (%/s)')
        ax3.set_title('Learning Efficiency (Accuracy/Time)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.output_dir, 'performance_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics saved to: {plot_path}")
        
        plt.close()
    
    def plot_comparison_table(self, stats: Dict):
        """生成性能对比表格图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        table_data = []
        
        if 'training_info' in stats:
            info = stats['training_info']
            table_data.append(['Training Mode', info['mode'].upper()])
            table_data.append(['Number of GPUs', str(info['num_gpus'])])
            table_data.append(['Batch Size', f"{info['batch_size']} (Effective: {info['effective_batch_size']})"])
        
        if 'time_stats' in stats:
            ts = stats['time_stats']
            table_data.append(['Total Training Time', f"{ts['total_time_hours']:.2f} hours"])
            table_data.append(['Average Epoch Time', f"{ts['avg_epoch_time']:.2f} ± {ts['std_epoch_time']:.2f} seconds"])
        
        if 'throughput_stats' in stats:
            tp = stats['throughput_stats']
            table_data.append(['Sample Processing Speed', f"{tp['avg_samples_per_second']:.2f} samples/s"])
        
        if 'accuracy_stats' in stats:
            acc = stats['accuracy_stats']
            table_data.append(['Best Test Accuracy', f"{acc['best_test_acc']:.2f}%"])
            table_data.append(['Final Test Accuracy', f"{acc['final_test_acc']:.2f}%"])
        
        # 创建表格
        table = ax.table(cellText=table_data, 
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.5, 0.5])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置行颜色
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#f0f0f0')
        
        plt.title('Key Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 保存图片
        plot_path = os.path.join(self.output_dir, 'performance_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance summary saved to: {plot_path}")
        
        plt.close()
    
    def analyze(self):
        """执行完整分析"""
        print("\n" + "="*80)
        print(" "*20 + "Starting Performance Analysis")
        print("="*80)
        
        # 计算统计指标
        stats = self.compute_statistics()
        
        # 打印报告
        self.print_report(stats)
        
        # 保存JSON报告
        self.save_report(stats)
        
        # 生成可视化
        print("\nGenerating visualizations...")
        self.plot_training_curves()
        self.plot_performance_metrics()
        self.plot_comparison_table(stats)
        
        print(f"\nAll analysis results saved to: {self.output_dir}")
        print("="*80)
        
        return stats


def compare_checkpoints(checkpoint_paths: List[str], labels: List[str], output_dir: str = './comparison_results'):
    """
    比较多个checkpoint的性能
    
    Args:
        checkpoint_paths: checkpoint文件路径列表
        labels: 每个checkpoint的标签
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(" "*20 + "Multi-Model Performance Comparison")
    print("="*80)
    
    # 加载所有checkpoint
    checkpoints = []
    for path, label in zip(checkpoint_paths, labels):
        print(f"\nLoading {label}: {path}")
        ckpt = torch.load(path, map_location='cpu')
        checkpoints.append((label, ckpt))
    
    # 对比图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. 测试准确率对比
    ax1 = axes[0, 0]
    for label, ckpt in checkpoints:
        history = ckpt.get('history', {})
        test_acc = history.get('test_acc', [])
        epochs = range(1, len(test_acc) + 1)
        ax1.plot(epochs, test_acc, label=label, marker='o', markersize=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练时间对比
    ax2 = axes[0, 1]
    for label, ckpt in checkpoints:
        history = ckpt.get('history', {})
        epoch_time = history.get('epoch_time', [])
        epochs = range(1, len(epoch_time) + 1)
        ax2.plot(epochs, epoch_time, label=label, marker='o', markersize=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 平均时间柱状图
    ax3 = axes[1, 0]
    avg_times = []
    for label, ckpt in checkpoints:
        history = ckpt.get('history', {})
        epoch_time = history.get('epoch_time', [])
        avg_times.append(np.mean(epoch_time))
    ax3.bar(labels, avg_times, color='skyblue', edgecolor='black')
    ax3.set_ylabel('Average Time (seconds)')
    ax3.set_title('Average Epoch Time Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 最佳准确率柱状图
    ax4 = axes[1, 1]
    best_accs = []
    for label, ckpt in checkpoints:
        best_acc = ckpt.get('best_acc', 0)
        best_accs.append(best_acc)
    ax4.bar(labels, best_accs, color='lightgreen', edgecolor='black')
    ax4.set_ylabel('Best Accuracy (%)')
    ax4.set_title('Best Accuracy Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存对比图
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    plt.close()
    
    # 打印对比表格
    print("\n" + "="*80)
    print(" "*25 + "Performance Comparison Summary")
    print("="*80)
    print(f"{'Model':<20} {'Best Accuracy':<15} {'Avg Time':<15} {'Total Time':<15}")
    print("-"*80)
    
    for label, ckpt in checkpoints:
        history = ckpt.get('history', {})
        best_acc = ckpt.get('best_acc', 0)
        epoch_time = history.get('epoch_time', [])
        avg_time = np.mean(epoch_time)
        total_time = sum(epoch_time) / 3600  # 转换为小时
        
        print(f"{label:<20} {best_acc:<15.2f} {avg_time:<15.2f} {total_time:<15.2f}")
    
    print("="*80)


# 使用示例
if __name__ == '__main__':
    # 单个checkpoint分析
    checkpoint_path = './checkpoints/1.pth'  # 修改为你的checkpoint路径
    
    if os.path.exists(checkpoint_path):
        analyzer = PerformanceAnalyzer(
            checkpoint_path=checkpoint_path,
            output_dir='./analysis_results'
        )
        stats = analyzer.analyze()
    else:
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print("\nPlease modify the checkpoint_path variable to your actual path")
    
    # 多模型对比示例（可选）
    # compare_checkpoints(
    #     checkpoint_paths=[
    #         './checkpoints/single_gpu/final_model.pth',
    #         './checkpoints/dp/final_model.pth',
    #         './checkpoints/ddp/final_model.pth',
    #     ],
    #     labels=['Single GPU', 'DataParallel', 'DistributedDataParallel'],
    #     output_dir='./comparison_results'
    # )