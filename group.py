"""
分布式训练对比实验批量脚本
按照控制变量法设计的5个实验组，每组结果保存在独立文件夹中

实验组设计：
1. 训练模式与硬件配置综合对比
2. 模型复杂度在不同训练模式下的性能影响
3. 数据加载器参数在不同训练模式下的优化效果
4. 批处理大小在不同训练模式和硬件配置下的影响
5. 流水线并行chunks参数优化

使用方式:
    python batch_train.py --experiment all          # 运行所有实验组
    python batch_train.py --experiment 1            # 运行实验组1
    python batch_train.py --experiment 1,3,5        # 运行指定的多个实验组
    python batch_train.py --experiment all --epochs 20  # 所有实验但减少epochs
"""

import subprocess
import os
import sys
import time
import json
from datetime import datetime
from itertools import product
from typing import List, Dict, Any, Optional
import argparse


class ExperimentGroup:
    """实验组定义"""
    
    def __init__(self, name: str, description: str, config: Dict[str, List[Any]], 
                 control_variables: str, comparison_focus: str):
        self.name = name
        self.description = description
        self.config = config
        self.control_variables = control_variables
        self.comparison_focus = comparison_focus
    
    def get_safe_name(self) -> str:
        """获取文件系统安全的名称"""
        return self.name.lower().replace(' ', '_').replace('/', '_').replace('vs', 'vs')


class TrainingScheduler:
    """训练任务调度器"""
    
    def __init__(self, base_dir: str = './experiments'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        self.experiment_groups = self._define_experiment_groups()
        self.results = {}
        
    def _define_experiment_groups(self) -> Dict[int, ExperimentGroup]:
        """定义所有实验组"""
        groups = {}
        
        # 实验组1：训练模式与硬件配置综合对比
        groups[1] = ExperimentGroup(
            name="Training Mode and Hardware Configuration Comparison",
            description="全面比较不同训练模式(single/dp/ddp/mp)在不同硬件配置(1卡/2卡/4卡)下的性能表现",
            config={
                'mode': ['single', 'dp', 'ddp', 'mp', 'hp'],
                'model': ['resnet50'],
                'batch_size': [128],
                'epochs': [30],
                'dataset': ['cifar100'],
                'gpu_ids': [[1], [1, 2], [1, 2, 3, 4]],
                'num_workers': [2],
                'prefetch_factor': [1],
                'chunks': [0, 32]
            },
            control_variables="模型架构(ResNet50)、批处理大小(128)、数据加载参数(workers=2, prefetch=1)",
            comparison_focus="训练模式(single/dp/ddp/mp/pp) × 硬件配置(1卡/2卡/4卡)的性能矩阵对比"
        )

        # 实验组2：模型复杂度在不同训练模式下的性能影响
        groups[2] = ExperimentGroup(
            name="Model Complexity Impact Across Training Modes",
            description="测试不同复杂度模型(ResNet18/34/50)在单机训练和分布式训练模式下的性能差异和扩展性",
            config={
                'mode': ['single', 'ddp'],
                'model': ['resnet18', 'resnet34', 'resnet50'],
                'batch_size': [128],
                'epochs': [30],
                'dataset': ['cifar100'],
                'gpu_ids': [[1], [1, 2, 3, 5]],
                'num_workers': [2],
                'prefetch_factor': [1],
                'chunks': [0]
            },
            control_variables="批处理大小(128)、数据加载参数(workers=2, prefetch=1)、硬件配置对比(1卡vs4卡)",
            comparison_focus="模型复杂度(ResNet18/34/50) × 训练模式(single/ddp) × 硬件配置(1卡/4卡)"
        )
        
        # 实验组3：数据加载器参数在不同训练模式下的优化效果
        groups[3] = ExperimentGroup(
            name="DataLoader Optimization Across Training Modes",
            description="系统性测试不同数据加载worker数量(0-16)在单机和分布式训练模式下对训练性能的影响",
            config={
                'mode': ['single', 'ddp'],
                'model': ['resnet50'],
                'batch_size': [128],
                'epochs': [30],
                'dataset': ['cifar100'],
                'gpu_ids': [[1], [1, 2, 3, 5]],
                'num_workers': [0, 1, 2, 4, 8, 16],
                'prefetch_factor': [1],
                'chunks': [0]
            },
            control_variables="模型架构(ResNet50)、批处理大小(128)、硬件配置对比(1卡vs4卡)",
            comparison_focus="数据加载worker数量(0/1/2/4/8/16) × 训练模式(single/ddp) × 硬件配置(1卡/4卡)"
        )
        
        # 实验组4：批处理大小在不同训练模式和硬件配置下的影响
        groups[4] = ExperimentGroup(
            name="Batch Size Impact Across Training Modes and Hardware",
            description="分析不同批处理大小(64/128/256)在单机和分布式训练中的性能表现和内存使用效率",
            config={
                'mode': ['single', 'ddp'],
                'model': ['resnet50'],
                'batch_size': [32, 64, 128, 256],
                'epochs': [30],
                'dataset': ['cifar100'],
                'gpu_ids': [[1], [1, 2, 3, 5]],
                'num_workers': [2],
                'prefetch_factor': [1],
                'chunks': [0]
            },
            control_variables="模型架构(ResNet50)、数据加载参数(workers=2, prefetch=1)、硬件配置对比(1卡vs4卡)",
            comparison_focus="批处理大小(64/128/256) × 训练模式(single/ddp) × 硬件配置(1卡/4卡)"
        )
        
        # 实验组5：流水线并行chunks参数优化
        groups[5] = ExperimentGroup(
            name="Pipeline Parallel Chunks Parameter Optimization",
            description="专门针对流水线并行模式，测试不同chunks设置(16/32/64)对训练吞吐量和内存效率的影响",
            config={
                'mode': ['pp'],
                'model': ['resnet50'],
                'batch_size': [128],
                'epochs': [30],
                'dataset': ['cifar100'],
                'gpu_ids': [[1, 2, 3, 5]],
                'num_workers': [2],
                'prefetch_factor': [1],
                'chunks': [16, 32, 64]
            },
            control_variables="训练模式(流水线并行)、模型架构(ResNet50)、硬件配置(4卡)、批处理大小(128)、数据加载参数",
            comparison_focus="流水线并行chunks参数: 16 vs 32 vs 64"
        )
        
        return groups
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """验证参数组合是否有效"""
        mode = params['mode']
        num_gpus = len(params['gpu_ids'])
        
        # single模式只能使用1个GPU
        if mode == 'single' and num_gpus != 1:
            return False
        
        # dp, ddp, mp, pp模式需要至少2个GPU
        if mode in ['dp', 'ddp', 'mp', 'pp'] and num_gpus < 2:
            return False
        
        if mode == 'hp' and num_gpus !=4:
            return False
        
        # 非pp模式chunks必须为0
        if mode != 'pp' and params['chunks'] != 0:
            return False
        
        # pp模式chunks必须大于等于GPU数量
        if mode == 'pp' and params['chunks'] < num_gpus:
            return False
        
        return True
    
    def generate_experiment_name(self, params: Dict[str, Any]) -> str:
        """生成实验名称，包含所有关键参数以确保唯一性"""
        name_parts = [
            params['mode'],
            params['model'],
            f"bs{params['batch_size']}",
            f"gpu{len(params['gpu_ids'])}"
        ]
        
        # 始终包含worker数量，因为这是重要的对比维度
        name_parts.append(f"nw{params['num_workers']}")

        # 始终包含chunks参数，用于区分不同实验
        name_parts.append(f"chunks{params['chunks']}")
        
        return '_'.join(name_parts)
    
    def build_command(self, params: Dict[str, Any], checkpoint_path: str) -> List[str]:
        """构建训练命令"""

        cmd = ['python', '-u', 'main.py']
        
        # 添加所有训练参数
        for key, value in params.items():
            if key == 'gpu_ids':
                cmd.extend(['--gpu-ids', ','.join(map(str, value))])
            elif key == 'num_gpus':
                continue  # 这是计算得出的参数，不需要传递
            elif isinstance(value, bool):
                if value:
                    cmd.append(f'--{key.replace("_", "-")}')
            elif value is not None:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])
        
        cmd.extend(['--final-checkpoint-path', checkpoint_path])
        return cmd
    
    def run_single_experiment(self, params: Dict[str, Any], exp_name: str, 
                            log_dir: str, checkpoint_dir: str) -> Dict[str, Any]:
        """运行单个实验"""
        print(f"\n🚀 Running: {exp_name}")
        
        # 生成文件路径
        checkpoint_path = os.path.join(checkpoint_dir, f"{exp_name}.pth")
        log_file = os.path.join(log_dir, f"{exp_name}.log")
        
        # 检查实验是否已完成，避免重复运行
        if os.path.exists(checkpoint_path):
            print(f"⚠️  Skipping existing experiment: {exp_name}")
            return {
                'experiment_name': exp_name,
                'params': params,
                'success': True,
                'skipped': True,
                'checkpoint_path': checkpoint_path,
                'log_file': log_file
            }
        
        # 构建训练命令
        cmd = self.build_command(params, checkpoint_path)
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w', buffering=1) as f:
                # 写入实验元信息到日志文件
                f.write("="*100 + "\n")
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Parameters: {json.dumps(params, indent=2)}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Start Time: {datetime.now().isoformat()}\n")
                f.write("="*100 + "\n\n")
                f.flush()
                
                # 启动训练进程
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    preexec_fn=None if sys.platform == 'win32' else os.setpgrp
                )
                
                # 实时输出训练日志并保存到文件
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                process.wait()
                return_code = process.returncode
            
            elapsed_time = time.time() - start_time
            success = return_code == 0 and os.path.exists(checkpoint_path)
            
            result = {
                'experiment_name': exp_name,
                'params': params,
                'success': success,
                'return_code': return_code,
                'elapsed_time': elapsed_time,
                'elapsed_time_str': f"{elapsed_time/3600:.2f}h",
                'checkpoint_path': checkpoint_path,
                'log_file': log_file,
                'timestamp': datetime.now().isoformat()
            }
            
            status = "✅ Success" if success else "❌ Failed"
            print(f"{status} - Time: {elapsed_time/3600:.2f}h")
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Error: {str(e)}")
            
            return {
                'experiment_name': exp_name,
                'params': params,
                'success': False,
                'error': str(e),
                'elapsed_time': elapsed_time,
                'checkpoint_path': checkpoint_path,
                'log_file': log_file,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_experiment_group(self, group_id: int, custom_epochs: Optional[int] = None) -> Dict[str, Any]:
        """运行单个实验组的所有实验"""
        if group_id not in self.experiment_groups:
            raise ValueError(f"Experiment group {group_id} not found")
        
        group = self.experiment_groups[group_id]
        
        print("\n" + "="*100)
        print(f"🧪 EXPERIMENT GROUP {group_id}: {group.name}")
        print("="*100)
        print(f"📝 Description: {group.description}")
        print(f"🔧 Control Variables: {group.control_variables}")
        print(f"🎯 Comparison Focus: {group.comparison_focus}")
        print("="*100)
        
        # 创建实验组专用目录结构
        group_dir = os.path.join(self.base_dir, f"group_{group_id}_{group.get_safe_name()}")
        checkpoint_dir = os.path.join(group_dir, 'checkpoints')
        log_dir = os.path.join(group_dir, 'logs')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 保存实验组配置信息
        info_file = os.path.join(group_dir, 'experiment_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump({
                'group_id': group_id,
                'name': group.name,
                'description': group.description,
                'control_variables': group.control_variables,
                'comparison_focus': group.comparison_focus,
                'config': group.config,
                'created_at': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        # 应用自定义epochs设置（如果提供）
        config = group.config.copy()
        if custom_epochs is not None:
            config['epochs'] = [custom_epochs]
            print(f"🔄 Using custom epochs: {custom_epochs}")
        
        # 生成所有可能的参数组合
        keys = list(config.keys())
        values = list(config.values())
        all_combinations = list(product(*values))
        
        # 过滤出有效的参数组合
        valid_combinations = []
        for combination in all_combinations:
            params = dict(zip(keys, combination))
            params['num_gpus'] = len(params['gpu_ids'])
            
            if self.validate_params(params):
                valid_combinations.append(params)
        
        print(f"📊 Total experiments in this group: {len(valid_combinations)}")
        
        # 依次运行所有有效实验
        group_results = []
        for i, params in enumerate(valid_combinations, 1):
            exp_name = self.generate_experiment_name(params)
            print(f"\n[{i}/{len(valid_combinations)}] ", end="")
            
            result = self.run_single_experiment(params, exp_name, log_dir, checkpoint_dir)
            group_results.append(result)
            
            # 实时保存中间结果
            results_file = os.path.join(group_dir, 'results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'group_info': {
                        'group_id': group_id,
                        'name': group.name,
                        'description': group.description,
                        'control_variables': group.control_variables,
                        'comparison_focus': group.comparison_focus
                    },
                    'total_experiments': len(valid_combinations),
                    'completed_experiments': i,
                    'results': group_results
                }, f, indent=2, ensure_ascii=False)
            
            # 分布式训练后等待GPU资源完全释放
            if params['mode'] in ['dp', 'ddp', 'mp', 'pp']:
                time.sleep(5)
        
        # 生成实验组详细总结报告
        self.generate_group_summary(group_id, group, group_results, group_dir)
        
        return {
            'group_id': group_id,
            'group_name': group.name,
            'total_experiments': len(valid_combinations),
            'successful_experiments': sum(1 for r in group_results if r['success']),
            'results': group_results,
            'group_dir': group_dir
        }
    
    def generate_group_summary(self, group_id: int, group: ExperimentGroup, 
                             results: List[Dict], group_dir: str):
        """生成实验组详细总结报告"""
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_time = sum(r.get('elapsed_time', 0) for r in results if not r.get('skipped', False))
        
        summary_file = os.path.join(group_dir, 'summary.md')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# 实验组 {group_id}: {group.name}\n\n")
            f.write(f"## 实验描述\n{group.description}\n\n")
            f.write(f"## 控制变量\n{group.control_variables}\n\n")
            f.write(f"## 对比重点\n{group.comparison_focus}\n\n")
            
            f.write("## 实验统计\n")
            f.write(f"- 总实验数: {len(results)}\n")
            f.write(f"- 成功: {successful}\n")
            f.write(f"- 失败: {failed}\n")
            f.write(f"- 总耗时: {total_time/3600:.2f} 小时\n\n")
            
            f.write("## 实验结果详情\n")
            f.write("| 实验名称 | 状态 | 耗时 | 备注 |\n")
            f.write("|---------|------|------|------|\n")
            
            for result in results:
                status = "✅" if result['success'] else "❌"
                if result.get('skipped'):
                    status = "⏭️"
                    time_str = "跳过"
                else:
                    time_str = result.get('elapsed_time_str', 'N/A')
                
                note = ""
                if result.get('skipped'):
                    note = "已存在"
                elif not result['success']:
                    note = result.get('error', '失败')
                
                f.write(f"| {result['experiment_name']} | {status} | {time_str} | {note} |\n")
        
        print(f"\n📋 Summary saved to: {summary_file}")
    
    def run_experiments(self, experiment_ids: List[int], custom_epochs: Optional[int] = None):
        """运行指定的实验组"""
        print("\n" + "🎯" * 50)
        print("分布式训练对比实验开始")
        print("🎯" * 50)
        
        if custom_epochs:
            print(f"🔄 使用自定义epochs: {custom_epochs}")
        
        all_results = {}
        
        for group_id in experiment_ids:
            if group_id not in self.experiment_groups:
                print(f"⚠️  Warning: Experiment group {group_id} not found, skipping...")
                continue
            
            try:
                result = self.run_experiment_group(group_id, custom_epochs)
                all_results[group_id] = result
                
                print(f"\n✅ Group {group_id} completed: {result['successful_experiments']}/{result['total_experiments']} successful")
                
            except Exception as e:
                print(f"\n❌ Error in group {group_id}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 生成跨实验组的总体分析报告
        self.generate_overall_summary(all_results)
        
        print("\n" + "🎉" * 50)
        print("所有实验完成！")
        print("🎉" * 50)
    
    def generate_overall_summary(self, all_results: Dict[int, Dict]):
        """生成跨实验组的总体分析报告"""
        summary_file = os.path.join(self.base_dir, 'overall_summary.md')
        
        total_experiments = sum(r['total_experiments'] for r in all_results.values())
        total_successful = sum(r['successful_experiments'] for r in all_results.values())
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# 分布式训练对比实验总体报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 总体统计\n")
            f.write(f"- 实验组数: {len(all_results)}\n")
            f.write(f"- 总实验数: {total_experiments}\n")
            f.write(f"- 成功实验: {total_successful}\n")
            f.write(f"- 成功率: {total_successful/total_experiments*100:.1f}%\n\n")
            
            f.write("## 各实验组结果汇总\n")
            for group_id, result in all_results.items():
                group = self.experiment_groups[group_id]
                success_rate = result['successful_experiments'] / result['total_experiments'] * 100
                
                f.write(f"### 实验组 {group_id}: {group.name}\n")
                f.write(f"- 实验目标: {group.description}\n")
                f.write(f"- 对比维度: {group.comparison_focus}\n")
                f.write(f"- 实验数量: {result['total_experiments']}\n")
                f.write(f"- 成功数量: {result['successful_experiments']}\n")
                f.write(f"- 成功率: {success_rate:.1f}%\n")
                f.write(f"- 结果目录: `{result['group_dir']}`\n\n")
            
            f.write("## 实验设计说明\n")
            for group_id, group in self.experiment_groups.items():
                if group_id in all_results:
                    f.write(f"**实验组 {group_id}**: {group.comparison_focus}\n")
            
            f.write(f"\n## 目录结构说明\n")
            f.write("```\n")
            f.write("experiments/\n")
            f.write("├── overall_summary.md          # 总体分析报告\n")
            for group_id in all_results.keys():
                group = self.experiment_groups[group_id]
                f.write(f"├── group_{group_id}_{group.get_safe_name()}/\n")
                f.write(f"│   ├── experiment_info.json   # 实验组配置信息\n")
                f.write(f"│   ├── results.json          # 详细实验结果数据\n")
                f.write(f"│   ├── summary.md            # 实验组分析报告\n")
                f.write(f"│   ├── checkpoints/          # 训练完成的模型文件\n")
                f.write(f"│   └── logs/                 # 详细训练日志\n")
            f.write("```\n")
        
        print(f"\n📊 Overall summary saved to: {summary_file}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='分布式训练对比实验')
    
    parser.add_argument('--experiment', type=str, default='all',
                       help='实验组ID，支持: all, 1, 2, 3, 4, 5 或组合如 1,3,5')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='自定义epochs数量，覆盖默认设置')
    
    parser.add_argument('--base-dir', type=str, default='./experiments',
                       help='实验基础目录')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 解析实验组ID参数
    if args.experiment.lower() == 'all':
        experiment_ids = list(range(1, 6))  # 1-5 (现在只有5个实验组)
    else:
        try:
            experiment_ids = [int(x.strip()) for x in args.experiment.split(',')]
            # 验证实验组ID的有效性
            valid_ids = list(range(1, 6))
            for exp_id in experiment_ids:
                if exp_id not in valid_ids:
                    print(f"❌ Invalid experiment ID: {exp_id}. Valid IDs: {valid_ids}")
                    return
        except ValueError:
            print(f"❌ Invalid experiment format: {args.experiment}")
            print("Use: all, 1, 2, 3, 4, 5 or combinations like 1,3,5")
            return
    
    print("🧪 分布式训练对比实验")
    print(f"📁 实验目录: {args.base_dir}")
    print(f"🎯 运行实验组: {experiment_ids}")
    if args.epochs:
        print(f"🔄 自定义epochs: {args.epochs}")
    
    # 显示将要运行的实验组信息
    scheduler = TrainingScheduler(base_dir=args.base_dir)
    print("\n📋 实验组列表:")
    for exp_id in experiment_ids:
        group = scheduler.experiment_groups[exp_id]
        print(f"  {exp_id}. {group.name}")
        print(f"     {group.comparison_focus}")
    
    # 确认开始实验
    print(f"\n总计将运行 {len(experiment_ids)} 个实验组")
    
    # 开始执行所有实验
    scheduler.run_experiments(experiment_ids, args.epochs)


if __name__ == '__main__':
    main()
