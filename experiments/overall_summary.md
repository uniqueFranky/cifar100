# 分布式训练对比实验总体报告

生成时间: 2025-12-15 09:42:22

## 总体统计
- 实验组数: 1
- 总实验数: 8
- 成功实验: 8
- 成功率: 100.0%

## 各实验组结果汇总
### 实验组 1: Training Mode and Hardware Configuration Comparison
- 实验目标: 全面比较不同训练模式(single/dp/ddp/mp)在不同硬件配置(1卡/2卡/4卡)下的性能表现
- 对比维度: 训练模式(single/dp/ddp/mp/pp) × 硬件配置(1卡/2卡/4卡)的性能矩阵对比
- 实验数量: 8
- 成功数量: 8
- 成功率: 100.0%
- 结果目录: `./experiments/group_1_training_mode_and_hardware_configuration_comparison`

## 实验设计说明
**实验组 1**: 训练模式(single/dp/ddp/mp/pp) × 硬件配置(1卡/2卡/4卡)的性能矩阵对比

## 目录结构说明
```
experiments/
├── overall_summary.md          # 总体分析报告
├── group_1_training_mode_and_hardware_configuration_comparison/
│   ├── experiment_info.json   # 实验组配置信息
│   ├── results.json          # 详细实验结果数据
│   ├── summary.md            # 实验组分析报告
│   ├── checkpoints/          # 训练完成的模型文件
│   └── logs/                 # 详细训练日志
```
