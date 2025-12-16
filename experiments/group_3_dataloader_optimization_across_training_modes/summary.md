# 实验组 3: DataLoader Optimization Across Training Modes

## 实验描述
系统性测试不同数据加载worker数量(0-16)在单机和分布式训练模式下对训练性能的影响

## 控制变量
模型架构(ResNet50)、批处理大小(128)、硬件配置对比(1卡vs4卡)

## 对比重点
数据加载worker数量(0/1/2/4/8/16) × 训练模式(single/ddp) × 硬件配置(1卡/4卡)

## 实验统计
- 总实验数: 12
- 成功: 12
- 失败: 0
- 总耗时: 5.69 小时

## 实验结果详情
| 实验名称 | 状态 | 耗时 | 备注 |
|---------|------|------|------|
| single_resnet50_bs128_gpu1_nw0_chunks0 | ✅ | 1.14h |  |
| single_resnet50_bs128_gpu1_nw1_chunks0 | ✅ | 0.66h |  |
| single_resnet50_bs128_gpu1_nw2_chunks0 | ✅ | 0.60h |  |
| single_resnet50_bs128_gpu1_nw4_chunks0 | ✅ | 0.59h |  |
| single_resnet50_bs128_gpu1_nw8_chunks0 | ✅ | 0.60h |  |
| single_resnet50_bs128_gpu1_nw16_chunks0 | ✅ | 0.60h |  |
| ddp_resnet50_bs128_gpu4_nw0_chunks0 | ✅ | 0.40h |  |
| ddp_resnet50_bs128_gpu4_nw1_chunks0 | ✅ | 0.21h |  |
| ddp_resnet50_bs128_gpu4_nw2_chunks0 | ✅ | 0.21h |  |
| ddp_resnet50_bs128_gpu4_nw4_chunks0 | ✅ | 0.21h |  |
| ddp_resnet50_bs128_gpu4_nw8_chunks0 | ✅ | 0.22h |  |
| ddp_resnet50_bs128_gpu4_nw16_chunks0 | ✅ | 0.24h |  |
