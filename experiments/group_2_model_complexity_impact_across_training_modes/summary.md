# 实验组 2: Model Complexity Impact Across Training Modes

## 实验描述
测试不同复杂度模型(ResNet18/34/50)在单机训练和分布式训练模式下的性能差异和扩展性

## 控制变量
批处理大小(128)、数据加载参数(workers=2, prefetch=1)、硬件配置对比(1卡vs4卡)

## 对比重点
模型复杂度(ResNet18/34/50) × 训练模式(single/ddp) × 硬件配置(1卡/4卡)

## 实验统计
- 总实验数: 6
- 成功: 6
- 失败: 0
- 总耗时: 1.73 小时

## 实验结果详情
| 实验名称 | 状态 | 耗时 | 备注 |
|---------|------|------|------|
| single_resnet18_bs128_gpu1_nw2_chunks0 | ✅ | 0.33h |  |
| single_resnet34_bs128_gpu1_nw2_chunks0 | ✅ | 0.34h |  |
| single_resnet50_bs128_gpu1_nw2_chunks0 | ✅ | 0.60h |  |
| ddp_resnet18_bs128_gpu4_nw2_chunks0 | ✅ | 0.10h |  |
| ddp_resnet34_bs128_gpu4_nw2_chunks0 | ✅ | 0.15h |  |
| ddp_resnet50_bs128_gpu4_nw2_chunks0 | ✅ | 0.21h |  |
