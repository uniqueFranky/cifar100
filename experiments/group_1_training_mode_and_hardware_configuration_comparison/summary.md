# 实验组 1: Training Mode and Hardware Configuration Comparison

## 实验描述
全面比较不同训练模式(single/dp/ddp/mp)在不同硬件配置(1卡/2卡/4卡)下的性能表现

## 控制变量
模型架构(ResNet50)、批处理大小(128)、数据加载参数(workers=2, prefetch=1)

## 对比重点
训练模式(single/dp/ddp/mp/pp) × 硬件配置(1卡/2卡/4卡)的性能矩阵对比

## 实验统计
- 总实验数: 8
- 成功: 8
- 失败: 0
- 总耗时: 0.60 小时

## 实验结果详情
| 实验名称 | 状态 | 耗时 | 备注 |
|---------|------|------|------|
| single_resnet50_bs128_gpu1_nw2_chunks0 | ✅ | 0.60h |  |
| dp_resnet50_bs128_gpu2_nw2_chunks0 | ⏭️ | 跳过 | 已存在 |
| dp_resnet50_bs128_gpu4_nw2_chunks0 | ⏭️ | 跳过 | 已存在 |
| ddp_resnet50_bs128_gpu2_nw2_chunks0 | ⏭️ | 跳过 | 已存在 |
| ddp_resnet50_bs128_gpu4_nw2_chunks0 | ⏭️ | 跳过 | 已存在 |
| mp_resnet50_bs128_gpu2_nw2_chunks0 | ⏭️ | 跳过 | 已存在 |
| mp_resnet50_bs128_gpu4_nw2_chunks0 | ⏭️ | 跳过 | 已存在 |
| hp_resnet50_bs128_gpu4_nw2_chunks0 | ⏭️ | 跳过 | 已存在 |
