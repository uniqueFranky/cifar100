# 实验组 4: Batch Size Impact Across Training Modes and Hardware

## 实验描述
分析不同批处理大小(64/128/256)在单机和分布式训练中的性能表现和内存使用效率

## 控制变量
模型架构(ResNet50)、数据加载参数(workers=2, prefetch=1)、硬件配置对比(1卡vs4卡)

## 对比重点
批处理大小(64/128/256) × 训练模式(single/ddp) × 硬件配置(1卡/4卡)

## 实验统计
- 总实验数: 8
- 成功: 8
- 失败: 0
- 总耗时: 4.37 小时

## 实验结果详情
| 实验名称 | 状态 | 耗时 | 备注 |
|---------|------|------|------|
| single_resnet50_bs32_gpu1_nw2_chunks0 | ✅ | 1.29h |  |
| single_resnet50_bs64_gpu1_nw2_chunks0 | ✅ | 0.71h |  |
| single_resnet50_bs128_gpu1_nw2_chunks0 | ✅ | 0.59h |  |
| single_resnet50_bs256_gpu1_nw2_chunks0 | ✅ | 0.63h |  |
| ddp_resnet50_bs32_gpu4_nw2_chunks0 | ✅ | 0.45h |  |
| ddp_resnet50_bs64_gpu4_nw2_chunks0 | ✅ | 0.27h |  |
| ddp_resnet50_bs128_gpu4_nw2_chunks0 | ✅ | 0.21h |  |
| ddp_resnet50_bs256_gpu4_nw2_chunks0 | ✅ | 0.22h |  |
