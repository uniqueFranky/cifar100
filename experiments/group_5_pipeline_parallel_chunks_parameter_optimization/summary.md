# 实验组 5: Pipeline Parallel Chunks Parameter Optimization

## 实验描述
专门针对流水线并行模式，测试不同chunks设置(16/32/64)对训练吞吐量和内存效率的影响

## 控制变量
训练模式(流水线并行)、模型架构(ResNet50)、硬件配置(4卡)、批处理大小(128)、数据加载参数

## 对比重点
流水线并行chunks参数: 16 vs 32 vs 64

## 实验统计
- 总实验数: 3
- 成功: 0
- 失败: 3
- 总耗时: 20.83 小时

## 实验结果详情
| 实验名称 | 状态 | 耗时 | 备注 |
|---------|------|------|------|
| pp_resnet50_bs128_gpu4_nw2_chunks16 | ❌ | 2.97h | 失败 |
| pp_resnet50_bs128_gpu4_nw2_chunks32 | ❌ | 6.29h | 失败 |
| pp_resnet50_bs128_gpu4_nw2_chunks64 | ❌ | 11.57h | 失败 |
