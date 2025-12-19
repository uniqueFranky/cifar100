import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import time
import os

from torch.distributed.pipelining import (
    pipeline,
    SplitPoint,
    ScheduleGPipe,
    PipelineStage,
)

from models import get_model
from utils import (
    get_dataloader,
    PerformanceMonitor,
    setup_distributed,
    cleanup_distributed,
    is_main_process
)


class LossAccumulator:
    """
    Loss 累积器，用于记录所有 micro-batch 的 loss。
    解决 ScheduleGPipe 不直接返回 loss 的问题。
    """
    def __init__(self, criterion):
        self.criterion = criterion
        self.reset()

    def __call__(self, outputs, targets):
        """
        每个 micro-batch 调用一次。
        outputs: [micro_batch_size, num_classes]
        targets: [micro_batch_size]
        """
        loss = self.criterion(outputs, targets)
        batch_size = targets.size(0)

        # 记录加权 loss（按样本数），避免不等长 micro-batch 时 bias
        self.total_loss += loss.detach().item() * batch_size
        self.total_samples += batch_size
        
        # 同时记录准确率
        _, predicted = outputs.max(1)
        self.total_correct += predicted.eq(targets).sum().item()

        return loss

    def get_average_loss(self):
        """返回当前累计的平均 loss，并清空缓存。"""
        if self.total_samples == 0:
            return 0.0

        avg = self.total_loss / self.total_samples
        return avg
    
    def get_accuracy(self):
        """返回当前累计的准确率"""
        if self.total_samples == 0:
            return 0.0
        return 100. * self.total_correct / self.total_samples

    def reset(self):
        """重置累积器。"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_correct = 0


class PipelineParallelTrainer:
    """
    流水线并行训练器 - 与 DDP 完全相同的历史记录格式
    
    历史记录格式（与 DDP 一致）：
    {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': [],
        'learning_rate': [],
        'gpu_memory_per_device': []
    }
    """
    def __init__(self, config):
        self.config = config
        self.world_size = config.num_gpus

    def launch(self):
        """启动多进程训练"""

        # 设置微批次
        self.chunks = getattr(self.config, 'chunks', self.world_size * 4)

        # 确保 Batch Size 能被 Chunks 整除（保证 micro-batch 等长）
        if self.config.batch_size % self.chunks != 0:
            new_bs = ((self.config.batch_size + self.chunks - 1) // self.chunks) * self.chunks
            print(f"⚠️  警告: Batch size {self.config.batch_size} 不能被 chunks {self.chunks} 整除")
            print(f"✅ 自动调整 Batch size 为: {new_bs}")
            self.config.batch_size = new_bs

        print(f"\n{'='*60}")
        print(f"🚀 启动 Pipeline Parallelism 训练 (CIFAR-100)")
        print(f"{'='*60}")
        print(f"📊 配置信息:")
        print(f"  - 数据集: CIFAR-100 (100 类)")
        print(f"  - GPU 数量: {self.world_size}")
        print(f"  - GPU IDs: {self.config.gpu_ids}")
        print(f"  - Global Batch Size: {self.config.batch_size}")
        print(f"  - Micro-batches (Chunks): {self.chunks}")
        print(f"  - Micro-batch Size: {self.config.batch_size // self.chunks}")
        print(f"  - 模型: {self.config.model}")
        print(f"  - Epochs: {self.config.epochs}")
        print(f"  - 随机种子: {self.config.seed}")
        print(f"{'='*60}\n")
        print(f'gpu_ids: {self.config.gpu_ids}, world_size: {self.world_size}')
        mp.spawn(
            self.train_worker,
            args=(self.world_size,),
            nprocs=self.world_size,
            join=True
        )

    def _get_split_spec(self, world_size, model_name):
        """根据 GPU 数量和模型类型返回切分策略"""
        if 'resnet' in model_name.lower():
            if world_size == 2:
                return {'layer3': SplitPoint.BEGINNING}
            elif world_size == 3:
                return {
                    'layer2': SplitPoint.BEGINNING,
                    'layer3': SplitPoint.BEGINNING
                }
            elif world_size == 4:
                return {
                    'layer2': SplitPoint.BEGINNING,
                    'layer3': SplitPoint.BEGINNING,
                    'layer4': SplitPoint.BEGINNING
                }
            else:
                return {'layer2': SplitPoint.BEGINNING}
        else:
            # 其它模型暂时不切
            return {}

    def train_worker(self, rank, world_size):
        """每个进程的训练 worker"""

        # ====================================================
        # 1. 初始化分布式
        # ====================================================
        gpu_id = self.config.gpu_ids[rank]
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Rank {rank} 使用 GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
        torch.cuda.set_device(device)

        setup_distributed(rank, world_size, backend=self.config.dist_backend, device=device)

        if rank == 0:
            print(f"✅ 进程组初始化完成 (Backend: {self.config.dist_backend})")

        # ====================================================
        # 2. 统一随机种子
        # ====================================================
        torch.manual_seed(self.config.seed)

        if rank == 0:
            print(f"🔧 所有 rank 使用统一随机种子: {self.config.seed}")

        # ====================================================
        # 3. 构建模型并切分为流水线 stages
        # ====================================================
        base_model = get_model(self.config.model, num_classes=self.config.num_classes)
        split_spec = self._get_split_spec(world_size, self.config.model)

        if rank == 0:
            print(f"🔧 构建模型: {self.config.model}")
            print(f"✂️  切分策略: {split_spec}")

        # ====================================================
        # 4. 创建流水线（Pipeline）
        # ====================================================
        mb_size = self.config.batch_size // self.chunks
        example_input = torch.randn(mb_size, 3, 32, 32)

        if rank == 0:
            print(f"🔍 使用示例输入 {example_input.shape} 进行追踪...")

        pipe = pipeline(
            module=base_model,
            mb_args=(example_input,),
            split_spec=split_spec
        )

        my_submodule = pipe.get_stage_module(rank)
        my_submodule.to(device)

        stage = PipelineStage(
            my_submodule,
            stage_index=rank,
            num_stages=pipe.num_stages,
            device=device,
        )

        # ====================================================
        # 5. 创建 Loss 累积器和调度器
        # ====================================================
        criterion = nn.CrossEntropyLoss()
        loss_accumulator = LossAccumulator(criterion)

        schedule = ScheduleGPipe(
            stage,
            n_microbatches=self.chunks,
            loss_fn=loss_accumulator
        )

        # ====================================================
        # 6. 优化器 & 学习率调度
        # ====================================================
        optimizer = optim.SGD(
            my_submodule.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度器（与 DDP 一致）
        if self.config.lr_schedule == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_schedule == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs
            )
        elif self.config.lr_schedule == 'multistep':
            milestones = [int(self.config.epochs * 0.5), int(self.config.epochs * 0.75)]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=self.config.lr_gamma
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # ====================================================
        # 7. 数据加载器
        # ====================================================
        trainloader, testloader, _ = get_dataloader(self.config, distributed=False)

        if rank == 0:
            print(f"✅ 数据加载器创建完成")
            print(f"📊 训练集: {len(trainloader.dataset)} 样本")
            print(f"📊 测试集: {len(testloader.dataset)} 样本\n")

        monitor = PerformanceMonitor()

        # 初始化历史记录（与 DDP 完全相同的格式）
        history = {}
        best_acc = 0.0
        if rank == world_size - 1:
            history = {
                'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': [],
                'epoch_time': [],
                'learning_rate': [],
                'gpu_memory_per_device': []
            }

        dist.barrier()

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"🎯 开始训练 {self.config.epochs} 个 Epochs")
            print(f"{'='*60}\n")

        # ====================================================
        # 8. 训练循环
        # ====================================================
        for epoch in range(self.config.epochs):
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{self.config.epochs}")
                print(f"{'='*60}")

            epoch_start = time.time()
            
            # 训练一个 epoch
            train_loss, train_acc = self.train_epoch(
                my_submodule, schedule, trainloader, optimizer,
                loss_accumulator, device, rank, world_size, epoch
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            epoch_time = time.time() - epoch_start

            # 收集所有 rank 的 GPU 内存信息（与 DDP 格式一致）
            gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            
            if rank == world_size - 1:
                # 收集所有 rank 的内存信息
                all_gpu_ids = [torch.zeros(1, dtype=torch.int32).to(device) for _ in range(world_size)]
                all_mem_allocated = [torch.zeros(1).to(device) for _ in range(world_size)]
                all_mem_reserved = [torch.zeros(1).to(device) for _ in range(world_size)]
            else:
                all_gpu_ids = None
                all_mem_allocated = None
                all_mem_reserved = None
            
            local_gpu_id = torch.tensor([gpu_id], dtype=torch.int32).to(device)
            local_mem_allocated = torch.tensor([gpu_mem_allocated]).to(device)
            local_mem_reserved = torch.tensor([gpu_mem_reserved]).to(device)
            
            if rank == world_size - 1:
                dist.gather(local_gpu_id, gather_list=all_gpu_ids, dst=world_size-1)
                dist.gather(local_mem_allocated, gather_list=all_mem_allocated, dst=world_size-1)
                dist.gather(local_mem_reserved, gather_list=all_mem_reserved, dst=world_size-1)
            else:
                dist.gather(local_gpu_id, dst=world_size-1)
                dist.gather(local_mem_allocated, dst=world_size-1)
                dist.gather(local_mem_reserved, dst=world_size-1)
            
            # 在训练循环中
            test_loss, test_acc = self.evaluate_with_pipeline(
                my_submodule, schedule, testloader, criterion,
                loss_accumulator, device, rank, world_size
            )


            # 只在最后一个 rank 记录和打印（与 DDP 格式完全一致）
            if rank == world_size - 1:
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
                history['epoch_time'].append(epoch_time)
                history['learning_rate'].append(current_lr)
                
                # 记录所有 GPU 的内存（与 DDP 格式一致）
                gpu_mem_per_device = []
                for r in range(world_size):
                    gpu_mem_per_device.append({
                        'device_id': int(all_gpu_ids[r].item()),
                        'allocated': all_mem_allocated[r].item(),
                        'reserved': all_mem_reserved[r].item()
                    })
                history['gpu_memory_per_device'].append(gpu_mem_per_device)

                # 打印格式与 DDP 一致
                print(f'\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
                print(f'测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
                print(f'学习率: {current_lr:.6f}, 时间: {epoch_time:.2f}s')
                print('GPU内存使用:')
                for mem_info in gpu_mem_per_device:
                    print(f'  GPU {mem_info["device_id"]} - '
                          f'已分配: {mem_info["allocated"]:.2f}GB, '
                          f'已保留: {mem_info["reserved"]:.2f}GB')
                
                # 保存最佳模型
                if test_acc > best_acc:
                    print(f'最佳准确率更新: {best_acc:.2f}% -> {test_acc:.2f}%')
                    best_acc = test_acc
                    # self.save_checkpoint(
                    #     rank, epoch, my_submodule, optimizer, scheduler,
                    #     history, best_acc, is_best=True
                    # )
                
                # # 定期保存
                # if (epoch + 1) % self.config.save_interval == 0:
                #     self.save_checkpoint(
                #         rank, epoch, my_submodule, optimizer, scheduler,
                #         history, best_acc
                #     )

            dist.barrier()

        # 训练结束
        if rank == world_size - 1:
            print(f'\n{"="*60}')
            print('训练完成!')
            print(f'{"="*60}')
            print(f'总时间: {sum(history["epoch_time"]):.2f}秒 ({sum(history["epoch_time"])/60:.2f}分钟)')
            print(f'平均每epoch: {sum(history["epoch_time"])/self.config.epochs:.2f}秒')
            print(f'最佳测试准确率: {best_acc:.2f}%')

            self.save_checkpoint(
                rank,
                self.config.epochs - 1,
                my_submodule,
                optimizer,
                scheduler,
                history,
                best_acc,
                final=True
            )

        cleanup_distributed()

    def train_epoch(self, model, schedule, trainloader, optimizer,
                   loss_accumulator, device, rank, world_size, epoch):
        """训练一个 epoch"""
        model.train()
        
        torch.manual_seed(self.config.seed + epoch)
        
        # 重置累积器
        loss_accumulator.reset()
        
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # 检查 batch 大小
            if inputs.size(0) != self.config.batch_size:
                if rank == 0:
                    print(f"⚠️  跳过不完整的 batch {batch_idx}")
                continue
            
            optimizer.zero_grad()

            # Stage 不同，行为不同
            if rank == 0:
                # 第一段：只需要输入
                inputs = inputs.to(device, non_blocking=True)
                schedule.step(inputs)
            elif rank == world_size - 1:
                # 最后一段：需要标签，并计算 loss 和 accuracy
                targets = targets.to(device, non_blocking=True)
                schedule.step(target=targets)
                
                # 获取这个 batch 的统计
                loss_value = loss_accumulator.get_average_loss()
                acc_value = loss_accumulator.get_accuracy()
                
                running_loss += loss_value * loss_accumulator.total_samples
                running_correct += loss_accumulator.total_correct
                running_total += loss_accumulator.total_samples
                num_batches += 1
                
                # 重置累积器，准备下一个 batch
                loss_accumulator.reset()
            else:
                # 中间段：只做前后向流水
                schedule.step()

            optimizer.step()

            # 只在最后一段打印日志
            if rank == world_size - 1 and batch_idx % self.config.log_interval == 0:
                avg_loss = running_loss / running_total if running_total > 0 else 0
                avg_acc = 100. * running_correct / running_total if running_total > 0 else 0
                gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
                gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f'Rank {rank} | Epoch: {epoch} [{batch_idx}/{len(trainloader)}] '
                      f'Loss: {loss_value:.4f} | Acc: {acc_value:.2f}% '
                      f'| GPU Mem: {gpu_mem:.2f}GB/{gpu_mem_reserved:.2f}GB')

        # 计算平均 loss 和 accuracy
        if rank == world_size - 1:
            avg_loss = running_loss / running_total if running_total > 0 else 0
            avg_acc = 100. * running_correct / running_total if running_total > 0 else 0
            return avg_loss, avg_acc
        else:
            return 0.0, 0.0

    def evaluate_with_pipeline(self, model, schedule, testloader, criterion, 
                            loss_accumulator, device, rank, world_size):
        """
        使用 Pipeline 进行评估
        """
        model.eval()  # 设置模型为评估模式
        loss_accumulator.reset()
        
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        num_batches = 0
        
        # 完全移除torch.no_grad()，让Pipeline正常工作
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # 检查 batch 大小，确保所有rank处理相同数量的batch
            if inputs.size(0) != self.config.batch_size:
                continue
            
            # Stage 不同，行为不同
            if rank == 0:
                inputs = inputs.to(device, non_blocking=True)
                schedule.step(inputs)
            elif rank == world_size - 1:
                targets = targets.to(device, non_blocking=True)
                schedule.step(target=targets)
                
                loss_value = loss_accumulator.get_average_loss()
                acc_value = loss_accumulator.get_accuracy()
                
                running_loss += loss_value * loss_accumulator.total_samples
                running_correct += loss_accumulator.total_correct
                running_total += loss_accumulator.total_samples
                num_batches += 1
                
                loss_accumulator.reset()
            else:
                # 中间stage也需要参与Pipeline
                schedule.step()
        
        # 计算平均值
        if rank == world_size - 1:
            avg_loss = running_loss / running_total if running_total > 0 else 0
            avg_acc = 100. * running_correct / running_total if running_total > 0 else 0
            
            # 广播结果
            result = torch.tensor([avg_loss, avg_acc]).to(device)
            dist.broadcast(result, src=world_size-1)
            
            return avg_loss, avg_acc
        else:
            result = torch.zeros(2).to(device)
            dist.broadcast(result, src=world_size-1)
            
            return result[0].item(), result[1].item()



    def save_checkpoint(self, rank, epoch, model, optimizer, scheduler,
                       history, best_acc, is_best=False, final=False):
        """保存 Checkpoint（与 DDP 格式完全一致）"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        print(f"Rank {rank} 正在保存 checkpoint..., is_best={is_best}, final={final}")
        # 转换 config 为字典（与 DDP 一致）
        config_dict = self.config.__dict__.copy() if hasattr(self.config, '__dict__') else self.config


        if final:
            path = self.config.final_checkpoint_path
        if path is None:
            return

        # 保存格式（与 DDP 一致）
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'config': config_dict,
            # Pipeline 特有的额外信息
            'stage_index': rank,
            'num_stages': self.world_size,
            'history': history
        }


        torch.save(state, path)
        
        if is_best:
            print(f'保存最佳模型到: {path}')
        elif final:
            print(f'保存最终模型到: {path}')
        else:
            print(f'保存checkpoint到: {path}')
