"""
DistributedDataParallel训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os

from models import get_model
from utils import (
    get_dataloader,
    PerformanceMonitor,
    setup_distributed,
    cleanup_distributed,
    is_main_process
)


class DDPTrainer:
    """DistributedDataParallel训练器"""
    def __init__(self, config):
        self.config = config
    
    def launch(self):
        """启动多进程训练"""
        world_size = self.config.num_gpus
        
        print(f"\n启动DDP训练...")
        print(f"使用 {world_size} 个GPU: {self.config.gpu_ids}")
        print(f"每个GPU的batch size: {self.config.batch_size}")
        print(f"有效batch size: {self.config.effective_batch_size}")
        
        # 使用spawn启动多进程
        mp.spawn(
            self.train_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    
    def train_worker(self, rank, world_size):
        """每个进程的训练worker"""
        # 初始化分布式环境
        setup_distributed(rank, world_size, backend=self.config.dist_backend)
        
        # 设置设备
        device = torch.device(f'cuda:{self.config.gpu_ids[rank]}')
        torch.cuda.set_device(device)
        
        if is_main_process(rank):
            print(f"\n进程 {rank} 初始化完成")
        
        # 创建模型
        model = get_model(self.config.model, self.config.num_classes).to(device)
        
        # 使用DDP包装模型
        model = DDP(
            model,
            device_ids=[self.config.gpu_ids[rank]],
            output_device=self.config.gpu_ids[rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )
        
        if is_main_process(rank):
            print(f"模型已包装为DDP")
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
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
        
        # 数据加载器（使用DistributedSampler）
        trainloader, testloader, train_sampler = get_dataloader(
            self.config,
            distributed=True,
            rank=rank,
            world_size=world_size
        )
        
        if is_main_process(rank):
            print(f"数据加载器创建完成")
            print(f"训练样本数: {len(trainloader.dataset)}")
            print(f"测试样本数: {len(testloader.dataset)}")
        
        # 性能监控
        monitor = PerformanceMonitor()
        
        # 训练历史（只在主进程记录）
        if is_main_process(rank):
            history = {
                'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': [],
                'epoch_time': [],
                'gpu_memory_per_device': []  # 统一格式：每个epoch记录所有设备的内存
            }
        
        start_epoch = 0
        best_acc = 0
        
        # 恢复训练
        if self.config.resume and is_main_process(rank):
            checkpoint = torch.load(self.config.resume, map_location=device)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            if 'history' in checkpoint:
                history = checkpoint['history']
            print(f'恢复训练从epoch {start_epoch}, 最佳准确率: {best_acc:.2f}%')
        
        # 同步所有进程
        dist.barrier()
        
        if is_main_process(rank):
            print(f"\n开始训练 {self.config.epochs} 个epochs...")
        
        total_start = time.time()
        
        # 训练循环
        for epoch in range(start_epoch, self.config.epochs):
            if is_main_process(rank):
                print(f'\n{"="*60}')
                print(f'Epoch {epoch+1}/{self.config.epochs}')
                print(f'{"="*60}')
            
            # 设置epoch（用于打乱数据）
            train_sampler.set_epoch(epoch)
            
            # 训练
            train_loss, train_acc, epoch_time, gpu_id, gpu_mem_allocated, gpu_mem_reserved = self.train_epoch(
                model, trainloader, criterion, optimizer,
                device, monitor, rank, epoch
            )
            
            # 评估
            test_loss, test_acc = self.evaluate(
                model, testloader, criterion,
                device, rank, world_size
            )
            
            # 更新学习率
            scheduler.step()
            
            # 收集所有rank的GPU内存信息（统一格式）
            if is_main_process(rank):
                # 创建tensor来收集所有rank的内存信息和GPU ID
                all_gpu_ids = [torch.zeros(1, dtype=torch.int32).to(device) for _ in range(world_size)]
                all_mem_allocated = [torch.zeros(1).to(device) for _ in range(world_size)]
                all_mem_reserved = [torch.zeros(1).to(device) for _ in range(world_size)]
            else:
                all_gpu_ids = None
                all_mem_allocated = None
                all_mem_reserved = None
            
            # 将本地信息发送到主进程
            local_gpu_id = torch.tensor([gpu_id], dtype=torch.int32).to(device)
            local_mem_allocated = torch.tensor([gpu_mem_allocated]).to(device)
            local_mem_reserved = torch.tensor([gpu_mem_reserved]).to(device)
            
            if is_main_process(rank):
                dist.gather(local_gpu_id, gather_list=all_gpu_ids, dst=0)
                dist.gather(local_mem_allocated, gather_list=all_mem_allocated, dst=0)
                dist.gather(local_mem_reserved, gather_list=all_mem_reserved, dst=0)
            else:
                dist.gather(local_gpu_id, dst=0)
                dist.gather(local_mem_allocated, dst=0)
                dist.gather(local_mem_reserved, dst=0)
            
            # 只在主进程记录和打印
            if is_main_process(rank):
                current_lr = optimizer.param_groups[0]['lr']
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
                history['epoch_time'].append(epoch_time)
                
                # 记录所有GPU的内存（统一格式）
                gpu_mem_per_device = []
                for r in range(world_size):
                    gpu_mem_per_device.append({
                        'device_id': int(all_gpu_ids[r].item()),
                        'allocated': all_mem_allocated[r].item(),
                        'reserved': all_mem_reserved[r].item()
                    })
                history['gpu_memory_per_device'].append(gpu_mem_per_device)
                
                print(f'\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
                print(f'测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
                print(f'学习率: {current_lr:.6f}, 时间: {epoch_time:.2f}s')
                # 打印所有GPU的内存使用
                print('GPU内存使用:')
                for mem_info in gpu_mem_per_device:
                    print(f'  GPU {mem_info["device_id"]} - 已分配: {mem_info["allocated"]:.2f}GB, 已保留: {mem_info["reserved"]:.2f}GB')
                
                # 保存最佳模型
                if test_acc > best_acc:
                    print(f'最佳准确率更新: {best_acc:.2f}% -> {test_acc:.2f}%')
                    best_acc = test_acc
                    self.save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, best_acc, history,
                        is_best=True
                    )
                
                # 定期保存
                if (epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, best_acc, history
                    )
            
            # 同步所有进程
            dist.barrier()
        
        total_time = time.time() - total_start
        
        # 打印最终统计（只在主进程）
        if is_main_process(rank):
            print(f'\n{"="*60}')
            print('训练完成!')
            print(f'{"="*60}')
            print(f'总时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)')
            print(f'平均每epoch: {total_time/self.config.epochs:.2f}秒')
            print(f'最佳测试准确率: {best_acc:.2f}%')
            
            # 保存最终模型
            self.save_checkpoint(
                model, optimizer, scheduler,
                self.config.epochs - 1, best_acc, history,
                is_final=True
            )
        
        # 清理分布式环境
        cleanup_distributed()
    
    def train_epoch(self, model, trainloader, criterion, optimizer,
                   device, monitor, rank, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            monitor.start_timer()
            
            # 数据转移到GPU
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time = monitor.end_timer('train_batch_time')
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印日志（只在主进程）
            if is_main_process(rank) and batch_idx % self.config.log_interval == 0:
                gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f'Rank {rank} | Epoch: {epoch} [{batch_idx}/{len(trainloader)}] '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% '
                      f'| Time: {batch_time:.4f}s | GPU Mem: {gpu_mem_allocated:.2f}GB/{gpu_mem_reserved:.2f}GB')
        
        epoch_time = time.time() - epoch_start
        
        # 计算本地统计
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        
        # 记录GPU内存使用（统一格式：使用实际的GPU ID）
        gpu_id = self.config.gpu_ids[rank]
        gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        # 同步所有进程的统计（可选）
        # 这里我们只返回rank 0的统计，也可以聚合所有进程的统计
        
        return epoch_loss, epoch_acc, epoch_time, gpu_id, gpu_mem_allocated, gpu_mem_reserved
    
    def evaluate(self, model, testloader, criterion, device, rank, world_size):
        """评估模型"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # 将本地统计转换为tensor
        local_correct = torch.tensor(correct, dtype=torch.float32).to(device)
        local_total = torch.tensor(total, dtype=torch.float32).to(device)
        local_loss = torch.tensor(test_loss, dtype=torch.float32).to(device)
        
        # 聚合所有GPU的结果
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        
        # 计算全局指标
        test_acc = 100. * local_correct.item() / local_total.item()
        test_loss = local_loss.item() / (len(testloader) * world_size)
        
        return test_loss, test_acc
    
    def save_checkpoint(self, model, optimizer, scheduler,
                       epoch, best_acc, history,
                       is_best=False, is_final=False):
        """保存checkpoint，包含完整config（只在主进程）"""
        # 转换config为字典格式以确保完整保存
        config_dict = self.config.__dict__.copy() if hasattr(self.config, '__dict__') else self.config
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # 注意：使用module
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'history': history,
            'config': config_dict,  # 保存完整config内容
        }
        
        if is_best:
            path = os.path.join(self.config.save_dir, 'best_model_ddp.pth')
            torch.save(state, path)
            print(f'保存最佳模型到: {path}')
        elif is_final:
            # 支持自定义最终checkpoint路径
            if self.config.final_checkpoint_path:
                path = self.config.final_checkpoint_path
            else:
                path = os.path.join(self.config.save_dir, 'final_model_ddp.pth')
            torch.save(state, path)
            print(f'保存最终模型到: {path}')
        else:
            path = os.path.join(self.config.save_dir, f'checkpoint_ddp_epoch_{epoch+1}.pth')
            torch.save(state, path)
            print(f'保存checkpoint到: {path}')
