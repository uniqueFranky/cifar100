# trainers/hybrid_parallel.py
"""
混合并行训练器 - 结合模型并行和数据并行
每个模型分布在2个GPU上，多个这样的模型并行训练不同数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import time
import os

from models import get_model
from .model_parallel import ModelParallelResNet
from utils import (
    get_dataloader,
    PerformanceMonitor,
    setup_distributed,
    cleanup_distributed,
    is_main_process
)


class HybridParallelTrainer:
    """混合并行训练器"""
    
    def __init__(self, config):
        self.config = config
        
        # 验证GPU配置
        self._validate_gpu_config()
        
        # 计算并行配置
        self.gpus_per_model = 2  # 每个模型使用2个GPU
        self.num_models = len(config.gpu_ids) // self.gpus_per_model
        
        if len(config.gpu_ids) % self.gpus_per_model != 0:
            raise ValueError(f"GPU数量({len(config.gpu_ids)})必须是{self.gpus_per_model}的倍数")
        
        # GPU分组
        self.gpu_groups = []
        for i in range(self.num_models):
            start_idx = i * self.gpus_per_model
            gpu_group = config.gpu_ids[start_idx:start_idx + self.gpus_per_model]
            self.gpu_groups.append(gpu_group)
    
    def _validate_gpu_config(self):
        """验证GPU配置"""
        if len(self.config.gpu_ids) < 4:
            raise ValueError("混合并行模式至少需要4个GPU")
        
        if len(self.config.gpu_ids) % 2 != 0:
            raise ValueError("混合并行模式GPU数量必须是偶数")
    
    def launch(self):
        """启动混合并行训练"""
        world_size = self.num_models
        
        print(f"\n启动混合并行训练...")
        print(f"使用 {len(self.config.gpu_ids)} 个GPU: {self.config.gpu_ids}")
        print(f"模型实例数: {world_size}")
        print(f"每个模型使用GPU数: {self.gpus_per_model}")
        print(f"每个GPU的batch size: {self.config.batch_size}")
        print(f"有效batch size: {self.config.effective_batch_size}")
        
        for i, gpu_group in enumerate(self.gpu_groups):
            print(f"  模型 {i}: GPU {gpu_group} (主GPU: {gpu_group[0]})")
        
        # 使用spawn启动多进程
        mp.spawn(
            self.train_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    
    def train_worker(self, rank, world_size):
        """每个进程的训练worker"""
        try:
            # 获取当前进程的GPU组
            device_ids = self.gpu_groups[rank]
            # 使用第一个GPU作为主GPU（用于分布式通信）
            main_gpu = device_ids[0]
            main_device = torch.device(f'cuda:{main_gpu}')
            
            if is_main_process(rank):
                print(f"\n进程 {rank} 开始初始化")
                print(f"GPU组: {device_ids}")
                print(f"主GPU: {main_gpu} (用于分布式通信)")
                print(f"模型并行GPU: {device_ids}")
            
            # 初始化分布式环境，传递主GPU
            setup_distributed(rank, world_size, backend=self.config.dist_backend, device=main_gpu)
            
            if is_main_process(rank):
                print(f"进程 {rank} 分布式环境初始化完成")
            
            # 创建模型并行模型
            model = self._create_model_parallel_model(device_ids)
            
            if is_main_process(rank):
                print(f"模型并行模型创建完成")
                print(f"设备映射: {model.device_map}")
            
            # 获取输出设备（fc层所在的设备）
            output_device = torch.device(model.device_map['fc'])
            
            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss().to(output_device)
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
                checkpoint = torch.load(self.config.resume, map_location=main_device)
                model.load_state_dict(checkpoint['model_state_dict'])
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
                train_loss, train_acc, epoch_time, gpu_mem_info = self.train_epoch(
                    model, trainloader, criterion, optimizer,
                    device_ids, main_device, monitor, rank, epoch
                )
                
                # 评估
                test_loss, test_acc = self.evaluate(
                    model, testloader, criterion,
                    device_ids, main_device, rank, world_size
                )
                
                # 更新学习率
                scheduler.step()
                
                # 收集所有rank的GPU内存信息（统一格式）
                if is_main_process(rank):
                    # 创建tensor来收集所有rank的内存信息和GPU ID
                    all_gpu_ids = [torch.zeros(len(device_ids), dtype=torch.int32).to(main_device) for _ in range(world_size)]
                    all_mem_allocated = [torch.zeros(len(device_ids)).to(main_device) for _ in range(world_size)]
                    all_mem_reserved = [torch.zeros(len(device_ids)).to(main_device) for _ in range(world_size)]
                else:
                    all_gpu_ids = None
                    all_mem_allocated = None
                    all_mem_reserved = None
                
                # 将本地信息发送到主进程（确保在主设备上）
                local_gpu_ids = torch.tensor(device_ids, dtype=torch.int32).to(main_device)
                local_mem_allocated = torch.tensor([info['allocated'] for info in gpu_mem_info]).to(main_device)
                local_mem_reserved = torch.tensor([info['reserved'] for info in gpu_mem_info]).to(main_device)
                
                if is_main_process(rank):
                    dist.gather(local_gpu_ids, gather_list=all_gpu_ids, dst=0)
                    dist.gather(local_mem_allocated, gather_list=all_mem_allocated, dst=0)
                    dist.gather(local_mem_reserved, gather_list=all_mem_reserved, dst=0)
                else:
                    dist.gather(local_gpu_ids, dst=0)
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
                        for i in range(len(device_ids)):
                            gpu_mem_per_device.append({
                                'device_id': int(all_gpu_ids[r][i].item()),
                                'allocated': all_mem_allocated[r][i].item(),
                                'reserved': all_mem_reserved[r][i].item()
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
        
        except Exception as e:
            print(f"Rank {rank} 训练出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理分布式环境
            if dist.is_initialized():
                cleanup_distributed()
    
    def _create_model_parallel_model(self, device_ids):
        """创建模型并行模型"""
        # 首先创建原始模型
        orig_model = get_model(self.config.model, self.config.num_classes)
        
        # 创建设备映射：将模型的不同部分分配到不同GPU
        device_map = self._create_device_map(device_ids)
        
        # 创建模型并行版本
        model = ModelParallelResNet(orig_model, device_map)
        
        return model
    
    def _create_device_map(self, device_ids):
        """创建设备映射，将模型组件分配到不同GPU"""
        device1 = f'cuda:{device_ids[0]}'  # 第一个GPU
        device2 = f'cuda:{device_ids[1]}'  # 第二个GPU
        
        # 将模型前半部分放在第一个GPU，后半部分放在第二个GPU
        device_map = {
            'conv1': device1,
            'bn1': device1,
            'layer1': device1,
            'layer2': device1,      # 前半部分在GPU1
            'layer3': device2,      # 后半部分在GPU2
            'layer4': device2,
            'avg_pool': device2,
            'fc': device2
        }
        
        return device_map
    
    def _sync_gradients(self, model, main_device):
        """同步模型梯度 - 将所有梯度移动到主设备进行同步"""
        # 收集所有参数的梯度，并移动到主设备
        all_grads = []
        param_shapes = []
        param_devices = []
        
        for param in model.parameters():
            if param.grad is not None:
                # 记录原始形状和设备
                param_shapes.append(param.grad.shape)
                param_devices.append(param.grad.device)
                
                # 将梯度移动到主设备并展平
                grad_on_main = param.grad.data.to(main_device).flatten()
                all_grads.append(grad_on_main)
        
        if not all_grads:
            return
        
        # 将所有梯度连接成一个大张量
        combined_grads = torch.cat(all_grads)
        
        # 执行分布式同步
        dist.all_reduce(combined_grads, op=dist.ReduceOp.SUM)
        combined_grads /= dist.get_world_size()
        
        # 将同步后的梯度分割并移回原始设备
        start_idx = 0
        param_idx = 0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_size = param_shapes[param_idx].numel()
                original_device = param_devices[param_idx]
                
                # 提取对应的梯度片段
                grad_slice = combined_grads[start_idx:start_idx + grad_size]
                
                # 重塑并移回原始设备
                synced_grad = grad_slice.view(param_shapes[param_idx]).to(original_device)
                param.grad.data.copy_(synced_grad)
                
                start_idx += grad_size
                param_idx += 1
    
    def train_epoch(self, model, trainloader, criterion, optimizer,
                   device_ids, main_device, monitor, rank, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        # 获取输入和输出设备
        input_device = torch.device(f'cuda:{device_ids[0]}')
        output_device = torch.device(model.device_map['fc'])
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            monitor.start_timer()
            
            # 数据转移到输入设备
            inputs = inputs.to(input_device, non_blocking=True)
            targets = targets.to(output_device, non_blocking=True)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 同步梯度（传递主设备）
            self._sync_gradients(model, main_device)
            
            optimizer.step()
            
            batch_time = monitor.end_timer('train_batch_time')
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印日志（只在主进程）
            if is_main_process(rank) and batch_idx % self.config.log_interval == 0:
                # 计算所有设备的内存使用
                gpu_mem_info = []
                for device_id in device_ids:
                    device = torch.device(f'cuda:{device_id}')
                    gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    gpu_mem_info.append(f"GPU{device_id}: {gpu_mem_allocated:.2f}GB/{gpu_mem_reserved:.2f}GB")
                
                mem_str = " | ".join(gpu_mem_info)
                print(f'Rank {rank} | Epoch: {epoch} [{batch_idx}/{len(trainloader)}] '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% '
                      f'| Time: {batch_time:.4f}s | {mem_str}')
        
        epoch_time = time.time() - epoch_start
        
        # 计算本地统计
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        
        # 记录GPU内存使用（统一格式）
        gpu_mem_info = []
        for device_id in device_ids:
            device = torch.device(f'cuda:{device_id}')
            gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            gpu_mem_info.append({
                'device_id': device_id,
                'allocated': gpu_mem_allocated,
                'reserved': gpu_mem_reserved
            })
        
        return epoch_loss, epoch_acc, epoch_time, gpu_mem_info
    
    def evaluate(self, model, testloader, criterion, device_ids, main_device, rank, world_size):
        """评估模型"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # 获取输入和输出设备
        input_device = torch.device(f'cuda:{device_ids[0]}')
        output_device = torch.device(model.device_map['fc'])
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(input_device, non_blocking=True)
                targets = targets.to(output_device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # 将本地统计转换为tensor，并确保在主设备上进行分布式通信
        local_correct = torch.tensor(correct, dtype=torch.float32).to(main_device)
        local_total = torch.tensor(total, dtype=torch.float32).to(main_device)
        local_loss = torch.tensor(test_loss, dtype=torch.float32).to(main_device)
        
        # 聚合所有进程的结果（现在所有tensor都在主设备上）
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
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'history': history,
            'config': config_dict,  # 保存完整config内容
        }
        
        if is_best:
            path = os.path.join(self.config.save_dir, 'best_model_hybrid.pth')
            torch.save(state, path)
            print(f'保存最佳模型到: {path}')
        elif is_final:
            # 支持自定义最终checkpoint路径
            if self.config.final_checkpoint_path:
                path = self.config.final_checkpoint_path
            else:
                path = os.path.join(self.config.save_dir, 'final_model_hybrid.pth')
            torch.save(state, path)
            print(f'保存最终模型到: {path}')
        else:
            path = os.path.join(self.config.save_dir, f'checkpoint_hybrid_epoch_{epoch+1}.pth')
            torch.save(state, path)
            print(f'保存checkpoint到: {path}')
