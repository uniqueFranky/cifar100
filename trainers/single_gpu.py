"""
单GPU训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

from models import get_model
from utils import get_dataloader, PerformanceMonitor


class SingleGPUTrainer:
    """单GPU训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu_ids[0]}')
        
        print(f"\n初始化单GPU训练器...")
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = get_model(config.model, config.num_classes).to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        if config.lr_schedule == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma
            )
        elif config.lr_schedule == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs
            )
        elif config.lr_schedule == 'multistep':
            milestones = [int(config.epochs * 0.5), int(config.epochs * 0.75)]
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=config.lr_gamma
            )
        
        # 数据加载器
        self.trainloader, self.testloader, _ = get_dataloader(config, distributed=False)
        
        # 性能监控
        self.monitor = PerformanceMonitor()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_time': [],
            'gpu_memory_per_device': []  # 统一格式：每个epoch记录所有设备的内存
        }
        
        self.start_epoch = 0
        self.best_acc = 0
        
        # 恢复训练
        if config.resume:
            self.load_checkpoint(config.resume)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            self.monitor.start_timer()
            
            # 数据转移到GPU
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            batch_time = self.monitor.end_timer('train_batch_time')
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印日志
            if batch_idx % self.config.log_interval == 0:
                gpu_mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f'Epoch: {epoch} [{batch_idx}/{len(self.trainloader)}] '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% '
                      f'| Time: {batch_time:.4f}s | GPU Mem: {gpu_mem_allocated:.2f}GB/{gpu_mem_reserved:.2f}GB')
        
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(self.trainloader)
        epoch_acc = 100. * correct / total
        
        # 记录GPU内存使用（统一格式）
        gpu_mem_per_device = [{
            'device_id': self.config.gpu_ids[0],
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
            'reserved': torch.cuda.memory_reserved(self.device) / 1024**3
        }]
        
        return epoch_loss, epoch_acc, epoch_time, gpu_mem_per_device
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(self.testloader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self):
        """完整训练流程"""
        print(f"\n开始训练 {self.config.epochs} 个epochs...")
        print(f"Batch size: {self.config.batch_size}")
        print(f"学习率: {self.config.lr}")
        
        total_start = time.time()
        
        for epoch in range(self.start_epoch, self.config.epochs):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch+1}/{self.config.epochs}')
            print(f'{"="*60}')
            
            # 训练
            train_loss, train_acc, epoch_time, gpu_mem_per_device = self.train_epoch(epoch)
            
            # 评估
            test_loss, test_acc = self.evaluate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['epoch_time'].append(epoch_time)
            self.history['gpu_memory_per_device'].append(gpu_mem_per_device)
            
            # 打印结果
            print(f'\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
            print(f'学习率: {current_lr:.6f}, 时间: {epoch_time:.2f}s')
            print('GPU内存使用:')
            for mem_info in gpu_mem_per_device:
                print(f'  GPU {mem_info["device_id"]} - 已分配: {mem_info["allocated"]:.2f}GB, 已保留: {mem_info["reserved"]:.2f}GB')
            
            # 保存最佳模型
            if test_acc > self.best_acc:
                print(f'最佳准确率更新: {self.best_acc:.2f}% -> {test_acc:.2f}%')
                self.best_acc = test_acc
                self.save_checkpoint(epoch, is_best=True)
            
            # 定期保存
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
        
        total_time = time.time() - total_start
        
        # 打印最终统计
        print(f'\n{"="*60}')
        print('训练完成!')
        print(f'{"="*60}')
        print(f'总时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)')
        print(f'平均每epoch: {total_time/self.config.epochs:.2f}秒')
        print(f'最佳测试准确率: {self.best_acc:.2f}%')
        
        # 保存最终模型
        self.save_checkpoint(self.config.epochs - 1, is_final=True)
    
    def save_checkpoint(self, epoch, is_best=False, is_final=False):
        """保存checkpoint，包含完整config"""
        # 转换config为字典格式以确保完整保存
        config_dict = self.config.__dict__.copy() if hasattr(self.config, '__dict__') else self.config
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history,
            'config': config_dict,  # 保存完整config内容
        }
        
        if is_best:
            path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(state, path)
            print(f'保存最佳模型到: {path}')
        elif is_final:
            # 支持自定义最终checkpoint路径
            if self.config.final_checkpoint_path:
                path = self.config.final_checkpoint_path
            else:
                path = os.path.join(self.config.save_dir, 'final_model.pth')
            torch.save(state, path)
            print(f'保存最终模型到: {path}')
        else:
            path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(state, path)
            print(f'保存checkpoint到: {path}')
    
    def load_checkpoint(self, path):
        """加载checkpoint"""
        print(f'从 {path} 加载checkpoint...')
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        self.history = checkpoint['history']
        
        print(f'恢复训练从epoch {self.start_epoch}, 最佳准确率: {self.best_acc:.2f}%')
