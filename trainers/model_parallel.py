"""
简单的ModelParallel训练器

说明:
- 将ResNet的不同模块切分到多个GPU上(按模块划分)
- 在forward中处理跨设备的数据搬移
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import torch.nn.functional as F

from models import get_model
from utils import get_dataloader, PerformanceMonitor


class ModelParallelResNet(nn.Module):
    """把ResNet的子模块部署到不同设备上的包装器"""
    def __init__(self, orig_model, device_map):
        super(ModelParallelResNet, self).__init__()
        # 保留原始子模块引用并移动到目标设备
        self.device_map = device_map

        # 按名称绑定子模块，保持state_dict键名一致
        self.conv1 = orig_model.conv1.to(device_map['conv1'])
        self.bn1 = orig_model.bn1.to(device_map['bn1'])
        self.layer1 = orig_model.layer1.to(device_map['layer1'])
        self.layer2 = orig_model.layer2.to(device_map['layer2'])
        self.layer3 = orig_model.layer3.to(device_map['layer3'])
        self.layer4 = orig_model.layer4.to(device_map['layer4'])
        self.avg_pool = orig_model.avg_pool.to(device_map['avg_pool'])
        self.fc = orig_model.fc.to(device_map['fc'])

    def forward(self, x):
        # stem
        x = x.to(self.device_map['conv1'])
        x = self.conv1(x)
        if self.device_map['bn1'] != self.device_map['conv1']:
            x = x.to(self.device_map['bn1'])
        x = self.bn1(x)
        x = F.relu(x)

        # layer1
        if x.device != self.device_map['layer1']:
            x = x.to(self.device_map['layer1'])
        x = self.layer1(x)

        # layer2
        if x.device != self.device_map['layer2']:
            x = x.to(self.device_map['layer2'])
        x = self.layer2(x)

        # layer3
        if x.device != self.device_map['layer3']:
            x = x.to(self.device_map['layer3'])
        x = self.layer3(x)

        # layer4
        if x.device != self.device_map['layer4']:
            x = x.to(self.device_map['layer4'])
        x = self.layer4(x)

        # head
        if x.device != self.device_map['avg_pool']:
            x = x.to(self.device_map['avg_pool'])
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if x.device != self.device_map['fc']:
            x = x.to(self.device_map['fc'])
        x = self.fc(x)

        return x


class ModelParallelTrainer:
    """基于模块划分的ModelParallel训练器（多卡模型并行）"""
    def __init__(self, config):
        self.config = config
        self.gpu_ids = config.gpu_ids

        print(f"\n初始化ModelParallel训练器...")
        print(f"使用GPU: {self.gpu_ids}")

        # 创建基础模型
        base_model = get_model(config.model, config.num_classes)

        # 拆分模块列表（保持与ResNet实现一致）
        components = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'avg_pool', 'fc']

        # 将组件按顺序分配到GPUs（尽量连续划分）
        num_stages = len(self.gpu_ids)
        groups = [[] for _ in range(num_stages)]
        # distribute contiguous components
        per = max(1, len(components) // num_stages)
        idx = 0
        for comp in components:
            groups[min(idx // per, num_stages - 1)].append(comp)
            idx += 1

        # 若某个gpu没分到组件，给最后一个补齐
        for i in range(len(groups)):
            if len(groups[i]) == 0:
                groups[i].append(components[-1])

        # 产生device_map：为每个组件指定device
        device_map = {}
        for stage_idx, comps in enumerate(groups):
            dev = torch.device(f'cuda:{self.gpu_ids[stage_idx]}')
            for c in comps:
                device_map[c] = dev

        # ============ 新增: 展示设备分配信息 ============
        print(f"\n{'='*60}")
        print("模型并行 - 层分配方案")
        print(f"{'='*60}")
        
        # 按GPU分组展示
        for stage_idx, gpu_id in enumerate(self.gpu_ids):
            dev = torch.device(f'cuda:{gpu_id}')
            layers_on_this_gpu = [comp for comp, d in device_map.items() if d == dev]
            print(f"\nGPU {gpu_id} (cuda:{gpu_id}):")
            print(f"  └─ 分配的层: {', '.join(layers_on_this_gpu)}")
            
            # 计算该GPU上的参数量
            temp_model = get_model(config.model, config.num_classes)
            param_count = 0
            for comp in layers_on_this_gpu:
                if hasattr(temp_model, comp):
                    module = getattr(temp_model, comp)
                    param_count += sum(p.numel() for p in module.parameters())
            print(f"  └─ 参数量: {param_count:,} ({param_count/1e6:.2f}M)")
        
        # 展示详细的层到设备映射
        print(f"\n{'-'*60}")
        print("详细映射表:")
        print(f"{'-'*60}")
        for comp in components:
            dev = device_map[comp]
            gpu_id = dev.index
            print(f"  {comp:12s} -> GPU {gpu_id} (cuda:{gpu_id})")
        print(f"{'='*60}\n")
        # ============================================

        # 构建ModelParallelResNet
        self.model = ModelParallelResNet(base_model, device_map)

        # criterion放在最后一个设备上
        self.criterion_device = device_map['fc']
        self.criterion = nn.CrossEntropyLoss()

        # 优化器需要包含所有参数
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
            'gpu_memory_per_device': []  # 记录每个GPU的内存使用
        }

        self.start_epoch = 0
        self.best_acc = 0

        # 保存路径准备
        os.makedirs(config.save_dir, exist_ok=True)

        # 恢复训练
        if config.resume:
            self.load_checkpoint(config.resume)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            self.monitor.start_timer()

            # inputs -> first module所在设备
            first_dev = list(self.model.device_map.values())[0]
            last_dev = self.criterion_device
            inputs = inputs.to(first_dev, non_blocking=True)
            targets = targets.to(last_dev, non_blocking=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time = self.monitor.end_timer('train_batch_time')

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # predicted on last_dev; move to cpu for counting or compare on same device
            correct += predicted.eq(targets).sum().item()

            if batch_idx % self.config.log_interval == 0:
                # 记录所有GPU的内存使用
                gpu_mem_info = []
                for gpu_id in self.gpu_ids:
                    mem_allocated = torch.cuda.memory_allocated(f'cuda:{gpu_id}') / 1024**3
                    gpu_mem_info.append(f'GPU{gpu_id}:{mem_allocated:.2f}GB')
                gpu_mem_str = ', '.join(gpu_mem_info)
                print(f'Epoch: {epoch} [{batch_idx}/{len(self.trainloader)}] '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% '
                      f'| Time: {batch_time:.4f}s | GPU Mem: [{gpu_mem_str}]')

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(self.trainloader)
        epoch_acc = 100. * correct / total

        # 记录所有GPU的内存使用（统一格式）
        gpu_mem_per_device = []
        for gpu_id in self.gpu_ids:
            gpu_mem_per_device.append({
                'device_id': gpu_id,
                'allocated': torch.cuda.memory_allocated(f'cuda:{gpu_id}') / 1024**3,
                'reserved': torch.cuda.memory_reserved(f'cuda:{gpu_id}') / 1024**3
            })

        return epoch_loss, epoch_acc, epoch_time, gpu_mem_per_device

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        last_dev = self.criterion_device

        with torch.no_grad():
            for inputs, targets in self.testloader:
                first_dev = list(self.model.device_map.values())[0]
                inputs = inputs.to(first_dev, non_blocking=True)
                targets = targets.to(last_dev, non_blocking=True)

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
        print(f"\n开始训练 {self.config.epochs} 个epochs...")
        print(f"每个GPU的batch size: {self.config.batch_size}")
        print(f"学习率: {self.config.lr}")

        total_start = time.time()

        for epoch in range(self.start_epoch, self.config.epochs):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch+1}/{self.config.epochs}')
            print(f'{"="*60}')

            train_loss, train_acc, epoch_time, gpu_mem_per_device = self.train_epoch(epoch)
            test_loss, test_acc = self.evaluate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['epoch_time'].append(epoch_time)
            self.history['gpu_memory_per_device'].append(gpu_mem_per_device)

            print(f'\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
            print(f'学习率: {current_lr:.6f}, 时间: {epoch_time:.2f}s')
            # 打印每个GPU的内存使用
            print('GPU内存使用:')
            for mem_info in gpu_mem_per_device:
                print(f'  GPU {mem_info["device_id"]} - 已分配: {mem_info["allocated"]:.2f}GB, 已保留: {mem_info["reserved"]:.2f}GB')

            if test_acc > self.best_acc:
                print(f'最佳准确率更新: {self.best_acc:.2f}% -> {test_acc:.2f}%')
                self.best_acc = test_acc
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

        total_time = time.time() - total_start

        print(f'\n{"="*60}')
        print('训练完成!')
        print(f'{"="*60}')
        print(f'总时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)')
        print(f'平均每epoch: {total_time/self.config.epochs:.2f}秒')
        print(f'最佳测试准确率: {self.best_acc:.2f}%')

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
            path = os.path.join(self.config.save_dir, 'best_model_mp.pth')
            torch.save(state, path)
            print(f'保存最佳模型到: {path}')
        elif is_final:
            # 支持自定义最终checkpoint路径
            if self.config.final_checkpoint_path:
                path = self.config.final_checkpoint_path
            else:
                path = os.path.join(self.config.save_dir, 'final_model_mp.pth')
            torch.save(state, path)
            print(f'保存最终模型到: {path}')
        else:
            path = os.path.join(self.config.save_dir, f'checkpoint_mp_epoch_{epoch+1}.pth')
            torch.save(state, path)
            print(f'保存checkpoint到: {path}')

    def load_checkpoint(self, path):
        print(f'从 {path} 加载checkpoint...')
        # 把checkpoint加载到cpu，然后将参数分配到对应设备
        checkpoint = torch.load(path, map_location='cpu')

        # 加载模型权重（会根据模块所在device自动拷贝）
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器和scheduler状态
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 将optimizer中的state张量移动到对应参数的device
            for p, state in self.optimizer.state.items():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(p.device)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            print('警告: 无法完整恢复optimizer/scheduler状态')

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint.get('best_acc', 0)
        self.history = checkpoint.get('history', self.history)

        print(f'恢复训练从epoch {self.start_epoch}, 最佳准确率: {self.best_acc:.2f}%')
