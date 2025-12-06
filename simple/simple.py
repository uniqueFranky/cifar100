import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import psutil
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 性能监控类
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
        
    def start_timer(self):
        self.start_time = time.time()
        
    def end_timer(self, metric_name):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.metrics[metric_name].append(elapsed)
            return elapsed
        return 0
    
    def record_memory(self, metric_name):
        # GPU内存使用
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.metrics[f'{metric_name}_gpu_memory'].append(gpu_memory)
        
        # CPU内存使用
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        self.metrics[f'{metric_name}_cpu_memory'].append(cpu_memory)
    
    def get_average(self, metric_name):
        if metric_name in self.metrics:
            return np.mean(self.metrics[metric_name])
        return 0

# 简单的ResNet模型
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def prepare_data(batch_size=128):
    """准备CIFAR-100数据集"""
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 加载数据集
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_epoch(model, trainloader, criterion, optimizer, device, monitor):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 记录批次开始时间和内存
        monitor.start_timer()
        monitor.record_memory('train_batch_start')
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        forward_start = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        forward_time = time.time() - forward_start
        monitor.metrics['forward_time'].append(forward_time)
        
        # 反向传播
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        monitor.metrics['backward_time'].append(backward_time)
        
        # 记录批次结束时间和内存
        batch_time = monitor.end_timer('train_batch_time')
        monitor.record_memory('train_batch_end')
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%, '
                  f'Time: {batch_time:.4f}s')
    
    epoch_time = time.time() - epoch_start_time
    epoch_acc = 100. * correct / total
    epoch_loss = running_loss / len(trainloader)
    
    return epoch_loss, epoch_acc, epoch_time

def evaluate(model, testloader, criterion, device, monitor):
    """评估模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # 记录推理开始时间和内存
            monitor.start_timer()
            monitor.record_memory('eval_batch_start')
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 推理
            inference_start = time.time()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            inference_time = time.time() - inference_start
            monitor.metrics['inference_time'].append(inference_time)
            
            # 记录批次结束时间和内存
            batch_time = monitor.end_timer('eval_batch_time')
            monitor.record_memory('eval_batch_end')
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    eval_time = time.time() - eval_start_time
    test_acc = 100. * correct / total
    test_loss = test_loss / len(testloader)
    
    return test_loss, test_acc, eval_time

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化性能监控器
    monitor = PerformanceMonitor()
    
    # 准备数据
    print("准备数据...")
    trainloader, testloader = prepare_data(batch_size=128)
    
    # 创建模型
    print("创建模型...")
    model = ResNet(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # 训练参数
    num_epochs = 100
    
    # 记录训练历史
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    epoch_times = []
    
    print("开始训练...")
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 训练
        train_loss, train_acc, train_time = train_epoch(
            model, trainloader, criterion, optimizer, device, monitor)
        
        # 评估
        test_loss, test_acc, eval_time = evaluate(
            model, testloader, criterion, device, monitor)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        epoch_times.append(train_time + eval_time)
        
        print(f'训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Time: {train_time:.2f}s')
        print(f'测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, Time: {eval_time:.2f}s')
    
    total_time = time.time() - total_start_time
    
    # 打印性能统计
    print("\n" + "="*60)
    print("性能统计报告")
    print("="*60)
    
    print(f"总训练时间: {total_time:.2f}秒")
    print(f"平均每个epoch时间: {np.mean(epoch_times):.2f}秒")
    
    print(f"\n训练阶段:")
    print(f"  平均批次时间: {monitor.get_average('train_batch_time'):.4f}秒")
    print(f"  平均前向传播时间: {monitor.get_average('forward_time'):.4f}秒")
    print(f"  平均反向传播时间: {monitor.get_average('backward_time'):.4f}秒")
    
    print(f"\n推理阶段:")
    print(f"  平均批次时间: {monitor.get_average('eval_batch_time'):.4f}秒")
    print(f"  平均推理时间: {monitor.get_average('inference_time'):.4f}秒")
    
    if torch.cuda.is_available():
        print(f"\nGPU内存使用:")
        print(f"  训练时平均: {monitor.get_average('train_batch_end_gpu_memory'):.2f}GB")
        print(f"  推理时平均: {monitor.get_average('eval_batch_end_gpu_memory'):.2f}GB")
        print(f"  峰值GPU内存: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
    
    print(f"\nCPU内存使用:")
    print(f"  训练时平均: {monitor.get_average('train_batch_end_cpu_memory'):.2f}GB")
    print(f"  推理时平均: {monitor.get_average('eval_batch_end_cpu_memory'):.2f}GB")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和测试损失')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('训练和测试准确率')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epoch_times)
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('每个Epoch的训练时间')
    
    plt.tight_layout()
    plt.savefig('training_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'performance_metrics': dict(monitor.metrics)
    }, 'cifar100_model_checkpoint.pth')
    
    print(f"\n模型和性能数据已保存到 'cifar100_model_checkpoint.pth'")

if __name__ == '__main__':
    main()
