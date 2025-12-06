import torch
import torchvision
import torchvision.transforms as transforms

def verify_cifar100_stats():
    """验证CIFAR-100的统计值"""
    
    # 加载原始数据（不做标准化）
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                           download=True, transform=transform)
    
    # 计算所有训练数据的统计值
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    # 计算均值
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    
    # 计算标准差
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        std += ((data - mean.unsqueeze(1))**2).sum([0, 2])
    
    std = torch.sqrt(std / (total_samples * 32 * 32))
    
    print("实际计算的CIFAR-100统计值:")
    print(f"均值: ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    print(f"标准差: ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
    
    print("\n常用的'标准'值:")
    print("均值: (0.5071, 0.4867, 0.4408)")
    print("标准差: (0.2675, 0.2565, 0.2761)")
    
    return mean, std

if __name__ == "__main__":
    verify_cifar100_stats()
