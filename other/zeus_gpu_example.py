import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from zeus.monitor import ZeusMonitor
from zeus.monitor.power import PowerMonitor

class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(ThreeLayerMLP, self).__init__()
        # 第一层 (输入层 -> 隐藏层1)
        self.layer1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        
        # 第二层 (隐藏层1 -> 隐藏层2)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        
        # 第三层 (隐藏层2 -> 输出层)
        self.layer3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x


if __name__ == "__main__":
    # 指定 GPU，避免后续方法中未指定时导致默认行为不一
    gpu_indices = [torch.cuda.current_device()]
    
    # 1. ZeusMonitor 用于记录代码块级别的能耗总量 (Energy) 和时间
    monitor = ZeusMonitor(gpu_indices=gpu_indices)
    
    # 2. PowerMonitor 开启后台进程，持续拉取并记录瞬时功率
    power_monitor = PowerMonitor(
        gpu_indices=gpu_indices,
        power_domains=["device_instant"] # 显式指定我们要测的是 GPU 瞬时功率
    )
    
    # ==========================================
    # 1. 准备模拟数据 (Dummy Data)
    # ==========================================
    num_samples = 8192
    input_size = 20
    output_size = 2
    batch_size = 128

    X = torch.randn(num_samples, input_size).cuda() # 如果想测 GPU 功耗，数据建议搬到 GPU
    Y = torch.randint(0, output_size, (num_samples,)).cuda()

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ThreeLayerMLP(input_size=input_size, hidden1=64, hidden2=32, output_size=output_size).cuda()

    # ==========================================
    # 2. 设置损失函数和优化器
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 3. 训练循环 (Epoch Loop & Step Loop)
    # ==========================================
    num_epochs = 5
    
    # 记录整体训练阶段的时间戳，方便去 PowerMonitor 里查这期间的数据
    train_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        steps = []

        monitor.begin_window("epoch")
        for step, (batch_x, batch_y) in enumerate(dataloader):
            monitor.begin_window("step")
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            result = monitor.end_window("step")
            steps.append(result)
            epoch_loss += loss.item()
        
        mes = monitor.end_window("epoch")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: avg loss:  {avg_epoch_loss:.4f}, consumed {mes.time:.4f} s and {mes.total_energy:.4f} J.")
       
        avg_time = sum(map(lambda m: m.time, steps)) / len(steps)
        avg_energy = sum(map(lambda m: m.total_energy, steps)) / len(steps)
        print(f"One step took {avg_time:.4f} s and {avg_energy:.4f} J on average. \n")

    # 记录结束时间戳
    train_end_time = time.time()

    # ==========================================
    # 4. 获取并计算最大瞬时功率
    # ==========================================
    # 注意：不要传 gpu_index，get_power_timeline 默认返回所有被监控 GPU 的字典
    # 返回格式为 dict[int, list] (GPU_ID -> Samples)
    timeline_dict = power_monitor.get_power_timeline(
        power_domain="device_instant",
        start_time=train_start_time,
        end_time=train_end_time
    )
    
    # 提取我们指定 GPU 的监控列表
    gpu_id = gpu_indices[0]
    samples = timeline_dict.get(gpu_id, [])
    
    if samples:
        max_power_w = 0.0
        
        for s in samples:
            # 兼容不同 Zeus 版本的返回值结构
            if hasattr(s, "power_mw"):
                # 如果是较新版的 PowerSample 数据类
                power_w = s.power_mw / 1000.0
            elif isinstance(s, tuple) or isinstance(s, list):
                # 如果是旧版返回的 tuple (timestamp, power_in_watts)
                power_w = s[1]
            else:
                # 兜底：如果是直接返回的数值
                power_w = float(s)
                
            if power_w > max_power_w:
                max_power_w = power_w
                
        print(f"====== Power Report ======")
        print(f"Total samples collected: {len(samples)}")
        print(f"Maximum Instantaneous Power: {max_power_w:.2f} W")
        print(samples)
    else:
        print("No power samples collected. (Training might have completed too fast for the polling rate)")