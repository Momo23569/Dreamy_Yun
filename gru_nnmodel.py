import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch

from tem_base import BaseModel
import os
import torch
from dataset import UniversalDataset
from torch.utils.data import DataLoader, TensorDataset


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, num_features]
        out, _ = self.gru(x)  # GRU输出
        out = self.fc(out[:, -1, :])  # 只取序列中的最后一个时间点的输出
        return out.reshape(-1, 47, 47)  # 调整输出形状以匹配 y 的形状

dataset = UniversalDataset()
dataset.load_toy_dataset()


# 假设 dataset.x 和 dataset.y 已经是 torch.Tensor 类型
x_tensor = torch.tensor(dataset.x, dtype=torch.float32)
y_tensor = torch.tensor(dataset.y, dtype=torch.float32)
y_tensor = y_tensor.squeeze(-1)  # 移除最后一个维度，因为它是1

mean_x = x_tensor.mean(dim=0, keepdim=True)
std_x = x_tensor.std(dim=0, keepdim=True)
x_tensor = (x_tensor - mean_x) / std_x

# 创建 TensorDataset 和 DataLoader
dataset_tensor = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset_tensor, batch_size=10, shuffle=True)

# 定义模型参数
input_dim = 4
hidden_dim = 64
output_dim = 47 * 47  # 2209，因为需要预测47x47的矩阵


# 实例化模型
model = GRUNet(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 定义训练函数
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}')


# 调用训练函数
train(model, dataloader, criterion, optimizer, epochs=10)

