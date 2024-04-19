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

class GRUModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.name = 'GRU'
        self.data = None

        # 定义 GRU 层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)

        # 定义输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 初始化超参数
        self.lr = 0.01
        self.weight_decay = 1e-4

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 从 GRU 层获取输出
        out, _ = self.gru(x, h0)

        # 取序列的最后一个时间点的输出
        out = out[:, -1, :]

        # 通过线性层获得最终的输出
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    def initialize(self):
            pass


dataset = UniversalDataset()
dataset.load_toy_dataset()




# 定义模型参数
input_dim = 4
hidden_dim = 64
output_dim = 47 * 47  # 2209，因为需要预测47x47的矩阵
# Initialize GRUModel
model = GRUModel(input_dim=4, hidden_dim=64, output_dim=2209)
model.to(device='cuda' if torch.cuda.is_available() else 'cpu')

# Since BaseModel expects the data in a specific format, ensure your dataset class provides it
# Assuming 'data' is an object that includes .to(device), .train_mask, .val_mask, etc.

import torch
from torch_geometric.data import Data

# 假设 dataset 是以某种方式加载的
# 这里我们直接使用它的属性
x = torch.tensor(dataset.x, dtype=torch.float32) if not isinstance(dataset.x, torch.Tensor) else dataset.x.float()
y = torch.tensor(dataset.y, dtype=torch.float32) if not isinstance(dataset.y, torch.Tensor) else dataset.y.float()
y = y.squeeze(-1)  # 移除最后一个维度，因为它是1
edge_index = torch.tensor(dataset.edge_index, dtype=torch.long) if not isinstance(dataset.edge_index, torch.Tensor) else dataset.edge_index.long()
# edge_attr = torch.tensor(dataset.edge_attr, dtype=torch.float32) if not isinstance(dataset.edge_attr, torch.Tensor) else dataset.edge_attr.float()
states = torch.tensor(dataset.states, dtype=torch.float32) if not isinstance(dataset.states, torch.Tensor) else dataset.states.float()

# Now, create a Data object to encapsulate all these tensors.
data = Data(x=x, y=y, edge_index=edge_index, states=states)



model.data = data
# Train the model
model.fit(data=data, train_iters=1000, initialize=True, verbose=True, patience=100)
