import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

# from .base_model import BaseModel
from tem_base import BaseModel


from tem_base import BaseModel
import os
import torch
from dataset import UniversalDataset
from torch.utils.data import DataLoader, TensorDataset


class LSTM(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, bias=True, device=None):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.bias = bias
        self.device = device
        self.name = 'LSTM'

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bias=bias)

        # 定义输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 从 LSTM 层获取输出
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 取序列的最后一个时间点的输出
        out = out[:, -1, :]

        # 通过线性层获得最终的输出
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

    def initialize(self):
        pass
        # # 自定义初始化
        # for name, param in self.lstm.named_parameters():
        #     if 'weight_ih' in name:
        #         torch.nn.init.xavier_uniform_(param)
        #     elif 'weight_hh' in name:
        #         torch.nn.init.orthogonal_(param)
        #     elif 'bias' in name:
        #         param.fill_(0.01)  # 给偏置设置一个小的非零值


dataset = UniversalDataset()
dataset.load_toy_dataset()

# 定义模型参数
input_dim = 4
hidden_dim = 64
output_dim = 47 * 47  # 2209，因为需要预测47x47的矩阵
# Initialize GRUModel
model = LSTM(input_dim=4, hidden_dim=64, output_dim=2209)
model.to(device='cuda' if torch.cuda.is_available() else 'cpu')

import torch
from torch_geometric.data import Data

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
