import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# テンソルの定義の仕方
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(a)

# テンソルの形状
print(a.shape)

# 形状変換
b = a.view(1, 4)
c = a.view(4, 1)
print(b)
print(c)
print('test')