import torch
from torch import nn
from torch import functional


# FFN

class FFN(nn.Module):
    def __init__ (self, vector_size):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(vector_size, vector_size)
        self.w2 = nn.Linear(vector_size, vector_size)
        self.w3 = nn.Linear(vector_size, vector_size)

    def forward(self,x):
        x = functional.F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * norm

class ResidualConnection(nn.Module):
    def __init__(self, module):
        super(ResidualConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class ROPE(nn.Module):
    def __init__ (self):
        super(ROPE,self).__init__()


class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,kv_head):
        super(MultiHeadAttention, self).__init__()
        pass
    def forward(self, x):
        # Placeholder for multi-head attention logic
        return x
