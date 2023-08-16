import numpy as np
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, model_dim, head_num):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.atten_dim = model_dim // head_num

        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=False)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, L, D = x.shape

        Q = self.W_Q(x).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        atten = torch.mean(self.softmax(scores), dim=1)

        return atten
