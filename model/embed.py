import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, model_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, model_dim):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=model_dim,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, input_dim, model_dim):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(input_dim=input_dim, model_dim=model_dim)
        self.position_embedding = PositionalEmbedding(model_dim=model_dim)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return x
