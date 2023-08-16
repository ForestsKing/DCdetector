import torch
from einops import rearrange, repeat, reduce
from torch import nn

from model.atten import Attention
from model.embed import DataEmbedding


class DCdetector(nn.Module):
    def __init__(self, args):
        super(DCdetector, self).__init__()
        self.args = args

        self.embedding_in_patch = nn.ModuleList()
        self.embedding_patch_wise = nn.ModuleList()
        for patch_index, patch_size in enumerate(self.args.patch_size):
            patch_num = self.args.window_size // patch_size
            self.embedding_in_patch.append(DataEmbedding(patch_num, self.args.model_dim))
            self.embedding_patch_wise.append(DataEmbedding(patch_size, self.args.model_dim))

        self.encoder = nn.ModuleList()
        for _ in range(self.args.encoder_layer):
            self.encoder.append(Attention(self.args.model_dim, self.args.head_num))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, L, C = x.shape

        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True, unbiased=False) + 1e-8
        x = (x - mean) / std

        pos_list, neg_list = [], []
        for patch_index, patch_size in enumerate(self.args.patch_size):
            patch_num = self.args.window_size // patch_size

            x_pos = rearrange(x, 'B (N P) C -> (B C) P N', P=patch_size)
            x_neg = rearrange(x, 'B (N P) C -> (B C) N P', P=patch_size)

            embed_pos = self.embedding_in_patch[patch_index](x_pos)
            embed_neg = self.embedding_patch_wise[patch_index](x_neg)

            atten_pos, atten_neg = [], []
            for layer in self.encoder:
                atten_pos.append(layer(embed_pos).unsqueeze(0))
                atten_neg.append(layer(embed_neg).unsqueeze(0))
            atten_pos = torch.mean(torch.concat(atten_pos, dim=0), dim=0)
            atten_neg = torch.mean(torch.concat(atten_neg, dim=0), dim=0)

            atten_pos = repeat(atten_pos, 'BC P1 P2 -> BC (N1 P1) (N2 P2)', N1=patch_num, N2=patch_num)
            atten_neg = repeat(atten_neg, 'BC N1 N2 -> BC (N1 P1) (N2 P2)', P1=patch_size, P2=patch_size)

            atten_pos = reduce(atten_pos, '(B C) L1 L2-> B L1 L2', 'mean', B=B)
            atten_neg = reduce(atten_neg, '(B C) L1 L2-> B L1 L2', 'mean', B=B)

            pos_list.append(atten_pos.unsqueeze(0))
            neg_list.append(atten_neg.unsqueeze(0))

        pos = self.softmax(torch.sum(torch.concat(pos_list, dim=0), dim=0))
        neg = self.softmax(torch.sum(torch.concat(neg_list, dim=0), dim=0))

        return pos, neg
