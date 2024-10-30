import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger
from utils.freqfusion import FreqFusion

logger = get_logger()


class MLP(nn.Module):
    """
    Linear Embedding:
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(
        self,
        in_channels=[64, 128, 320, 512],
        num_classes=40,
        dropout_ratio=0.1,
        norm_layer=nn.BatchNorm2d,
        embed_dim=768,
        align_corners=False,
    ):

        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.ff_c43 = FreqFusion(lr_channels=embedding_dim, hr_channels=embedding_dim)
        self.ff_c432 = FreqFusion(lr_channels=embedding_dim * 2, hr_channels=embedding_dim)
        self.ff_c4321 = FreqFusion(lr_channels=embedding_dim * 3, hr_channels=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        # print(c1.shape)
        # print(c2.shape)
        # print(c3.shape)
        # print(c4.shape)
        # (N, 32, 120, 160), (N, 64, 60, 80), (N, 128, 30, 40), (N, 256, 15, 20)
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = (self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]))
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=self.align_corners)

        _c3 = (self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]))
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=self.align_corners)

        _c2 = (self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]))
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=self.align_corners)

        _c1 = (self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3]))

        # _, _c3, _c4_up = self.ff_c43(hr_feat=_c3, lr_feat=_c4)
        # _, _c2, _c34_up = self.ff_c432(hr_feat=_c2, lr_feat=torch.cat([_c3, _c4_up], dim=1))
        # _, _c1, _c234_up = self.ff_c4321(hr_feat=_c1, lr_feat=torch.cat([_c2, _c34_up], dim=1))
        # _c = self.linear_fuse(torch.cat([_c1, _c234_up], dim=1)) # channel=4c, 1/4 img size
        

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # print("max:", torch.max(_c))
        # print("min:", torch.min(_c))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print(torch.max(x))

        return x
