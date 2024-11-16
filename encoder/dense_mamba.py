import sys

sys.path.append("/home/suguilin/MMFS/")
sys.path.append("/home/suguilin/zigma/")
import os
from torch import nn
import torch
from collections import OrderedDict
import math
from timm.models.layers import trunc_normal_
import torchvision.transforms as transforms
from mamba_ssm.modules.mamba_simple import Mamba
from utils.utils import *
from utils.modules import VSSBlock
# from encoder.cross_mamba import Fusion_Embed
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
import numpy as np


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class SimpleChannel(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans, 1)
        self.conv2 = nn.Conv2d(in_chans, out_chans*2, kernel_size=3, padding=1, stride=1, groups=out_chans)
        self.sg = SimpleGate()
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
                nn.Conv2d(2, 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 1, 1), 
                nn.Sigmoid()
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        short_x = x
        x = self.conv2(x * self.sigmoid(self.conv1(self.avg_pool(x)) + self.conv1(self.max_pool(x))))
        x = self.sg(x)
        x_mean_out = torch.mean(short_x, dim=1, keepdim=True)
        x_max_out, _ = torch.max(short_x, dim=1, keepdim=True)
        sp = self.mlp(torch.cat([x_mean_out, x_max_out], dim=1))
        # x = x * sp + x
        x = x * (1 + sp)
        return x


class MSDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.
    """
    def __init__(self,
                 embed_dims,
                 output_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                 act=nn.SiLU(),
                ):
        super(MSDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, 
            dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, 
            dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, 
            dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=output_dims,
            kernel_size=1
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        short_x = x
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)

        # refer https://github.com/hongyuanyu/SPAN/blob/main/basicsr/archs/span_arch.py line188
        sim_att = torch.sigmoid(x) - 0.5
        x = (short_x + x) * sim_att
        v = self.PW_conv(x)
        return v


class SAM(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SAM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(4, 4 * reduction, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 2, kernel_size), 
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True) #B  1  H  W
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)  #B  1  H  W
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True) #B  1  H  W
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)  #B  1  H  W                
        x_cat = torch.cat((x1_mean_out, x1_max_out, x2_mean_out, x2_max_out), dim=1) # B 4 H W
        spatial_weights = self.mlp(x_cat).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        sp_x1 = spatial_weights[0]  # B 1 H W
        sp_x2 = spatial_weights[1]  # B 1 H W
        return sp_x1, sp_x2


class SAB(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SAB, self).__init__()
        self.sab = SAM(kernel_size=kernel_size, reduction=reduction)
    def forward(self, x1, x2):
        w1, w2 = self.sab(x1, x2)
        x1 = w1 * x1
        x2 = w2 * x2
        return x1, x2


class CAM(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        out = x * self.sigmoid(attn)
        return out


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            CAM(num_feat, squeeze_factor),
        )
    def forward(self, x):
        x = self.cab(x)
        return x

class TokenSwapMamba(nn.Module):
    def __init__(self, dim, init_ratio=0.5):
        super(TokenSwapMamba, self).__init__()
        self.encoder_x1 = Mamba(dim, bimamba_type=None)
        self.encoder_x2 = Mamba(dim, bimamba_type=None)
        self.norm1 = nn.LayerNorm(dim) # LayerNorm(dim, 'with_bias')
        self.norm2 = nn.LayerNorm(dim) # LayerNorm(dim, 'with_bias')
        self.ratio = nn.Parameter(torch.tensor(init_ratio))
    def forward(self, x):
        x1, x2 = x
        x1_short = x1
        x2_short = x2
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        B, N, C = x1.shape
        ratio = torch.sigmoid(self.ratio)
        num_chans = int(C * ratio)
        exchange_indices = torch.arange(num_chans)
        retain_indices = torch.arange(num_chans, C)

        x1 = torch.cat([x2[:, :, exchange_indices], x1[:, :, retain_indices]], dim=2)
        x2 = torch.cat([x1[:, :, exchange_indices], x2[:, :, retain_indices]], dim=2)
        x1 = self.encoder_x1(x1) + x1_short
        x2 = self.encoder_x2(x2) + x2_short
        x = [x1, x2]
        return x

class CIM(nn.Module):
    def __init__(self, dim, compress_ratio=3, squeeze_factor=16, kernel_size=1, reduction=4):
        super(CIM, self).__init__()
        # self.channel_swap = nn.Sequential(
        #     TokenSwapMamba(dim=dim),
        #     TokenSwapMamba(dim=dim),
        # )
        self.cab1 = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.cab2 = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.sab = SAB(kernel_size=kernel_size, reduction=reduction)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2 // reduction, dim),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        x1_flat = x1.flatten(2).transpose(1, 2)  ##B H*W C
        x2_flat = x2.flatten(2).transpose(1, 2)  ##B H*W C
        # x1_flat, x2_flat = self.channel_swap([x1_flat, x2_flat])
        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=2))  ##B H*W C
        gated_weight = gated_weight.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  ## B C H W

        ca1 = self.cab1(x1)  ## B C H W
        ca2 = self.cab2(x2)  ## B C H W
        sp1, sp2 = self.sab(x1, x2)  ## B C H W
        attv_x1, attv_x2 = ca1 + sp1, ca2 + sp2 ## B C H W

        out_x1 = x1 + (1 - gated_weight) * attv_x2  # B C H W
        out_x2 = x2 + gated_weight * attv_x1  # B C H W
        return out_x1, out_x2


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        B, H, W, C = x1.shape
        x1 = x1.view(B, -1, C)
        x2 = x2.view(B, -1, C)
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((B, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        out_x1 = out_x1.permute(0, 2, 1).view(B, C, H, W)
        out_x2 = out_x2.permute(0, 2, 1).view(B, C, H, W)
        return out_x1, out_x2


class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x1, x2):
        for module in self.modules_list:
            x1, x2 = module(x1, x2)  # two input, two output, cycle
        return x1, x2


class DenseMambaBlock(nn.Module):
    def __init__(
        self,
        id=0,   
        depths=[1, 1, 1], 
        dims=[96, 96, 96, 96], 
        # =========================
        d_state=16, 
        # =========================
        drop_rate=0., 
        drop_path_rate=0.1, 
        norm_layer=nn.LayerNorm,
        **kwargs,
        ):
        super().__init__()
        self.num_layers = len(depths)
        self.id = id
        if isinstance(dims, int):
            dims = [int(dims) for i_layer in range(self.num_layers)]
        self.dims = dims
        self.embed_dim = dims[0]
        self.d_state = d_state if d_state is not None else math.ceil(dims[0] / 6)
        # [start, end, steps]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.apply(self._init_weights)
        self.vssm, self.conv = self._create_modules(depths, dims, dpr, norm_layer, d_state, drop_rate)

    def _create_modules(self, depths, dims, dpr, norm_layer, d_state, drop_rate):
        vssm = nn.ModuleList(
            self._make_layer(
                dim=dims[i],
                depth=depths[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                d_state=d_state,
                drop_rate=drop_rate,
            ) for i in range(self.num_layers)
        )
        conv = nn.ModuleList(
            nn.Sequential(
                Permute(0, 3, 1, 2),
                # nn.Conv2d(dims[i] * (i + 2), dims[i], 3, 1, 1, padding_mode="reflect"), 
                # nn.Conv2d(dims[i] * (i + 2), dims[i], 1), 
                (nn.Conv2d(dims[i] * (i + 2), dims[i], 1) 
                if i != self.num_layers - 1 else SimpleChannel(dims[i] * (i + 2), dims[i])),
                nn.ReLU(),
                # SimpleChannel(dims[i] * (i + 2), dims[i]),
                # MSDWConv(embed_dims=dims[i] * (i + 2), output_dims=dims[i]),
                Permute(0, 2, 3, 1),
            ) for i in range(self.num_layers)
        )
        return vssm, conv

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    

    @staticmethod
    def _make_layer(
        dim=32, 
        depth=1,
        drop_path=[0.1, 0.1], 
        norm_layer=nn.LayerNorm,
        d_state=16,
        drop_rate=0.0,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(
                VSSBlock(
                    hidden_dim=dim, 
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    d_state=d_state,
                    drop=drop_rate,
                    **kwargs,
                )
            )
        return CustomSequential(*blocks)#nn.Sequential(*blocks)
    
    def forward(self, modal):
        outs = []

        outs.append(modal)
        loss_aux = 0
        for i in range(self.num_layers):
            modal, loss_aux = self.vssm[i](modal, loss_aux)
            modal = self.conv[i](torch.cat([modal, outs[-1]], dim=3))
            outs.append(torch.cat([modal, outs[-1]], dim=3))
        # 这里的残差连接不知道影响不影响特征提取效果
        modal = modal + outs[0]
        return modal#, loss_aux
        # if self.id > 0 and self.id < 3:
        #     return modal, outs, loss_aux
        # else:
        #     return modal, loss_aux

'''
class DenseMamba(nn.Module):
    def __init__(
            self, 
            patch_size=4, 
            in_chans=1,  
            depths=[1, 1, 1], 
            dims=[32, 32, 32], 
            # =========================
            d_state=16, 
            # =========================
            drop_rate=0., 
            drop_path_rate=0.1, 
            patch_norm=True, 
            norm_layer=nn.LayerNorm,
            **kwargs,
        ):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.dims = dims
        self.embed_dim = dims[0]
        self.d_state = d_state if d_state is not None else math.ceil(dims[0] / 6)
        self.patch_embed_rgb = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans*3, embed_dim=self.embed_dim, 
            norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_ir = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim, 
            norm_layer=norm_layer if patch_norm else None)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        # [start, end, steps]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.vssm_rgb, self.conv_rgb = self._create_modules(depths, dims, dpr, norm_layer, d_state, drop_rate)
        self.vssm_ir, self.conv_ir = self._create_modules(depths, dims, dpr, norm_layer, d_state, drop_rate)
        self.apply(self._init_weights)

    def _create_modules(self, depths, dims, dpr, norm_layer, d_state, drop_rate):
        vssm = nn.ModuleList(
            self._make_layer(
                dim=dims[i],
                depth=depths[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                d_state=d_state,
                drop_rate=drop_rate,
            ) for i in range(self.num_layers)
        )
        conv = nn.ModuleList(
            nn.Sequential(
                Permute(0, 3, 1, 2),
                # nn.Conv2d(dims[i] * (i + 2), dims[i], 3, 1, 1, padding_mode="reflect"), 
                # nn.ReLU(),
                SimpleChannel(dims[i] * (i + 2), dims[i]),
                Permute(0, 2, 3, 1),
            ) for i in range(self.num_layers)
        )
        return vssm, conv


    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    

    @staticmethod
    def _make_layer(
        dim=32, 
        depth=2,
        drop_path=[0.1, 0.1], 
        norm_layer=nn.LayerNorm,
        d_state=16,
        drop_rate=0.0,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                d_state=d_state,
                drop=drop_rate,
                **kwargs,
            ))
        return nn.Sequential(*blocks)
    
    def forward(self, rgb, ir):
        rgb = self.patch_embed_rgb(rgb)
        ir = self.patch_embed_ir(ir)
        # print(f"After patch embedding - RGB: {rgb.min().item()}, {rgb.max().item()}, IR: {ir.min().item()}, {ir.max().item()}")
        # print(rgb.shape)
        rgb = self.pos_drop(rgb)
        ir = self.pos_drop(ir)
        # print(f"After positional dropout - RGB: {rgb.min().item()}, {rgb.max().item()}, IR: {ir.min().item()}, {ir.max().item()}")

        outs_rgb = []
        outs_ir = []

        outs_rgb.append(rgb)
        outs_ir.append(ir)
        for i in range(self.num_layers):
            rgb = self.vssm_rgb[i](rgb)
            ir = self.vssm_ir[i](ir)
            # print(f"After VSSM layer {i} - RGB: {rgb.min().item()}, {rgb.max().item()}, IR: {ir.min().item()}, {ir.max().item()}")
            
            rgb = self.conv_rgb[i](torch.cat([rgb, outs_rgb[-1]], dim=3))
            ir = self.conv_ir[i](torch.cat([ir, outs_ir[-1]], dim=3))
            # print(f"After conv layer {i} - RGB: {rgb.min().item()}, {rgb.max().item()}, IR: {ir.min().item()}, {ir.max().item()}")
            outs_rgb.append(torch.cat([rgb, outs_rgb[-1]], dim=3))
            outs_ir.append(torch.cat([ir, outs_ir[-1]], dim=3))
        return outs_rgb[-1].permute(0, 3, 1, 2), rgb.permute(0, 3, 1, 2), outs_ir[-1].permute(0, 3, 1, 2), ir.permute(0, 3, 1, 2)
'''

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class DenseMamba(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=1,  
        dim=96, 
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        num_layers=4, 
        drop_rate=0., 
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = dim
        self.num_layers = num_layers
        self.patch_embed_rgb = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans*3, embed_dim=self.embed_dim, 
            norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_ir = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans*3, embed_dim=self.embed_dim, 
            norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.vssm_rgb = self._create_modules(dim=dim)
        self.vssm_ir = self._create_modules(dim=dim)
        self.fuse = nn.ModuleList(
            CIM(dim=dim * 2 ** i)
            for i in range(self.num_layers)
        )

    @staticmethod
    def _make_layer(dim=32, depths=[1, 1, 1], id=0):
        blocks = []
        blocks.append(Permute(0, 3, 1, 2))
        if id == 0:
            blocks.append(nn.Identity())
        else:
            blocks.append(
                OverlapPatchEmbed(patch_size=3, stride=2, in_chans=dim//2, embed_dim=dim)
            )
        blocks.append(Permute(0, 2, 3, 1))
        blocks.append(DenseMambaBlock(dims=dim, depths=depths, id=id))
        
        # blocks.append(
        #     nn.Conv2d(
        #         dim,
        #         dim*2,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #     ))
        return nn.Sequential(*blocks)
    
    def _create_modules(self, dim):
        vssm = nn.ModuleList(
            self._make_layer(dim=int(dim * 2 ** i), depths=[2, 2, 2], id = i) if i == 0 else self._make_layer(dim=int(dim * 2 ** i), id=i)
            for i in range(self.num_layers)
        )
        return vssm
    

    def forward(self, rgb, ir):
        rgb = self.patch_embed_rgb(rgb)
        ir = self.patch_embed_ir(ir)
        # print(rgb.shape)
        # print(ir.shape)
        rgb = self.pos_drop(rgb)
        ir = self.pos_drop(ir)
        outs_rgb = []
        outs_ir = []
        branch_rgb = []
        branch_ir = []
    
        # outs_rgb.append(rgb.permute(0, 3, 1, 2))
        # outs_ir.append(ir.permute(0, 3, 1, 2))
        loss_aux = 0
        for i in range(self.num_layers):
            # print("rgb:", rgb.shape)
            # print("ir:", ir.shape)

            if i > 0 and i < 3:
                rgb, bra_rgb, l_aux_rgb = self.vssm_rgb[i](rgb)
                ir, bra_ir, l_aux_ir = self.vssm_ir[i](ir)
                if i == 1:
                    branch_rgb.append(rgb.permute(0, 3, 1, 2))
                    branch_ir.append(ir.permute(0, 3, 1, 2))
                    branch_rgb.append(bra_rgb[-1].permute(0, 3, 1, 2))
                    branch_ir.append(bra_ir[-1].permute(0, 3, 1, 2))
                else:
                    branch_rgb.append(rgb.permute(0, 3, 1, 2))
                    branch_ir.append(ir.permute(0, 3, 1, 2))
                # rgb = self.vssm_rgb[i][3](rgb)
                # ir = self.vssm_ir[i][3](ir)
            else:
                rgb, l_aux_rgb = self.vssm_rgb[i](rgb)
                ir, l_aux_ir = self.vssm_ir[i](ir)
                # rgb = self.vssm_rgb[i][3](rgb)
                # ir = self.vssm_ir[i][3](ir)

            # outs_rgb.append(rgb.permute(0, 3, 1, 2))
            # outs_ir.append(ir.permute(0, 3, 1, 2))
            loss_aux += (l_aux_rgb + l_aux_ir)
            
            rgb, ir = self.fuse[i](rgb.permute(0, 3, 1, 2), ir.permute(0, 3, 1, 2))
            outs_rgb.append(rgb)
            outs_ir.append(ir)
            rgb = rgb.permute(0, 2, 3, 1)
            ir = ir.permute(0, 2, 3, 1)
        # print("rgb:", rgb.shape)
        # print("ir:", ir.shape)
        # print(loss_aux)
        return outs_rgb, outs_ir, branch_rgb, branch_ir, loss_aux


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.feature_maps = {
            'outs_rgb': [],
            'outs_ir': [],
            'branch_rgb': [],
            'branch_ir': []
        }

        # Register forward hooks for each submodule of the model
        for name, module in self.model.named_modules():
            module.register_forward_hook(self.save_feature(name))

    def save_feature(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                for idx, out in enumerate(output):
                    if 'vssm_rgb' in name and idx == 0:  # 假设第一个是特征图
                        self.feature_maps['outs_rgb'].append(out.detach().cpu())
                    elif 'vssm_ir' in name and idx == 0:  # 假设第一个是特征图
                        self.feature_maps['outs_ir'].append(out.detach().cpu())
                    elif 'branch_rgb' in name and idx == 0:
                        self.feature_maps['branch_rgb'].append(out.detach().cpu())
                    elif 'branch_ir' in name and idx == 0:
                        self.feature_maps['branch_ir'].append(out.detach().cpu())
            else:
                if 'vssm_rgb' in name:
                    self.feature_maps['outs_rgb'].append(output.detach().cpu())
                elif 'vssm_ir' in name:
                    self.feature_maps['outs_ir'].append(output.detach().cpu())
                elif 'branch_rgb' in name:
                    self.feature_maps['branch_rgb'].append(output.detach().cpu())
                elif 'branch_ir' in name:
                    self.feature_maps['branch_ir'].append(output.detach().cpu())
        return hook

    def forward(self, rgb, ir):
        outs_rgb, outs_ir, branch_rgb, branch_ir, loss_aux = self.model(rgb, ir)
        return outs_rgb, outs_ir, branch_rgb, branch_ir, loss_aux
    


# def visualize_and_save_feature_maps(feature_maps, save_dir):
#     fig = plt.figure(figsize=(16, 8))
#     gs = gridspec.GridSpec(2, 7)

#     # 显示原始图像
#     ax = fig.add_subplot(gs[0, 0])
#     ax.imshow(np.random.rand(128, 192), cmap='gray')
#     ax.set_title('A young boy chases\nducks through the leaves.')
#     ax.axis('off')

#     # 可视化特征图
#     for i, (name, maps) in enumerate(feature_maps.items()):
#         for j, feature_map in enumerate(maps):
#             ax = fig.add_subplot(gs[i, j + 1])
#             ax.imshow(feature_map[0].detach().cpu().numpy(), cmap='viridis')
#             ax.set_title(f"{name} [{j}]")
#             ax.axis('off')

#     plt.tight_layout()

#     # 保存可视化结果
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, 'feature_maps.png'))
#     plt.close(fig)


def visualize_and_save_feature_maps(feature_maps, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 8))
    num_rows = len(feature_maps)  # 根据特征图数量动态调整行数
    num_cols = max(len(maps) for maps in feature_maps.values())  # 动态获取最大列数
    gs = gridspec.GridSpec(num_rows, num_cols)  # +1 为原图留出空间

    # 显示原始图像
    # ax = fig.add_subplot(gs[0, 0])
    # ax.imshow(np.random.rand(128, 192), cmap='gray')
    # ax.set_title('A young boy chases\nducks through the leaves.')
    # ax.axis('off')

    # 可视化特征图
    for i, (name, maps) in enumerate(feature_maps.items()):
        for j, feature_map in enumerate(maps):
            ax = fig.add_subplot(gs[i, j + 1])
            feature_map_np = feature_map[0].detach().cpu().numpy()
            if feature_map_np.ndim == 3:  # 形状为 (C, H, W)
                feature_map_np = feature_map_np.mean(axis=0)  # 取平均值以获得 2D 图像
            
            ax.imshow(feature_map_np, cmap='viridis')
            ax.set_title(f"{name} [{j}]")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_maps.png'))
    plt.close(fig)
    

# def visualize_feature_maps(feature_maps, save_dir):
#     os.makedirs(save_dir, exist_ok=True)

#     # 定义特征名称
#     feature_names = ['outs_rgb', 'outs_ir', 'branch_rgb', 'branch_ir']

#     # 计算总行数和最大列数
#     total_plots = sum(len(feature_maps[name]) for name in feature_names)
#     max_cols = max(len(feature_maps[name]) for name in feature_names)

#     # 设置图像大小
#     fig = plt.figure(figsize=(8 * max_cols, 4 * len(feature_names)))  # 增大图像尺寸

#     # 初始化绘图行索引
#     row_index = 0

#     for name in feature_names:
#         for i in range(len(feature_maps[name])):
#             feature_map = feature_maps[name][i]
#             feature_map_np = feature_map[0].detach().cpu().numpy()  # 选择 B=0

#             # 归一化所有通道
#             normalized_maps = []
#             for channel in range(feature_map_np.shape[0]):
#                 ch_map = feature_map_np[channel]
#                 min_val = np.min(ch_map)
#                 max_val = np.max(ch_map)

#                 # 处理分母为零的情况
#                 if max_val - min_val == 0:
#                     normalized_map = np.zeros_like(ch_map)
#                 else:
#                     normalized_map = (ch_map - min_val) / (max_val - min_val)
                
#                 normalized_maps.append(normalized_map)

#             # 叠加所有归一化后的通道
#             heatmap = np.max(normalized_maps, axis=0)  # 选择最大值作为热力图

#             # 绘制热力图
#             ax = fig.add_subplot(len(feature_names), max_cols, row_index + 1)
#             ax.imshow(heatmap, cmap='hot')
#             ax.set_title(f"{name} Heatmap (index {i})", fontsize=12)
#             ax.axis('off')

#             row_index += 1

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'feature_maps_heatmap.png'))
#     plt.close(fig)

def visualize_feature_maps(feature_maps, feature_sizes, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 定义特征名称
    feature_names = ['outs_rgb', 'outs_ir', 'branch_rgb', 'branch_ir']

    # 计算总行数和最大列数
    total_plots = sum(len(feature_maps[name]) for name in feature_names)
    max_cols = max(len(feature_maps[name]) for name in feature_names)

    # 设置图像大小
    fig = plt.figure(figsize=(8 * max_cols, 4 * len(feature_names)))

    # 初始化绘图行索引
    row_index = 0

    for name in feature_names:
        for i in range(len(feature_maps[name])):
            feature_map = feature_maps[name][i]
            feature_map_size = feature_sizes[name][i]
            feature_map_np = feature_map[0].detach().cpu().numpy()  # 选择 B=0

            # 归一化所有通道
            normalized_maps = []
            for channel in range(feature_map_np.shape[0]):
                ch_map = feature_map_np[channel]
                min_val = np.min(ch_map)
                max_val = np.max(ch_map)

                # 处理分母为零的情况
                if max_val - min_val == 0:
                    normalized_map = np.zeros_like(ch_map)
                else:
                    normalized_map = (ch_map - min_val) / (max_val - min_val)
                
                normalized_maps.append(normalized_map)

            # 叠加所有归一化后的通道
            heatmap = np.max(normalized_maps, axis=0)  # 选择最大值作为热力图

            # 绘制热力图
            ax = fig.add_subplot(len(feature_names), max_cols, row_index + 1)
            ax.imshow(heatmap, cmap='hot')
            ax.set_title(f"{name} Heatmap (size {feature_map_size})", fontsize=12)
            ax.axis('off')
            ax.set_aspect('equal')

            row_index += 1

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_maps_heatmap.png'))
    plt.close(fig)

def visualize_feature_maps(feature_maps, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 定义特征名称
    feature_names = ['outs_rgb', 'outs_ir', 'branch_rgb', 'branch_ir']

    # 计算总的特征图数量
    total_feature_maps = sum(len(feature_maps[name]) for name in feature_names)

    # 设置图像大小
    fig_cols = max(len(feature_maps[name]) for name in feature_names)  # 最大列数
    fig_rows = len(feature_names)  # 行数
    fig = plt.figure(figsize=(fig_cols * 4, fig_rows * 4))  # 根据列和行数设置图像大小

    row_index = 0
    for name in feature_names:
        for i, feature_map in enumerate(feature_maps[name]):
            feature_map_np = feature_map[0].detach().cpu().numpy()  # 选择 B=0

            # 归一化所有通道
            normalized_maps = []
            for channel in range(feature_map_np.shape[0]):
                ch_map = feature_map_np[channel]
                min_val = np.min(ch_map)
                max_val = np.max(ch_map)

                # 处理分母为零的情况
                if max_val - min_val == 0:
                    normalized_map = np.zeros_like(ch_map)
                else:
                    normalized_map = (ch_map - min_val) / (max_val - min_val)
                
                normalized_maps.append(normalized_map)

            # 叠加所有归一化后的通道
            heatmap = np.max(normalized_maps, axis=0)

            # 绘制热力图
            ax = fig.add_subplot(fig_rows, fig_cols, row_index + 1)
            ax.imshow(heatmap, cmap='hot')
            ax.set_title(f"{name} Heatmap {i+1}", fontsize=10)
            ax.axis('off')
            ax.set_aspect('equal')

            row_index += 1

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_maps_heatmap.png'))
    plt.close(fig)


if __name__ == '__main__':
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseMamba().to(device)
    # change = ChannelExchange()
    input_tensor1 = torch.randn(1, 3, 480, 640).to(device)  # 假设输入是 224x224 的 RGB 图像
    input_tensor2 = torch.randn(1, 3, 480, 640).to(device)
    # x_c, y_c = change(input_tensor1, input_tensor2)
    # print(x_c.shape)
    # print(y_c.shape)
    x, x_, y, y_, _ = model(input_tensor1, input_tensor2)
    fusion = Fusion_Embed().to(device)
    fused = fusion(x, x_)
    # print("branch:", y[0].shape)
    # x, y = model(input_tensor1, input_tensor2)
    # print(x.shape)
    # print(x_.shape)
    # print(y.shape)
    # print(y_.shape)
    for m in x:
        print(m.shape)
    for m in fused:
        print(m.shape)
    # print(y[1].shape)
    '''
    input_image_rgb = Image.open('00003N_vi.png')
    input_image_ir = Image.open('00003N_ir.png').convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((480, 640)),  
        transforms.ToTensor(),
    ])

    input_tensor_rgb = transform(input_image_rgb).unsqueeze(0).cuda(1)
    input_tensor_ir = transform(input_image_ir).unsqueeze(0).cuda(1)

    # 使用 DenseMamba 模型和特征提取器
    model = DenseMamba().cuda(1)
    feature_extractor = FeatureExtractor(model)

    # 获取模型输出
    outs_rgb, outs_ir, branch_rgb, branch_ir, loss_aux = feature_extractor(input_tensor_rgb, input_tensor_ir)

    # 创建保存目录
    save_dir = 'feature_visualizations'
    os.makedirs(save_dir, exist_ok=True)

    # 可视化特征图
    for i, (rgb_out, ir_out) in enumerate(zip(outs_rgb, outs_ir)):
        rgb_out = rgb_out.detach().cpu().numpy()  
        print(f"Layer {i} RGB output shape: {rgb_out.shape}")  

        rgb_out = rgb_out.squeeze(0)  
        features_rgb = rgb_out.reshape(rgb_out.shape[0], -1).T  # (height * width, channels)

        # apply PCA
        pca_rgb = PCA(n_components=3)
        rgb_pca = pca_rgb.fit_transform(features_rgb)

        # normization
        rgb_pca = (rgb_pca - rgb_pca.min()) / (rgb_pca.max() - rgb_pca.min())
        rgb_image = rgb_pca.reshape(rgb_out.shape[1], rgb_out.shape[2], 3)  # (height, width, 3)

        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title(f'PCA Visualization of RGB Layer {i}')
        plt.savefig(f'{save_dir}/rgb_layer_{i}.png', bbox_inches='tight')
        plt.close()

        # 处理 IR 输出
        ir_out = ir_out.detach().cpu().numpy()  # 转换为 NumPy 数组
        print(f"Layer {i} IR output shape: {ir_out.shape}")

        ir_out = ir_out.squeeze(0)  # 去掉 batch_size 维度
        features_ir = ir_out.reshape(ir_out.shape[0], -1).T  # (height * width, channels)

        # 应用 PCA
        pca_ir = PCA(n_components=3)
        ir_pca = pca_ir.fit_transform(features_ir)

        # 归一化
        ir_pca = (ir_pca - ir_pca.min()) / (ir_pca.max() - ir_pca.min())
        ir_image = ir_pca.reshape(ir_out.shape[1], ir_out.shape[2], 3)  # (height, width, 3)

        # 保存 IR 图像
        plt.imshow(ir_image)
        plt.axis('off')
        plt.title(f'PCA Visualization of IR Layer {i}')
        plt.savefig(f'{save_dir}/ir_layer_{i}.png', bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {save_dir} directory.")

    feature_maps = {
    'outs_rgb': outs_rgb,
    'outs_ir': outs_ir,
    'branch_rgb': branch_rgb,
    'branch_ir': branch_ir
    }

    visualize_and_save_feature_maps(feature_maps, save_dir)
    visualize_feature_maps(feature_maps, save_dir)