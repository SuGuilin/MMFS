import sys
import os

sys.path.append("/home/suguilin/MMFS/")
sys.path.append("/home/suguilin/VMamba/")
sys.path.append("/home/suguilin/zigma/")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import yaml
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from utils.modules import Permute, VSSBlock
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from einops import rearrange
from typing import Callable
from functools import partial
from utils.utils import LayerNorm, Permute
from utils.modules import SS2D

from classification.models.vmamba import VSSM, LayerNorm2d

##########################################################################
## Backbone
class Backbone_VSSM(VSSM):
    def __init__(self, config, device, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, rgb_input, thermal_input, rgb_text, thermal_text):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        rgb_out = self.patch_embed(rgb_input)
        thermal_out = self.patch_embed(thermal_input)
        loss_aux = 0

        outs_rgb = []
        outs_thermal = []
        for i, layer in enumerate(self.layers):
            rgb_features, rgb_out = layer_forward(layer, rgb_out) 
            thermal_features, thermal_out = layer_forward(layer, thermal_out) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                rgb_norm_out = norm_layer(rgb_features)
                thermal_norm_out = norm_layer(thermal_features)
                if not self.channel_first:
                    rgb_norm_out = rgb_norm_out.permute(0, 2, 3, 1)
                    thermal_norm_out = thermal_norm_out.permute(0, 2, 3, 1)
                outs_rgb.append(rgb_norm_out.contiguous())
                outs_thermal.append(thermal_norm_out.contiguous())

        if len(self.out_indices) == 0:
            return rgb_out, thermal_out
        
        return outs_rgb, outs_thermal, loss_aux


##########################################################################
## Channel Dimensionality Reduction Learning
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


##########################################################################
## Channel Attendtion and Sptial Attendtion
class SAM(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SAM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(4, 4 * reduction, kernel_size=3, padding=1, stride=1, groups=4),
            nn.BatchNorm2d(4 * reduction),
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
            nn.BatchNorm2d(num_feat // squeeze_factor),
            nn.SiLU(inplace=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False), 
            nn.BatchNorm2d(num_feat),
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


def channel_shuffle(x, groups=2):
    batch_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(batch_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, w, h)
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
        
        x1 = channel_shuffle(x1)
        x2 = channel_shuffle(x2)
        x1 = self.encoder_x1(x1) + x1_short
        x2 = self.encoder_x2(x2) + x2_short
        x = [x1, x2]
        return x


class CIM(nn.Module):
    def __init__(self, dim, compress_ratio=3, squeeze_factor=16, kernel_size=1, reduction=4):
        super(CIM, self).__init__()
        self.channel_swap = nn.Sequential(
            TokenSwapMamba(dim=dim),
            TokenSwapMamba(dim=dim),
        )
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
        x1_flat, x2_flat = self.channel_swap([x1_flat, x2_flat])
        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=2))  ##B H*W C
        gated_weight = gated_weight.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  ## B C H W

        ca1 = self.cab1(x1)  ## B C H W
        ca2 = self.cab2(x2)  ## B C H W
        sp1, sp2 = self.sab(x1, x2)  ## B C H W
        attv_x1, attv_x2 = ca1 + sp1, ca2 + sp2 ## B C H W

        out_x1 = x1 + (1 - gated_weight) * attv_x2  # B C H W
        out_x2 = x2 + gated_weight * attv_x1  # B C H W
        return out_x1, out_x2


# Another version
class CMLLFF(nn.Module):
    def __init__(self, embed_dims, squeeze_factor=16, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(embed_dims*2, embed_dims*2 // squeeze_factor, 1, bias=False),
            nn.BatchNorm2d(embed_dims*2 // squeeze_factor),
            # nn.SiLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims*2 // squeeze_factor, embed_dims, 1, bias=False), 
            nn.BatchNorm2d(embed_dims),
            nn.Sigmoid(),
            )
            for _ in range(2)
        ])

        self.mlp = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(2, 4 * reduction, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(4 * reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, 1), 
            nn.Sigmoid()
            )
            for _ in range(2)
        ])

    def forward(self, x1, x2):
        gap_x1 = self.avg_pool(x1)
        gmp_x1 = self.max_pool(x1)
        ap_x1 = torch.mean(x1, dim=1, keepdim=True)
        mp_x1, _ = torch.max(x1, dim=1, keepdim=True) 
        gp_x1 = torch.cat([gap_x1, gmp_x1], dim=1)
        p_x1 = torch.cat([ap_x1, mp_x1], dim=1)
        gp_x1_ca = self.fc[0](gp_x1)
        p_x1_sp = self.mlp[0](p_x1)

        gap_x2 = self.avg_pool(x2)
        gmp_x2 = self.max_pool(x2)
        ap_x2 = torch.mean(x2, dim=1, keepdim=True)
        mp_x2, _ = torch.max(x2, dim=1, keepdim=True) 
        gp_x2 = torch.cat([gap_x2, gmp_x2], dim=1)
        p_x2 = torch.cat([ap_x2, mp_x2], dim=1)
        gp_x2_ca = self.fc[1](gp_x2)
        p_x2_sp = self.mlp[1](p_x2)

        out_x1 = x2 * gp_x1_ca + x2 * p_x1_sp + x1
        out_x2 = x1 * gp_x2_ca + x1 * p_x2_sp + x2
        
        return out_x1, out_x2 


##########################################################################
## Multi-branch Differential Bidirectional Fusion Network for RGB-T Semantic Segmentation
class MCS(nn.Module):
    def __init__(self, channels, reduction=4):
        super(MCS, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(1, 4 * reduction, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(4 * reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, 1), 
            nn.Sigmoid()
        )
    def forward(self, x):
        x_max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.mlp(x_max_out)
        return attention


class GCS(nn.Module):
    def __init__(self, channels):
        super(GCS, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = self.pool(x)
        attention = self.sigmoid(self.conv(pooled))
        return attention


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class TDE(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(TDE, self).__init__()
        self.mcs = MCS(in_channels)
        self.gcs = GCS(in_channels)
        self.cbr = CBR(in_channels*2, output_channels)

    def forward(self, RGB, TIR):
        R_c = torch.cat((RGB, TIR), dim=1) 
        F_tilde = self.cbr(R_c) 
        mask_r = self.mcs(RGB) 
        mask_t = self.gcs(TIR)  

        T_tilde = TIR * mask_r
        T_tilde = T_tilde * mask_t
        F_tilde = F_tilde * mask_r
        F_tilde = F_tilde + T_tilde
        return F_tilde


class RSE(nn.Module):
    def __init__(self, in_channels, fused_channels):
        super(RSE, self).__init__()
        self.mcs = MCS(in_channels)
        self.gcs = GCS(in_channels)

    def forward(self, RGB, TIR):
        Mask_i = self.mcs(TIR) 
        R_hat = RGB * Mask_i
        T_hat = TIR * Mask_i + R_hat

        R_tilde = self.gcs(R_hat) * R_hat
        T_tilde = self.gcs(T_hat) * T_hat

        F_tilde = R_tilde + T_tilde
        return F_tilde



# model = TDE(96, 96)
# x = torch.randn(4, 96, 480, 640)
# y = model(x, x)
# print(y.shape)


##########################################################################
## Multi-Order Feedback Network
class MSDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.
    """
    def __init__(self, embed_dims, output_dims, dw_dilation=[1, 2, 3,], channel_split=[1, 3, 4,], act=nn.SiLU(),):
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
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        v = self.PW_conv(x)
        return v


class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x1, x2):
        for module in self.modules_list:
            x1, x2 = module(x1, x2)  # two input, two output, cycle
        return x1, x2


##########################################################################
## Mamba Block
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
                (nn.Conv2d(dims[i] * (i + 2), dims[i], 1) 
                if i != self.num_layers - 1 else SimpleChannel(dims[i] * (i + 2), dims[i])),
                nn.ReLU(),
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
        modal = modal + outs[0]
        return modal#, loss_aux


##########################################################################
## Low-rank mixture of experts Block
class MoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        use_shuffle: bool = True,
        lr_space: str = "linear",
        recursive: int = 2,
    ):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0),
        )

        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, padding=2, groups=in_ch), nn.GELU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

        self.conv_2 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True), nn.GELU()
        )

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer(
            experts=[
                Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)
            ],  # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)

        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x)
        k = self.calibrate(k)

        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        # normalize the weights of the selected experts
        topk_weights = F.softmax(topk_weights, dim=1, dtype=torch.float).to(inputs.dtype)
        out = inputs.clone()

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i : i + 1, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i : i + 1, None, None]

        return out


class Expert(nn.Module):
    def __init__(
        self,
        in_ch: int,
        low_dim: int,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x  # here no more sigmoid
        x = self.conv_3(x)
        return x


class Router(nn.Module):
    def __init__(self, in_ch: int, num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c 1 1 -> b c"),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StripedConv2d(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int, depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
                groups=in_ch if depthwise else 1,
            ),
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
                groups=in_ch if depthwise else 1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ResMoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        lr_space: int = 1,
        recursive: int = 2,
        use_shuffle: bool = False,
    ):
        super().__init__()
        lr_space_mapping = {1: "linear", 2: "exp", 3: "double"}
        self.norm = LayerNorm(in_ch, data_format="channels_first")
        self.block = MoEBlock(
            in_ch=in_ch,
            num_experts=num_experts,
            topk=topk,
            use_shuffle=use_shuffle,
            recursive=recursive,
            lr_space=lr_space_mapping.get(lr_space, "linear"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm(x)) + x
        return x


##########################################################################
## Multi-Task Dense Prediction via Mixture of Low-Rank Experts
class LoraBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, rank=6):
        super().__init__()
        self.W = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.M = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W.bias)
        nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        nn.init.zeros_(self.M.bias)
    
    def forward(self, x):
        x = self.W(x)
        x = self.M(x)
        return x


class SpatialAtt(nn.Module):
    def __init__(self, dim, dim_out, im_size): # im_size=H*W
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim_out)
        self.convsp = nn.Conv2d(dim, dim, kernel_size=1) #nn.Linear(im_size, 1)
        self.ln_sp = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.conv3 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
    
    def forward(self, x): # output: (n, dim_out*2, 1, 1)
        n, _, h, w = x.shape
        feat = self.conv1(x)
        feat = self.ln(feat.reshape(n, -1, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, -1, h, w)
        feat = self.act(feat)
        feat = self.conv3(feat)

        feat_sp = torch.mean(x, dim=(2, 3), keepdim=True)
        feat_sp = self.convsp(feat_sp)
        feat_sp = self.ln_sp(feat_sp.reshape(n, -1)).reshape(n, -1, 1, 1)
        # feat_sp = self.convsp(x.reshape(n, -1, h * w)).reshape(n, 1, -1)
        # feat_sp = self.ln_sp(feat_sp).reshape(n, -1, 1, 1)
        feat_sp = self.act(feat_sp)
        feat_sp = self.conv2(feat_sp)
        
        n, c, h, w = feat.shape
        feat = torch.mean(feat.reshape(n, c, h * w), dim=2).reshape(n, c, 1, 1)
        feat = torch.cat([feat, feat_sp], dim=1)

        return feat


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.BatchNorm2d(num_feat // squeeze_factor),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
            nn.BatchNorm2d(num_feat),
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
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        x = self.cab(x)
        return x


class LoraMoEBlock(nn.Module):
    def __init__(self, config, final_embed_dim, im_size=120*160, kernel_size=3):
        super().__init__()
        self.num_lora = len(config.rank_list)
        self.config = config
        self.lora_list_1 = nn.ModuleList()
        rank_list = config.rank_list
        for i in range(self.num_lora):
            self.lora_list_1.append(LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=rank_list[i]))
            self.lora_list_1[i].init_weights()
        self.conv1 = nn.ModuleDict()
        self.conv2 = nn.ModuleDict()
        self.conv3 = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        self.bn_all = nn.ModuleDict()
        self.share_conv = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=3, padding=1)
        self.router_1 = nn.ModuleDict() 
        self.activate = nn.GELU()
        for modal in self.config.modals:
            self.conv1[modal] = CAB(final_embed_dim) #nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            self.conv3[modal] = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            self.conv2[modal] = LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=config.spe_rank)

            self.bn[modal] = nn.BatchNorm2d(final_embed_dim)
            self.bn_all[modal] = nn.BatchNorm2d(final_embed_dim)

        self.pre_softmax = config.pre_softmax
        self.desert_k = len(config.rank_list) - config.topk
        for modal in self.config.modals:
            self.router_1[modal] = nn.ModuleList()
            self.router_1[modal].append(SpatialAtt(final_embed_dim, final_embed_dim // 4, im_size=im_size))
            self.router_1[modal].append(nn.Conv2d(final_embed_dim // 2, self.num_lora * 2 + 1, kernel_size=1))
        
    def forward(self, x, modal):
        out_ori = self.conv1[modal](x)
        out = out_ori
        n, c, h, w = out.shape
        route_feat = self.router_1[modal][0](out)
        prob_all = self.router_1[modal][1](route_feat).unsqueeze(2)
        prob_lora, prob_mix = prob_all[:, :self.num_lora * 2], prob_all[:, self.num_lora * 2:]
        route_1_raw, stdev_1 = prob_lora.chunk(2, dim=1)  # n, 15, 1, 1, 1
        if self.training:
            noise = torch.randn_like(route_1_raw) * stdev_1
        else:
            noise = 0
        if self.pre_softmax:
            route_1_raw = route_1_raw + noise
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            # Set unselected expert gate values ​​to negative infinity
            for j in range(n):
                for i in range(self.desert_k):
                    route_1_raw[j, route_1_indice[j, i].reshape(-1)] = -1e10
            route_1 = torch.softmax(route_1_raw, dim=1)
        else:
            route_1_raw = torch.softmax(route_1_raw + noise, dim=1)
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            route_1 = route_1_raw.clone()
            for j in range(n):
                for i in range(self.desert_k):
                    route_1[j, route_1_indice[j, i].reshape(-1)] = 0
        
        lora_out_1 = []
        for i in range(self.num_lora):
            lora_out_1.append(self.lora_list_1[i](out).unsqueeze(1)) # n, 1, c, h, w
        lora_out_1 = torch.cat(lora_out_1, dim=1)
        lora_out_1 = torch.sum(lora_out_1 * route_1, dim=1)
        
        out = self.bn_all[modal](lora_out_1) + self.conv2[modal](out) * prob_mix[:, 0] + self.share_conv(out.detach())
        out = self.bn[modal](out)
        out = self.activate(out)
        out = self.conv3[modal](out)
        return out#, route_feat, route_1


##########################################################################
## Linear Attention
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding."""

    def __init__(self, dim, base=10000):
        super(RoPE, self).__init__()
        self.base = base
        self.dim = dim

    def forward(self, x):
        shape = x.shape[-3:-1]  # Get the last two dimensions as shape
        channel_dims = shape
        feature_dim = self.dim
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (self.base ** (torch.arange(k_max) / k_max))
        angles = torch.cat(
            [
                t.unsqueeze(-1) * theta_ks
                for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing="ij")
            ], dim=-1,).to(x.device)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)

        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    r"""Linear Attention with LePE and RoPE. 

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(dim=dim)

    def forward(self, x, h, w):
        """
        Args:
            x: input features with shape of (B, N, C)
            h: height
            w: width
        """
        b, n, c = x.shape

        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n**-0.5)) @ (v * (n**-0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"


class LinearAttention2(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(dim=dim)
        
    def forward(self, x1, x2, h, w):
        b, n, c = x1.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        # Query and Key for x1
        qk_x1 = self.qk(x1).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q1, k1 = qk_x1[0], qk_x1[1]
        q1 = self.elu(q1) + 1.0
        k1 = self.elu(k1) + 1.0

        # Query and Key for x2
        qk_x2 = self.qk(x2).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q2, k2 = qk_x2[0], qk_x2[1]
        q2 = self.elu(q2) + 1.0
        k2 = self.elu(k2) + 1.0
        
        # Rotate Position Embedding
        q1_rope = self.rope(q1.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k1_rope = self.rope(k1.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q2_rope = self.rope(q2.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k2_rope = self.rope(k2.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # Reshape q, k, v
        q1 = q1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k1 = k1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q2 = q2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k2 = k2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v1 = self.v(x1).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v2 = self.v(x2).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # Cross-attention
        z1 = 1 / (q1 @ k2.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        z2 = 1 / (q2 @ k1.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        
        kv2 = (k2_rope.transpose(-2, -1) * (n**-0.5)) @ (v2 * (n**-0.5))
        kv1 = (k1_rope.transpose(-2, -1) * (n**-0.5)) @ (v1 * (n**-0.5))
        
        x1_out = q1_rope @ kv2 * z1
        x2_out = q2_rope @ kv1 * z2

        x1_out = x1_out.transpose(1, 2).reshape(b, n, c)
        x2_out = x2_out.transpose(1, 2).reshape(b, n, c)
        
        v1 = v1.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        v2 = v2.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        x1_out = x1_out + self.lepe(v1).permute(0, 2, 3, 1).reshape(b, n, c)
        x2_out = x2_out + self.lepe(v2).permute(0, 2, 3, 1).reshape(b, n, c)

        x1_out = x1_out.permute(0, 2, 1).reshape(b, c, h, w)
        x2_out = x2_out.permute(0, 2, 1).reshape(b, c, h, w)
        
        return x1_out, x2_out


##########################################################################
## Detail Feature Extraction
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=in_dim, oup=out_dim, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=in_dim, oup=out_dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=in_dim, oup=out_dim, expand_ratio=2)
        self.shffleconv = nn.Conv2d(
            in_dim * 2, in_dim * 2, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, dim, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(in_dim=dim // 2, out_dim=dim // 2) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        # self.reduce_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        z = torch.cat((z1, z2), dim=1)
        # z = self.reduce_conv(z)
        return z


##########################################################################
## Detail Feature Extraction
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GatedFFN(nn.Module):
    def __init__(
        self,
        in_ch,
        mlp_ratio,
        kernel_size,
        act_layer,
    ):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio

        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )

        self.gate = nn.Conv2d(
            mlp_ch // 2,
            mlp_ch // 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=mlp_ch // 2,
        )

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x

    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)

        gate = self.gate(gate)
        x = x * gate

        x = self.fn_2(x)
        return x


class GatedPercept(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=2):
        super(GatedPercept, self).__init__()
        self.hidden_channels = 2 * in_channels * expansion_ratio
        self.rdconv = in_channels // 2

        self.norm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, self.hidden_channels, kernel_size=1, bias=False
        )
        self.dconv = nn.Conv2d(
            self.rdconv * 2,
            self.rdconv * 2,
            kernel_size=3,
            padding=1,
            groups=self.rdconv,
            bias=False,
        )
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            self.hidden_channels // 2, in_channels, kernel_size=1, bias=False
        )
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # Normalize
        x = self.norm(x)

        # Linear proj
        F_in = self.conv1(x)

        c1 = self.rdconv * 2
        c2 = self.hidden_channels // 2

        # Split into three parts
        F_alpha = F_in[:, :c1, :, :]
        F_beta = F_in[:, c1:c2, :, :]
        F_gamma = F_in[:, c2:, :, :]

        # Process each part
        F_alpha = self.dconv(F_alpha)
        F_gamma = self.gelu(F_gamma)

        # Concatenate and multiply
        F_mod = torch.cat([F_alpha, F_beta], dim=1) * F_gamma

        # Linear projection back to original channels
        F_out = self.conv3(self.conv2(F_mod) + x)

        return F_out


class VMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        drop_rate: float = 0,
        d_state: int = 16,
        ffn_expansion_factor: int = 2,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        **kwargs,
    ):
        super().__init__()
        self.dim = hidden_dim // 2
        self.channel_swap = TokenSwapMamba(dim=self.dim)

        self.norm1 = norm_layer(self.dim)
        self.op = SS2D(d_model=self.dim, dropout=drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale1 = nn.Parameter(
            torch.ones((1, 1, 1, self.dim)), requires_grad=True
        )
        self.conv_blk = CAB(self.dim)
        self.norm2 = norm_layer(self.dim)
        self.skip_scale2 = nn.Parameter(
            torch.ones((1, 1, 1, self.dim)), requires_grad=True
        )

        self.local_refine = GatedPercept(in_channels=self.dim, out_channels=self.dim)
        self.local_ffn = GatedFFN(self.dim, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

        self.sigmoid = nn.Sigmoid()

        self.neck = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)

        self.norm_ffn = LayerNorm(hidden_dim, LayerNorm_type)
        self.FFN = FeedForward(
            dim=hidden_dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias
        )

    def forward(self, input: torch.Tensor):
        B, C, H, W = input.shape
        input1 = input[:, ::2, :, :].permute(0, 2, 3, 1).view(B, -1, C // 2)
        input2 = input[:, 1::2, :, :].permute(0, 2, 3, 1).view(B, -1, C // 2)
        x1, x2 = self.channel_swap([input1, input2])

        x1 = x1.view(B, H, W, C // 2)
        x2 = x2.view(B, H, W, C // 2).permute(0, 3, 1, 2)

        x1 = x1 * self.skip_scale1 + self.drop_path(self.op(self.norm1(x1)))
        x1 = (
            x1 * self.skip_scale2
            + self.conv_blk(self.norm2(x1).permute(0, 3, 1, 2).contiguous())
            .permute(0, 2, 3, 1).contiguous()
        )
        x1 = x1.permute(0, 3, 1, 2)

        x2 = self.local_refine(x2)
        x2 = x2 + self.local_ffn(x2)

        x1 = x1 + self.sigmoid(x2)
        x2 = x2 + self.sigmoid(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.neck(x) + input

        x = self.FFN(self.norm_ffn(x)) + x
        return x



# def load_config(yaml_path):
#     with open(yaml_path, "r") as f:
#         config = yaml.safe_load(f)
#     return edict(config)

# config_path = '../configs/config_mfnet.yaml'
# config = load_config(config_path)

x1 = torch.randn(4, 96, 120, 160)
# model1 = LoraMoEBlock(config=config, final_embed_dim=96, im_size=120*160)
# y1, _, _ = model1(x, 'thermal')
# print(y1.shape)

# x2 = torch.randn(4, 120*160, 96)
# model2 = LinearAttention(dim=96, num_heads=8)
# y2 = model2(x2, 120, 160)
# print(y2.shape)

# model3 = LinearAttention2(dim=96, num_heads=8)
# y3_1, y3_2 = model3(x2, x2, 120, 160)
# print(y3_1.shape)

# model4 = DetailFeatureExtraction(dim=96)
# y4 = model4(x1)
# print(y4.shape)

# model5 = VMBlock(hidden_dim=96, ffn_expansion_factor=4, bias=False, LayerNorm_type="WithBias",)
# y5 = model5(x1)
# print(y5.shape)

# model6 = CMLLFF(embed_dims=96)
# y6_1, y6_2 = model6(x1, x1)
# print(y6_1.shape)
# print(y6_2.shape)
