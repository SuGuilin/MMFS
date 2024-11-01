import sys

sys.path.append("/home/suguilin/Graduation/myfusion/")
sys.path.append("/home/suguilin/zigma/")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.modules.mamba_simple import Mamba
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from utils.MixBlock import LinearAttention
from utils.NAFBlock import NAFBlock
from utils.utils import *
from utils.InvertedBlock import DetailFeatureExtraction
from utils.STMBlock import STMBlock
from encoder.cross_module import CrossFusionBlock
import warnings


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_model_rate=1,
        d_state=16,
        d_conv=3,
        ssm_ratio=2,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        conv_bias=True,
        bias=False,
        dtype=None,
        # ======================
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        shared_ssm=False,
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": dtype}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = (
            math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        )  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4 if not shared_ssm else 1

        self.in_proj = nn.Linear(
            self.d_model * d_model_rate, self.d_inner, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = [
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            )
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K * inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=self.K, merge=True
        )  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)  # (K * D)

        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    forward_core = forward_corev0

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        # x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)
        # y = y * F.silu(z)
        # out = self.out_proj(y)
        # if self.dropout is not None:
        #     out = self.dropout(out)
        return y

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )

    def forward(self, query, key, value):
        query = query.transpose(0, 1)  # (149, N, 768)
        key = key.transpose(0, 1)  # (1440, N, 768)
        value = value.transpose(0, 1)  # (1440, N, 768)

        attn_output, _ = self.multihead_attn(query, key, value)  # (149, N, 768)
        attn_output = attn_output.transpose(0, 1)  # (N, 149, 768)
        return attn_output

class LinearCrossAttention(nn.Module):
    def __init__(self, dim_A, dim_B):
        super().__init__()
        self.to_q_A = nn.Linear(dim_A, dim_A) 
        self.to_kv_B = nn.Linear(dim_B, dim_B * 2) 
        self.scale = dim_A ** -0.5
        self.to_out = nn.Linear(dim_A, dim_A)

    def forward(self, feature_A, feature_B):
        q_A = self.to_q_A(feature_A)
        k_B, v_B = self.to_kv_B(feature_B).chunk(2, dim=-1)
        attention = torch.einsum('bqd,bkd->bqk', q_A, k_B) * self.scale
        attention = attention.softmax(dim=-1)
        out = torch.einsum('bqk,bvd->bqd', attention, v_B)
        return self.to_out(out)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q(x).view(B, C, -1)  
        k = self.k(x).view(B, C, -1).permute(0, 2, 1)  
        v = self.v(x).view(B, C, -1) 

        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(attn, v).view(B, C, H, W)
        return out
    
class image2text(nn.Module):
    def __init__(self, in_channel, mid_channel, hidden_dim):
        super(image2text, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel, out_channels=mid_channel, kernel_size=1
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv(x)
        # (N, num_patches, hidden_dim)
        x = x.contiguous().view(
            x.size(0), x.size().numel() // x.size(0) // self.hidden_dim, self.hidden_dim
        )
        return x


class AttReduceDim(nn.Module):
    def __init__(self, in_chans, out_chans, height=288, width=384, stride=2, padding=1):
        super(AttReduceDim, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_chans,
            out_channels=in_chans,
            kernel_size=4,
            stride=stride,
            padding=padding,
        )
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans, 1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv3 = nn.Conv2d(in_chans, out_chans, 1)
        self.linear_attn = LinearAttention(
            dim=out_chans,
            input_resolution=(height, width),
            num_heads=8,
            qkv_bias=True,
        )
        self.height = height
        self.width = width
        self.out_chans = out_chans

    def forward(self, x):
        N, C, H, W = x.shape
        if H != 288 or W != 384:
            x = self.act(self.deconv(x))
            x = F.interpolate(x, [288, 384], mode="nearest")
        pooled = self.pool(x)
        ch_att = self.conv1(pooled)
        x = x * ch_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp_att = self.conv2(torch.cat([avg_out, max_out], dim=1))
        sp_att = torch.sigmoid(sp_att)
        x = x * (sp_att + 1)
        x = self.conv3(x)
        y = x.permute(0, 2, 3, 1).view(N, -1, self.out_chans)
        x = self.linear_attn(y, h=288, w=384).permute(0, 2, 1).view(N, self.out_chans, self.height, self.width) + x  # (N, L, C)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_chans=32, embed_dim=32):
        super().__init__()

        self.embed_dim = embed_dim
        if self.embed_dim == 128:
            self.proj1 = nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=patch_size,
                stride=stride,
                padding=patch_size // 2,
            )
            self.scale = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=1),
                nn.Sigmoid(),
            )
            self.proj2 = nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=patch_size // 2,
            )
        else:
            self.proj1 = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=patch_size // 2,
            )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
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
        x = self.proj1(x)
        if self.embed_dim == 128:
            y = self.scale(x)
            x = x * y
            x = self.proj2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x).permute(0, 3, 1, 2)
        return x


class CrossMamba(nn.Module):
    def __init__(
        self,
        in_chans: int = 128,
        text_inchans: int = 768,
        image2text_dim: int = 32,
        image_dim: int = 32,
        layers: int = 2,
        hidden_dim: int = 256,
        embed_dims=[32, 64, 128],
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.layers = layers
        # self.image2imgae = AttReduceDim(in_chans, image_dim)
        self.text_process = nn.Conv1d(text_inchans, hidden_dim, 1, 1, 0)
        self.text_encoding = Mamba(hidden_dim, bimamba_type=None)
        self.image2text_dim = image2text_dim

        self.image2imgae = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.op = nn.ModuleList()
        self.image2text = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.mlp = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.act1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.act2 = nn.ModuleList()
        self.proj = nn.ModuleList()
        
        # self.patchembed = nn.ModuleList()
        # self.patchembed = OverlapPatchEmbed(
        #     patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]
        # )
        # self.patchnormed = norm_layer(embed_dims[0])
        for i in range(self.layers):
            if i == 0:
                self.image2imgae.append(AttReduceDim(in_chans, image_dim))
            else:
                self.image2imgae.append(AttReduceDim(image_dim * 2, image_dim, stride=4, padding=0))
            self.op.append(
                SS2D(
                    d_model=image_dim,
                    d_model_rate=1,
                    dropout=attn_drop_rate,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    dt_rank=dt_rank,
                    shared_ssm=shared_ssm,
                    softmax_version=softmax_version,
                    **kwargs,
                )
            )

            self.image2text.append(
                image2text(int(image_dim * ssm_ratio), image2text_dim, hidden_dim)
            )
            self.norm.append(norm_layer(image_dim))
            self.proj.append(nn.Linear(int(image_dim * ssm_ratio), image_dim, bias=False))

            self.cross_attention.append(CrossAttention(embed_dim=hidden_dim, num_heads=8))
            self.mlp.append(Mlp(in_features=image_dim, hidden_features=int(image_dim * mlp_ratio), out_features=image_dim))

            self.conv1.append(nn.Conv2d(image2text_dim, image_dim, kernel_size=1))
            self.act1.append(nn.PReLU())
            self.conv2.append(nn.Conv2d(image_dim * 2, image_dim, kernel_size=1))
            self.act2.append(nn.PReLU())
            # if i < 2:
            #     self.patchembed.append(
            #         OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[i + 1])
            #     )

        self.drop_path = DropPath(drop_path)
        self.enhance = NAFBlock(c=image_dim * 2)
        self.conv = nn.Conv2d(image_dim*2, image_dim, kernel_size=1)
        # self.semantic_fea = []

    def forward(self, image_list, text):
        # image = self.image2imgae(image)
        # N, C, H, W = image.shape
        # fea_init = list(torch.split(image, 64, dim=1))
        # semantic_fea = []
        text = self.text_encoding(self.text_process(text.permute(0, 2, 1))).permute(0, 2, 1)  # (N, sql, C) -> (8, 256, 256)
        # cross_att = None
        cross_att_list = []
        for i in range(self.layers):
            image = self.image2imgae[i](image_list[i])
            N, C, H, W = image.shape
            image = self.norm[i](image.permute(0, 2, 3, 1))  # (N, 288, 384, 32)
            image = self.op[i](image).permute(0, 3, 1, 2)  # (N, 64, 288, 384)
            image_short = image  # (N, 64, 288, 384)
            image2text = self.image2text[i](image)  # (N, 13824, 256)
            cross_att = self.cross_attention[i](text, image2text, image2text)  # (N, 256, 256)
            cross_att = torch.nn.functional.adaptive_avg_pool1d(cross_att.permute(0, 2, 1), 1).permute(0, 2, 1)  # (N, 1, 256)
            cross_att = (image2text * cross_att).view(N, self.image2text_dim, H, W)  # (N, 13824, 256) -> (N, 32, 288, 384)
            cross_att = self.mlp[i](cross_att.flatten(2).transpose(1, 2), H, W).transpose(1, 2).view(N, -1, H, W) # (N, 32, 288, 384)
            image_short = self.proj[i](image_short.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (N, 64, 288, 384) -> (N, 32, 288, 384)
            # cross_att = self.act1[i](self.conv1[i](torch.cat([cross_att, F.interpolate(fea_init[i], [288, 384], mode="bilinear")], dim=1))) + self.drop_path(image_short)  # (N, 32, 288, 384)
            cross_att = self.act1[i](self.conv1[i](cross_att)) + self.drop_path(image_short)  # (N, 32, 288, 384)
            cross_att = self.act2[i](self.conv2[i](torch.cat([image_short, cross_att], dim=1)))  # (N, 32, 288, 384)
            cross_att_list.append(cross_att)
            # image = cross_att
            # if i == 0:
                # semantic_fea.append(self.patchnormed(cross_att.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            # else:
            #     # self.semantic_fea.append(self.patchembed[i - 1](image))
            #     semantic_fea.append(self.patchembed(cross_att))
        # cross_att = self.enhance(cross_att) + cross_att
        cross_att = self.conv(self.enhance(torch.cat(cross_att_list, dim=1)))
        # self.semantic_fea.append(cross_att)
        return cross_att #, semantic_fea

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


class CIM(nn.Module):
    def __init__(self, dim, compress_ratio=3, squeeze_factor=16, kernel_size=1, reduction=4):
        super(CIM, self).__init__()
        self.cab1 = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.cab2 = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.sab = SAB(kernel_size=kernel_size, reduction=reduction)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2 // reduction, dim),
            nn.Sigmoid()
        )
        # self.global_w1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Sigmoid(),
        # )
        # self.global_w2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Sigmoid(),
        # )
        # self.att = nn.Conv2d(dim, dim, 1)
        # self.norm = LayerNorm(dim, 'WithBias')
        # self.resconv = ResBlock(in_channels=dim, out_channels=dim)

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
        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=2))  ##B H*W C
        gated_weight = gated_weight.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  ## B C H W

        ca1 = self.cab1(x1)  ## B C H W
        ca2 = self.cab2(x2)  ## B C H W
        sp1, sp2 = self.sab(x1, x2)  ## B C H W
        attv_x1, attv_x2 = ca1 + sp1, ca2 + sp2 ## B C H W

        out_x1 = x1 + (1 - gated_weight) * attv_x2  # B C H W
        out_x2 = x2 + gated_weight * attv_x1  # B C H W
        
        # out_x1 = self.global_w1(out_x1) * out_x1
        # out_x2 = self.global_w2(out_x2) * out_x2
        # out = out_x1 + out_x2
        # out = self.norm(self.resconv(self.att(out)))
        return out_x1, out_x2
        # return out_x1, out_x2
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect", groups=out_channels // reduction),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
        )
        self.norm = norm_layer(out_channels)
    def forward(self, x):
        out = self.norm(self.conv(x) + self.residual(x))
        return out


class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim=[48, 96, 192, 384], bias=False, heads=8, norm_layer=nn.LayerNorm,):
        super(Fusion_Embed, self).__init__()
        self.num_heads = heads
        self.embed_dim = embed_dim
        '''
        self.detail_x1 = DetailFeatureExtraction(dim=embed_dim[0], num_layers=2)
        self.detail_x2 = DetailFeatureExtraction(dim=embed_dim[0], num_layers=2)
        self.global_x1 = nn.Sequential(
            Permute(0, 2, 3, 1),
            STMBlock(hidden_dim=embed_dim[0],
                drop_path=0.1,
                ssm_ratio=2.0,
                d_state=16,
                dt_rank="auto",
                mlp_ratio=4.0,
                norm_layer=nn.LayerNorm,),
            Permute(0, 3, 1, 2),
        )

        self.global_x2 = nn.Sequential(
            Permute(0, 2, 3, 1),
            STMBlock(hidden_dim=embed_dim[0],
                drop_path=0.1,
                ssm_ratio=2.0,
                d_state=16,
                dt_rank="auto",
                mlp_ratio=4.0,
                norm_layer=nn.LayerNorm,),
            Permute(0, 3, 1, 2),
        )

        self.FFN_x1 = nn.Conv2d(
            embed_dim[0] * 2,
            embed_dim[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.FFN_x2 = nn.Conv2d(
            embed_dim[0] * 2,
            embed_dim[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        '''
        # self.cross_attention = nn.ModuleList() 
        # self.attention = nn.ModuleList()
        '''
        self.fuse = nn.ModuleList()
        self.q_A = nn.ModuleList()
        self.k_A = nn.ModuleList() 
        self.v_A = nn.ModuleList() 
        self.q_B = nn.ModuleList()
        self.k_B = nn.ModuleList()
        self.v_B = nn.ModuleList()

        self.sp_A = nn.ModuleList()
        self.sp_B = nn.ModuleList()
        
        self.scale1 = nn.ParameterList()
        self.scale2 = nn.ParameterList()
        self.scale3 = nn.ParameterList()
        self.scale4 = nn.ParameterList()

        self.fusion_proj = nn.ModuleList()
        self.output = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)
        '''
        self.fused = nn.ModuleList()
        self.fusion_proj = nn.ModuleList()
        for i in range(len(embed_dim)):
            # self.fused.append(nn.Conv2d(embed_dim[i] * 2, embed_dim[i], kernel_size=3, padding=1, stride=1, bias=bias))
            # self.fused.append(ResBlock(in_channels=embed_dim[i] * 2, out_channels=embed_dim[i]))
            
            self.fused.append(
                CrossFusionBlock(
                    modals=2,
                    hidden_dim=embed_dim[i],
                    drop_path=0.1,
                    shared_ssm=False,
                    d_state=16,
                    ssm_ratio=2.0,
                    dt_rank="auto",
                    mlp_ratio=4,
                    norm_layer=norm_layer,
                )
            )

            self.fusion_proj.append(ResBlock(in_channels=embed_dim[i], out_channels=embed_dim[i]))
            '''
            self.fuse.append(CIM(dim=embed_dim[i]))
            # self.cross_attention.append(LinearCrossAttention(embed_dim[i], embed_dim[i]))
            self.fusion_proj.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim[i] * 4, embed_dim[i], kernel_size=3, padding=1, stride=1, bias=bias),
                    # nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=3, padding=1, stride=1, bias=bias), 
                    # nn.ReLU(),
                    # nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=1, stride=1, bias=bias),
                    # Permute(0, 2, 3, 1),
                    # nn.LayerNorm(embed_dim[i]),
                    # Permute(0, 3, 1, 2),
                )
            )
            self.output.append(nn.Conv2d(embed_dim[i] * 3, embed_dim[i], kernel_size=3, padding=1, stride=1, bias=bias))
            # self.attention.append(
            #     SelfAttention(in_channels=embed_dim[i])
            # )
            self.q_A.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(embed_dim[i]),
                    nn.ReLU(inplace=True),
                )
            )
            self.k_A.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(embed_dim[i]),
                    nn.ReLU(inplace=True),
                )
            )
            self.v_A.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(embed_dim[i]),
                    nn.ReLU(inplace=True),
                )
            )
            self.q_B.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(embed_dim[i]),
                    nn.ReLU(inplace=True),
                )
            )
            self.k_B.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(embed_dim[i]),
                    nn.ReLU(inplace=True),
                )
            )
            self.v_B.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(embed_dim[i]),
                    nn.ReLU(inplace=True),
                )
            )
            self.sp_A.append(SpatialAttention())
            self.sp_B.append(SpatialAttention()) 
            # self.kv_A.append(nn.Conv2d(embed_dim[i], embed_dim[i] * 2, kernel_size=1))
            # self.kv_B.append(nn.Conv2d(embed_dim[i], embed_dim[i] * 2, kernel_size=1))
            # self.q.append(nn.Conv2d(embed_dim[i] * 2, embed_dim[i], kernel_size=1))
            self.scale1.append(nn.Parameter(torch.ones(self.num_heads, 1, 1)))
            self.scale2.append(nn.Parameter(torch.ones(self.num_heads, 1, 1)))
            self.scale3.append(nn.Parameter(torch.ones(self.num_heads, 1, 1)))
            self.scale4.append(nn.Parameter(torch.ones(self.num_heads, 1, 1)))
            '''

    def forward(self, x_A, x_B):
        fusion_outs = []
        for j in range(len(self.embed_dim)):
            b, c, h, w = x_A[j].shape
            # fA = x_A[j].flatten(2).transpose(1, 2)
            # fB = x_B[j].flatten(2).transpose(1, 2)
            # x1 = self.cross_attention[j](fA, fB).transpose(1, 2).view(N, -1, H, W)
            # x2 = self.cross_attention[j](fB, fA).transpose(1, 2).view(N, -1, H, W)
            x1, x2 = x_A[j], x_B[j]
            # print(x1.shape)
            # if j == 0:
            #     x1_1, x2_1 = self.detail_x1(x1), self.detail_x2(x2)
            #     x1_2, x2_2 = self.global_x1(x1), self.global_x2(x2)
            #     x1 = self.FFN_x1(torch.cat((x1_1, x1_2), 1))
            #     x2 = self.FFN_x2(torch.cat((x2_1, x2_2), 1))
            # 先自注意力再卷积，试一下效果
            # x1 = self.attention[j](x1) + x1
            # x2 = self.attention[j](x2) + x2
            # x = torch.concat([x1, x2], dim=1)  # .permute(0, 3, 1, 2)
            # query = self.q[j](x)
            # k_A, v_A = self.kv_A[j](x1).chunk(2, dim=1)
            # k_B, v_B = self.kv_B[j](x2).chunk(2, dim=1)
            '''
            q_A = self.q_A[j](x1)
            k_A = self.k_A[j](x1)
            v_A = self.v_A[j](x1)
            q_B = self.q_B[j](x2)
            k_B = self.k_B[j](x2)
            v_B = self.v_B[j](x2)

            q_A = rearrange(q_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k_A = rearrange(k_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v_A = rearrange(v_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            q_B = rearrange(q_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k_B = rearrange(k_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v_B = rearrange(v_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            
            q_A = torch.nn.functional.normalize(q_A, dim=-1)
            k_A = torch.nn.functional.normalize(k_A, dim=-1)
            q_B = torch.nn.functional.normalize(q_B, dim=-1)
            k_B = torch.nn.functional.normalize(k_B, dim=-1)

            v = v_A + v_B
            
            attn_A1 = (q_B @ k_A.transpose(-2, -1)) * self.scale1[j]
            attn_A1 = self.softmax(attn_A1)
            out_A1 = (attn_A1 @ v_A) 
            out_A1 = rearrange(out_A1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) + x1

            attn_A2 = (q_A @ k_A.transpose(-2, -1)) * self.scale2[j]
            attn_A2 = self.softmax(attn_A2)
            out_A2 = (attn_A2 @ v) 
            out_A2 = rearrange(out_A2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) + x1


            attn_B1 = (q_A @ k_B.transpose(-2, -1)) * self.scale3[j]
            attn_B1 = self.softmax(attn_B1)
            out_B1 = (attn_B1 @ v_B) 
            out_B1 = rearrange(out_B1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) + x2

            attn_B2 = (q_B @ k_B.transpose(-2, -1)) * self.scale4[j]
            attn_B2 = self.softmax(attn_B2)
            out_B2 = (attn_B2 @ v) 
            out_B2 = rearrange(out_B2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) + x2
            


            out = torch.cat([out_A1, out_A2, out_B1, out_B2], dim=1)
            x = self.fusion_proj[j](out)  # .permute(0, 2, 3, 1)
            out1 = self.sp_A[j](x1)
            out2 = self.sp_B[j](x2)
            x = self.output[j](torch.cat([out1, out2, x], dim=1))
            '''
            # x = self.fused[j](torch.cat([x1, x2], dim=1))
            x = self.fused[j](x1, x2)
            x = self.fusion_proj[j](x)
            # x = self.fuse[j](x1, x2)
            # x = self.attention[j](x)
            fusion_outs.append(x)
            # print("fusion:", x.shape)
        return fusion_outs


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    text_process = nn.Conv1d(768, 256, 1, 1, 0).to(device)
    text_encoding = Mamba(256, bimamba_type=None).to(device)
    y = torch.randn(8, 256, 768).to(device)
    y = text_process(y.permute(0, 2, 1)).permute(0, 2, 1)
    print(y.shape)
    y = text_encoding(y)
    print(y.shape)

    reduce = AttReduceDim(128, 32).to(device)
    x = torch.randn(8, 128, 288, 384).to(device)
    x = reduce(x).permute(0, 2, 3, 1)
    print(x.shape)
    img_process = SS2D(
                    d_model=32,
                    d_model_rate=1,
                    dropout=0,
                    d_state=16,
                    ssm_ratio=2.0,
                    dt_rank="auto",
                    shared_ssm=False,
                    softmax_version=False,
                ).to(device)
    x = img_process(x).permute(0, 3, 1, 2)
    short = x
    print(x.shape)
    img2text = image2text(64, 32, 256).to(device)
    x = img2text(x)
    img = x
    print(x.shape)

    cross_att = CrossAttention(embed_dim=256, num_heads=8).to(device)
    z = cross_att(y, x, x)
    print(z.shape)
    z = torch.nn.functional.adaptive_avg_pool1d(z.permute(0, 2, 1), 1).permute(0, 2, 1)
    print(z.shape)
    print((img * z).shape) # torch.Size([8, 13824, 256])
    z = (img * z).view(8, 32, 288, 384)
    print(z.shape)

    conv1 = nn.Conv2d(32, 32, kernel_size=1).to(device)
    act1 = nn.PReLU().to(device)
    conv2 = nn.Conv2d(2 * 32, 32, kernel_size=1).to(device)
    act2 = nn.PReLU().to(device)
    proj = nn.Linear(64, 32, bias=False).to(device)
    cross_att = act1(conv1(z)) + proj(short.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    print(cross_att.shape)
    cross_att = act2(conv2(torch.cat([proj(short.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), cross_att], dim=1)))
    print(cross_att.shape)
    """
    # from textencoder import TextEncoder
    # import cv2
    # text_batch = [
    #     ['In this image with a resolution of 384X288, we can see a person walking down a street at night. The dense caption provides further details about the various objects present. A woman can be seen walking on the road, positioned between coordinates [97, 76, 171, 248]. Another person is also depicted walking, located at coordinates [256, 98, 276, 163]. A grey backpack can be observed on a person, positioned between coordinates [123, 110, 164, 173]. Moreover, a car is parked on the side of the road, situated at coordinates [63, 116, 111, 164]. Additionally, the region semantic provides additional information about different elements in the image. A woman is depicted walking with a suitcase between coordinates [0, 149, 383, 137]. There is also a black and white photo of a man standing in front of a wall, positioned at [341, 50, 42, 95]. Furthermore, a black object on a black background can be seen at coordinates [101, 169, 65, 45]. Additionally, a black and white photo of a building can be found, wherein the lights are on. It is located at [0, 97, 72, 33]. Lastly, a white figure is depicted against a black background, positioned at [124, 101, 38, 71].'],
    #     ]
    # image = torch.from_numpy(cv2.resize(cv2.imread('00001D.png'), (384, 288))).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
    # text_encoder = TextEncoder(device=device).to(device)
    # _, _, text = text_encoder(text_batch)
    # print(text.shape)
    text = torch.randn(8, 256, 768).to(device)
    image = torch.randn(8, 128, 288, 384).to(device)
    model = CrossMamba().to(device)
    res = model(image, text)
    print(res.shape)
