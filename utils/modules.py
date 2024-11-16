from torch import nn
import torch
import math
from einops import repeat
import sys

sys.path.append("/home/suguilin/zigma/")
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.modules.mamba_simple import Mamba
from typing import Callable
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from utils.utils import LayerNorm
from einops import rearrange
from typing import List
from .MixBlock import LinearAttention
from .utils import Permute
from .NAFBlock import Global_Dynamics, Local_Dynamics, SimpleGate

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=1.5,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
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

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )  # (K=4, D, N)  Context Independent
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

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
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        """
        b: batch
        l: W * H
        k: 4
        b: batch
        r: rank?
        """
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

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)

        # context-aware params
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)

        # context-independent params
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

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
        )
        out_y = out_y.view(B, K, -1, L)
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

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)  # SSM

        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  # Merge
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        # self.attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
        #     nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_feat // squeeze_factor),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
            # nn.BatchNorm2d(num_feat),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f"Input to ChannelAttention - min: {x.min().item()}, max: {x.max().item()}")
        # y = self.attention(x)
        # return x * y
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        out = x * self.sigmoid(attn)
        # print(f"Output from ChannelAttention - min: {out.min().item()}, max: {out.max().item()}")
        return out


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            # nn.ReLU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        # print(f"Input to CAB - min: {x.min().item()}, max: {x.max().item()}")
        x = self.cab(x)
        # print(f"Output from CAB - min: {x.min().item()}, max: {x.max().item()}")
        return x


class Mlp_(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0.0, requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation."""

    def __init__(
        self, embed_dims, ffn_dims, kernel_size=3, act=nn.GELU(), ffn_drop=0.0
    ):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.ffn_dims = ffn_dims

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=self.ffn_dims, kernel_size=1
        )

        self.dwconv = nn.Conv2d(
            in_channels=self.ffn_dims,
            out_channels=self.ffn_dims,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.ffn_dims,
        )

        self.act = act
        self.fc2 = nn.Conv2d(
            in_channels=self.ffn_dims, out_channels=embed_dims, kernel_size=1
        )

        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.ffn_dims,  # C -> 1
            out_channels=1,
            kernel_size=1,
        )
        self.sigma = ElementScale(self.ffn_dims, init_value=1e-5, requires_grad=True)
        self.decompose_act = act

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel."""

    def __init__(
        self,
        embed_dims,
        dw_dilation=[
            1,
            2,
            3,
        ],
        channel_split=[
            1,
            3,
            4,
        ],
        act=nn.SiLU(),
    ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1
        )

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
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1
        )

        self.proj = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1
        )
        self.act_gate = act

    def forward(self, x):
        g = self.gate(x)
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0 : self.embed_dims_0 + self.embed_dims_1, ...]
        )
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2 :, ...])
        x = torch.cat([x_0[:, : self.embed_dims_0, ...], x_1, x_2], dim=1)
        v = self.PW_conv(x)
        x = self.proj(self.act_gate(g) * self.act_gate(v))
        return x


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=scale[i],
                padding=scale[i] // 2,
                groups=channels,
            )
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class MS_FFN(nn.Module):  ### MS-FFN
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act=nn.GELU(),
        drop=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            act,
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = act
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvGate(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, num_experts, kernel_size=1),
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.Conv2d(dim, num_experts, kernel_size=1),
        )
        self.gate = nn.Linear(num_experts * 2, num_experts)

    def forward(self, x):
        feat1 = F.adaptive_avg_pool2d(self.scale1(x), 1)
        feat2 = F.adaptive_avg_pool2d(self.scale2(x), 1)
        gate_logits = self.gate(torch.cat([feat1, feat2], dim=1).reshape(x.size(0), -1))
        return gate_logits


class Router(nn.Module):
    def __init__(self, embed_size, num_out_path):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.conv_pool = nn.Linear(embed_size*36*36, embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.ReLU(True),
            nn.Linear(embed_size, num_out_path),
        )
        self.init_weights()

    def init_weights(self):
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):
        # x = x.reshape(x.shape[0],-1)
        avg_out = self.avgpool(x)
        max_out = self.maxpool(x)
        f_out = torch.cat([max_out, avg_out], 1)
        x = f_out.contiguous().view(f_out.size(0), -1)
        x = self.mlp(x)
        # soft_g = F.relu(torch.tanh(x)).reshape(x.shape[0], -1)
        soft_g = x
        return soft_g


"""
class MoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs_raw: torch.Tensor):
        ishape = inputs_raw.shape
        # inputs = inputs_raw.reshape(ishape[0], -1)
        inputs = inputs_raw.permute(0, 3, 1, 2)
        n, c, h, w = inputs.shape
        gate_logits = self.gate(inputs)
        # print(gate_logits)
        # l_aud
        gates = F.softmax(gate_logits, dim=1)
        # print(gates)
        indices1_s = torch.argmax(gates, dim=1)
        num_experts = int(gates.shape[1])
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)

        # Compute l_aux
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.float(), dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts

        weights, selected_experts = torch.topk(gates, self.num_experts_per_tok)
        # print(selected_experts)
        # weights = F.softmax(weights, dim=-1)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        # print(weights)
        # results = torch.zeros_like(inputs_raw.view(ishape[0], -1, ishape[-1]))
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            # print(selected_experts == i)
            batch_idx, nth_expert = torch.where(selected_experts == i)
            # print("batch_idx:", batch_idx)
            # print("nth_expert:", nth_expert)
            if len(batch_idx) == 0:
                continue
            else:
                for b, i in zip(batch_idx, nth_expert):
                    # print(inputs_raw[b:b + 1].view(1, -1, ishape[-1]).shape)
                    # results[b:b+1] += weights[b:b+1, i:i+1].view(-1, 1, 1) * expert(inputs_raw[b:b+1].view(1, -1, ishape[-1]), h, w)
                    results[b:b+1] += weights[b:b+1, i:i+1].view(-1, 1, 1) * expert(inputs[b:b+1])

        # results_out = results.view(ishape)
        results_out = results.permute(0, 2, 3, 1)
        return results_out, l_aux
"""


class MoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        num_experts: int,
        num_experts_per_tok: int,
        kth_experts: int,
        beta: float,
    ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.kth_experts = kth_experts
        self.beta = beta

    def forward(self, inputs_raw: torch.Tensor):
        ishape = inputs_raw.shape
        # inputs = inputs_raw.reshape(ishape[0], -1)
        inputs = inputs_raw.permute(0, 3, 1, 2)
        n, c, h, w = inputs.shape
        gate_logits = self.gate(inputs)
        gates = F.softmax(gate_logits, dim=1)  # (batch_size, num_experts)
        # print(gates)

        # Top-2 routing
        weights, selected_experts = torch.topk(gates, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        # print(weights)

        # Rank-k routing
        rank_k_value, rank_k_index = torch.kthvalue(gates, self.kth_experts, dim=1)

        indices1_s = torch.argmax(
            gates, dim=1
        )  # get the chosen expert id for every token
        num_experts = int(gates.shape[1])
        mask1 = F.one_hot(
            indices1_s, num_classes=num_experts
        )  # (batch_size, num_experts)

        # Compute l_aux
        # entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=1).mean()
        me = torch.mean(
            gates, dim=0
        )  # (num_experts, ) get the average weight (probability) of each expert being selected in the entire batch
        ce = torch.mean(
            mask1.float(), dim=0
        )  # get the frequency of each expert being selected in the actual selection
        # compare the probability distribution of each expert being selected (me) with the actual selection (ce)
        # measures how well each expert's actual usage matches their expected usage
        l_aux = torch.mean(me * ce) * num_experts * num_experts  # + entropy

        # Compute V_valid subset
        # max_top2 = torch.max(top2_values, dim=1, keepdim=True)[0]
        # v_valid_mask = (top2_values >= torch.log(self.alpha) + max_top2).float()
        results_strong = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if len(batch_idx) == 0:
                continue
            else:
                for b, i in zip(batch_idx, nth_expert):
                    results_strong[b : b + 1] += weights[b : b + 1, i : i + 1].view(
                        -1, 1, 1
                    ) * expert(inputs[b : b + 1])

        # results_weak = torch.zeros_like(inputs)
        # for b, i in enumerate(rank_k_index):
        #     results_weak[b:b+1] += self.experts[i](inputs[b:b+1])
        results_strong = results_strong.permute(0, 2, 3, 1)

        # results_weak = results_weak.permute(0, 2, 3, 1)
        #
        results_out = results_strong  # (1 + self.beta) * results_strong - self.beta * results_weak
        return results_out, l_aux


class VSSBlock(nn.Module):
    def __init__(
        self,
        num_experts: int = 4,
        topk: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.dim = hidden_dim
        self.num_heads = num_heads
        self.topk = topk
        # 2024-08-18 简单变换
        # self.preprocess = nn.Sequential(
        #     norm_layer(hidden_dim),
        #     Permute(0, 3, 1, 2),
        #     # nn.Conv2d(hidden_dim, hidden_dim, 1),
        #     nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 1, 1, groups=hidden_dim),
        #     SimpleGate(),
        #     Permute(0, 2, 3, 1),
        # )

        # self.sptial = nn.Sequential(
        #     norm_layer(hidden_dim),
        #     Permute(0, 3, 1, 2),
        #     MultiOrderGatedAggregation(embed_dims=hidden_dim),
        #     Permute(0, 2, 3, 1),
        # )

        # 局部全局分支特征权重调制
        # self.gmm = Global_Dynamics(hidden_dim)
        # self.lmm = Local_Dynamics(hidden_dim)

        # self.alpha = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, 1),
        #     nn.Sigmoid(),
        # )

        # 局部分支
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.conv1 = nn.Conv2d(self.dim, self.dim, 1)
        # self.conv2 = nn.Conv2d(self.dim, self.dim, 1)

        # self.local = ChannelAggregationFFN(embed_dims=hidden_dim, ffn_dims=int(hidden_dim*4))

        # 全局分支
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(d_model=hidden_dim, dropout=drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(
            torch.ones((1, 1, 1, hidden_dim)), requires_grad=True
        )

        # 合并
        # self.proj = nn.Sequential(
        #     nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
        #     nn.GELU(),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1),
        #     nn.GELU(),
        #     nn.BatchNorm2d(hidden_dim // 8),
        #     nn.Conv2d(hidden_dim // 8, hidden_dim, kernel_size=1),
        #     nn.BatchNorm2d(hidden_dim),
        #     )
        # self.gate = ConvGate(hidden_dim, num_experts)
        # self.gate = Router(hidden_dim, num_experts)

        # self.conv_blk = CAB(hidden_dim)
        self.ms_ffn = MS_FFN(in_features=hidden_dim, hidden_features=hidden_dim*2)
        self.norm2 = norm_layer(hidden_dim)
        self.skip_scale2 = nn.Parameter(
            torch.ones((1, 1, 1, hidden_dim)), requires_grad=True
        )
        # self.norm3 = norm_layer(hidden_dim)
        # self.mlp = Mlp_(in_features=hidden_dim, hidden_features=hidden_dim * 4, act_layer=nn.SiLU)

        # self.experts = [
        #     # LinearAttention(
        #     #     dim=self.dim,
        #     #     num_heads=self.num_heads,
        #     #     qkv_bias=True,
        #     # )
        #     # MS_FFN(
        #     #     in_features=hidden_dim,
        #     #     hidden_features=int(hidden_dim*4),
        #     #     )
        #     Mlp(
        #         in_features=hidden_dim,
        #         out_fratures=hidden_dim,
        #         ffn_expansion_factor=2,
        #     )
        #     for _ in range(self.num_experts)
        # ]

        # self.moe_layer = MoeLayer(
        #     experts=self.experts,
        #     gate=self.gate,
        #     num_experts=self.num_experts,
        #     num_experts_per_tok=self.topk,
        #     kth_experts=2,
        #     beta=0.5,
        # )

    # 局部分支
    def simpleReplace(self, x):
        return self.conv2(x * self.conv1(self.pool(x)))

    # def initialize_modules(self, x):
    #     B, H, W, C = x.shape
    #     self.experts = [
    #         LinearAttention(
    #             dim=self.dim,
    #             input_resolution=(H, W),
    #             num_heads=self.num_heads,
    #             qkv_bias=True,
    #         ).to(x.device)
    #         for _ in range(self.num_experts)
    #     ]

    #     self.moe_layer = MoeLayer(
    #         experts=self.experts,
    #         gate=self.gate,
    #         num_experts=self.num_experts,
    #         num_experts_per_tok=self.topk,
    #     )

    def forward(self, input: torch.Tensor, loss_aux=0):
        # B, H, W, C = input.shape
        # print("input:", input.shape)
        # self.initialize_modules(input)
        # preprocess
        # pre = self.preprocess(input)
        pre = input
        # pre = self.sptial(input)
        # print("preprocess:", pre.shape)
        # local branch
        # y = self.simpleReplace(pre.permute(0, 3, 1, 2))
        # y = self.local(pre.permute(0, 3, 1, 2))
        # gobal branch
        x = self.norm(pre)
        # x = self.norm(input)
        # 尝试调试权重后再残差连接
        x = self.drop_path(self.op(x))
        # weight modulation

        # x, y = self.gmm(x.permute(0, 3, 1, 2), y)
        # x, y = self.lmm(x, y)
        # element-wise sum
        # alpha = self.alpha(torch.cat([x, y.permute(0, 2, 3, 1)], dim=3))
        # x = x +  alpha * y.permute(0, 2, 3, 1)

        # residual connection
        x = input * self.skip_scale + x  # .permute(0, 2, 3, 1)
        # x = self.proj(x) + x

        # x = input * self.skip_scale + self.drop_path(self.op(x))

        # 使用moe，参考BlackMamba: 'Mixture of Experts for State-Space Models'思路，多了辅助损失
        # print("aftet merged:", x.shape)
        # x = x * self.skip_scale2 + self.conv_blk(self.norm2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale2 + self.ms_ffn(self.norm2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # res, l_aux = self.moe_layer(self.norm3(x))
        # 这点的残差连接也需要检验是否有存在的必要
        # x = x * self.skip_scale2 + res
        # loss_aux += l_aux
        # 尝试moe先注释掉，2024-08-19,22:19
        # x = x * self.skip_scale2 + self.conv_blk(self.norm2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        # print(f"After conv_blk - min: {x.min().item()}, max: {x.max().item()}")
        # x = x + self.drop_path(self.mlp(self.norm3(x)))
        # x = input + self.drop_path(self.op(self.norm(input)))
        return x, loss_aux  # l_aux


#######################################################################################################


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(x.shape)
        y = self.global_pool(x).view(b, c)
        # print(y.shape)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        # print(y.shape)
        return x * y.expand_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        context = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        out = self.out(context)
        return out


class SEBlock1D(nn.Module):
    def __init__(self, in_channels, reduction):
        super(SEBlock1D, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, N, C)
        B, N, C = x.shape
        y = x.mean(dim=1)  # (B, C)
        y = self.fc1(y)  # (B, C // reduction)
        y = self.relu(y)
        y = self.fc2(y)  # (B, C)
        y = self.sigmoid(y).unsqueeze(1)  # (B, 1, C)
        return x * y


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5
        # print(embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        # print(x.shape)
        Q = self.query(x)  # (B, N, C)

        K = self.key(x)  # (B, N, C)
        V = self.value(x)  # (B, N, C)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, N, N)
        attn = F.softmax(scores, dim=-1)  # (B, N, N)
        context = torch.matmul(attn, V)  # (B, N, C)
        return context


class LocalAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(LocalAttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                3,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels // 2,
                in_channels,
                3,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.Sigmoid(),
        )

    def forward(self, x_rgb, x_ir):
        attn_rgb = self.attention(x_rgb)
        attn_ir = self.attention(x_ir)
        fused = x_rgb * attn_rgb + x_ir * attn_ir
        return fused


class LocalAttention(nn.Module):
    def __init__(self, in_channels):
        super(LocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.conv1(x)
        proj_key = self.conv2(x)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.conv3(x)

        out = torch.bmm(proj_value, attention)

        out = self.gamma * out + x
        return out


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        B, C, N = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((B, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        out_x1 = out_x1.permute(0, 2, 1)
        out_x2 = out_x2.permute(0, 2, 1)
        return out_x1, out_x2


class TokenSwapMamba(nn.Module):
    def __init__(self, dim, reduction):
        super(TokenSwapMamba, self).__init__()
        # self.I1attention = SelfAttention(dim)
        # self.I2attention = SelfAttention(dim)
        self.I1encoder = Mamba(dim, bimamba_type=None)
        self.I2encoder = Mamba(dim, bimamba_type=None)
        self.norm1 = LayerNorm(dim, "with_bias")
        self.norm2 = LayerNorm(dim, "with_bias")
        self.ChannelExchange = ChannelExchange(p=2)
        self.se_block = SEBlock1D(dim, reduction)

    def forward(self, I1, I2):

        I1_residual = I1
        I2_residual = I2

        # I1 = self.I1attention(I1)
        # I2 = self.I2attention(I2)

        I1 = self.norm1(I1)  # + I1_residual
        I2 = self.norm2(I2)  # + I2_residual
        # print(I1.shape)
        # B, N, C = I1.shape

        I1_swap, I2_swap = self.ChannelExchange(I1, I2)
        I1_swap = self.se_block(I1_swap)
        I2_swap = self.se_block(I2_swap)

        I1_swap = self.I1encoder(I1_swap) + I1_residual
        I2_swap = self.I2encoder(I2_swap) + I2_residual
        return I1_swap, I2_swap


class M3(nn.Module):
    def __init__(self, dim):
        super(M3, self).__init__()
        self.multi_modal_mamba_block = Mamba(dim, bimamba_type="m3")
        self.norm1 = LayerNorm(dim, "with_bias")
        self.norm2 = LayerNorm(dim, "with_bias")
        self.norm3 = LayerNorm(dim, "with_bias")

        # self.attention = SelfAttention(dim)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.chan_att = ChannelAttention1(dim, 4)
        self.sp_att = SpatialAttention(5)

    def forward(self, I1, fusion_resi, I2, fusion, test_h, test_w):
        fusion_resi = fusion + fusion_resi
        # fusion = self.attention(fusion_resi)
        fusion = self.norm1(fusion_resi)
        I2 = self.norm2(I2 * self.sp_att(I2, test_h, test_w))
        I1 = self.norm3(I1 * self.sp_att(I1, test_h, test_w))

        global_f = self.multi_modal_mamba_block(
            self.norm1(fusion), extra_emb1=self.norm2(I2), extra_emb2=self.norm3(I1)
        )

        B, HW, C = global_f.shape
        fusion = global_f.transpose(1, 2).view(B, C, test_h, test_w)
        fusion = self.chan_att(fusion) * fusion
        fusion = (self.dwconv(fusion) + fusion).flatten(2).transpose(1, 2)
        return fusion, fusion_resi


class ChannelAttention1(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = x * self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, test_h, test_w):
        B, N, C = x.shape
        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=2)
        out = out.permute(0, 2, 1).view(B, 2, test_h, test_w)
        out = self.conv1(out)
        return self.sigmoid(out).flatten(2).transpose(1, 2)


#######################################################################################
"""
u: (B D L)
Delta: (B D L) Aware Private
A: (D H) Independent Common
B: (B H L) Aware Private
C: (B H L) Aware Private
D: (D) Independent Common
h'(t) = Ah(t) + Bx(t)
y(t)  = Ch(t) + Dx(t)
"""


class FusionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=32,
        d_conv=3,
        expand=1.5,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
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

        # x represent infrared images
        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        # y represent visible images
        self.y_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.y_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.y_proj], dim=0)
        )  # (K=4, N, inner)
        del self.y_proj

        # shared weight of dt_projs
        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )  # (K=4, D, N) Context Independent
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.mark_proj = nn.Parameter(
            nn.Linear(self.d_inner * 2, self.d_inner, bias=False).weight
        )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

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
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def recover_flip(self, out_y, B, H, W, L):
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
        return out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

    def forward_core(self, x: torch.Tensor, y: torch.Tensor):
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
        y_hwwh = torch.stack(
            [
                y.view(B, -1, L),
                torch.transpose(y, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        ys = torch.cat([y_hwwh, torch.flip(y_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        x_dts, x_Bs, x_Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        x_dts = torch.einsum(
            "b k r l, k d r -> b k d l", x_dts.view(B, K, -1, L), self.dt_projs_weight
        )

        y_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", ys.view(B, K, -1, L), self.y_proj_weight
        )
        y_dts, y_Bs, y_Cs = torch.split(
            y_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        y_dts = torch.einsum(
            "b k r l, k d r -> b k d l", y_dts.view(B, K, -1, L), self.dt_projs_weight
        )

        mark = torch.einsum(
            "b k C l, c C -> b k c l", torch.cat([xs, ys], dim=2), self.mark_proj
        )

        xs = xs + mark
        ys = ys + mark

        xs = (
            torch.cat(
                [xs.unsqueeze(3).transpose(-1, -2), ys.unsqueeze(3).transpose(-1, -2)],
                dim=-1,
            )
            .contiguous()
            .view(B, K, C, L * 2)
        )
        dts = (
            torch.cat(
                [
                    x_dts.unsqueeze(3).transpose(-1, -2),
                    y_dts.unsqueeze(3).transpose(-1, -2),
                ],
                dim=-1,
            )
            .contiguous()
            .view(B, K, y_dts.shape[2], L * 2)
        )
        Bs = (
            torch.cat(
                [
                    x_Bs.unsqueeze(3).transpose(-1, -2),
                    y_Bs.unsqueeze(3).transpose(-1, -2),
                ],
                dim=-1,
            )
            .contiguous()
            .view(B, K, y_Bs.shape[2], L * 2)
        )
        Cs = (
            torch.cat(
                [
                    x_Cs.unsqueeze(3).transpose(-1, -2),
                    y_Cs.unsqueeze(3).transpose(-1, -2),
                ],
                dim=-1,
            )
            .contiguous()
            .view(B, K, y_Cs.shape[2], L * 2)
        )

        xs = xs.float().view(B, -1, L * 2)  # (b, k * d, l)

        # context-aware params
        dts = dts.contiguous().float().view(B, -1, L * 2)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L * 2)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L * 2)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)

        # context-independent params
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

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
        )
        out_y = out_y.view(B, K, -1, L * 2)

        return self.recover_flip(
            out_y[:, :, :, 0::2], B=B, H=H, W=W, L=L
        ), self.recover_flip(out_y[:, :, :, 1::2], B=B, H=H, W=W, L=L)

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        yz = self.in_proj(y)
        x, z_x = xz.chunk(2, dim=-1)
        y, z_y = yz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))
        y = self.act(self.conv2d(y))

        x, y = self.forward_core(x, y)  # Cross SS2D

        x = torch.transpose(x, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        x = self.out_norm(x)

        y = y * F.silu(z_y)
        x = x * F.silu(z_x)
        out_y = self.out_proj(y)
        out_x = self.out_proj(x)

        if self.dropout is not None:
            out_y = self.dropout(out_y)
            out_x = self.dropout(out_x)
        return out_x, out_y


########################################################################################
"""参考Equivariant Multi-Modality imAge fusion (EMMA)"""


class Restormer_CNN_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Restormer_CNN_block, self).__init__()
        self.embed = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads=8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN = nn.Conv2d(
            out_dim * 2,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.embed(x)
        x1 = self.GlobalFeature(x)
        x2 = self.LocalFeature(x)
        out = self.FFN(torch.cat((x1, x2), 1))
        return out


class GlobalFeatureExtraction(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor=1.0,
        qkv_bias=False,
    ):
        super(GlobalFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, "WithBias")
        self.attn = AttentionBase(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, "WithBias")
        self.mlp = Mlp(
            in_features=dim,
            out_fratures=dim,
            ffn_expansion_factor=ffn_expansion_factor,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LocalFeatureExtraction(nn.Module):
    def __init__(
        self,
        dim=64,
        num_blocks=2,
    ):
        super(LocalFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(
            *[ResBlock(dim, dim) for i in range(num_blocks)]
        )

    def forward(self, x):
        return self.Extraction(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
        )

    def forward(self, x):
        out = self.conv(x)
        return out + x


class AttentionBase(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
    ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, out_fratures, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias
        )

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
            padding_mode="reflect",
        )

        self.project_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# Enhanced Spatial Attention
class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
