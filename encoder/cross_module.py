import sys

sys.path.append("/home/suguilin/MMFS/")
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
from utils.utils import *
from utils.modules import Mlp, CAB, LayerNorm
from encoder.dense_mamba import DenseMambaBlock, CIM
from net.IRNet import VMBlock
from net.FreqLearning import ResMoEBlock
from fusion.AttentionFusion import DynamicFusionModule

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
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

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
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)  # SSM
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 2 // reduction, self.dim * 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg_v = self.avg_pool(x).view(B, self.dim * 2) #B  2C
        max_v = self.max_pool(x).view(B, self.dim * 2)
        
        avg_se = self.mlp(avg_v).view(B, self.dim * 2, 1)
        max_se = self.mlp(max_v).view(B, self.dim * 2, 1)
        
        Stat_out = self.sigmoid(avg_se + max_se).view(B, self.dim * 2, 1)
        channel_weights = Stat_out.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SpatialAttention, self).__init__()
        self.mlp = nn.Sequential(
                    nn.Conv2d(4, 4*reduction, kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4*reduction, 2, kernel_size), 
                    nn.Sigmoid())
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True) #B  1  H  W
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)  #B  1  H  W
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True) #B  1  H  W
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)  #B  1  H  W                
        x_cat = torch.cat((x1_mean_out, x1_max_out, x2_mean_out, x2_max_out), dim=1) # B 4 H W
        spatial_weights = self.mlp(x_cat).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights

class MixAttention(nn.Module):
    def __init__(self, dim, reduction=1):
        super(MixAttention, self).__init__()
        self.dim = dim
        self.ca_gate = ChannelAttention(self.dim) 
        self.sa_gate = SpatialAttention(reduction=4)
        self.downsample = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        ca_out = self.ca_gate(x1, x2) # 2 B C 1 1
        sa_out = self.sa_gate(x1, x2)  # 2 B 1 H W
        mixatt_out = ca_out.mul(sa_out)  # 2 B C H W
        out_x1 = x1 + mixatt_out[1] * x2
        out_x2 = x2 + mixatt_out[0] * x1
        out = torch.cat([out_x1, out_x2], dim=1)
        out = self.relu(self.downsample(out))  # B C H W
        mixatt_out = out #F.interpolate(out, [120, 160], mode='nearest')
        # mixatt_out = out
        return mixatt_out

class CrossBlock(nn.Module):
    def __init__(
        self,
        modals: int = 2,
        hidden_dim: int = 0,
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
        self.modals = modals

        self.norm = nn.ModuleList()
        self.op = nn.ModuleList()
        self.in_proj = nn.ModuleList()
        self.out_proj = nn.ModuleList()
        for _ in range(self.modals):
            self.norm.append(norm_layer(hidden_dim))
            self.in_proj.append(
                nn.Linear(hidden_dim, int(hidden_dim * ssm_ratio), bias=False)
            )
            self.out_proj.append(
                nn.Linear(int(hidden_dim * ssm_ratio), hidden_dim, bias=False)
            )

            self.op.append(
                SS2D(
                    d_model=hidden_dim,
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

        d_model_rate = self.modals

        self.norm_share = norm_layer(hidden_dim * d_model_rate)
        self.op_share = SS2D(
            d_model=hidden_dim,
            d_model_rate=1,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs,
        )

        self.drop_path = DropPath(drop_path)
        self.downsample_rgb = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.downsample_ir = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        # self.fusion_mamba = Mamba(hidden_dim, bimamba_type="m3")
        # self.fusion_proj = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False)
        self.fusion_proj = MixAttention(hidden_dim)

    def forward(self, rgb, ir, H, W):
        B, C= rgb.shape[0], rgb.shape[1]
        rgb = rgb.permute(0, 2, 3, 1)
        ir = ir.permute(0, 2, 3, 1)
        rgb_ir = torch.cat([rgb, ir], dim=-1)
        rgb_ir = self.op_share(self.norm_share(rgb_ir))

        rgb_st = rgb
        ir_st = ir

        rgb_in = self.norm[0](rgb)
        rgb_op = self.op[0](rgb_in)
        g_vi = F.sigmoid(self.in_proj[0](rgb_in))
        rgb = g_vi * rgb_ir + (1 - g_vi) * rgb_op
        rgb = self.out_proj[0](rgb)

        ir_in = self.norm[1](ir)
        ir_op = self.op[1](ir_in)
        g_ir = F.sigmoid(self.in_proj[1](ir_in))
        ir = g_ir * rgb_ir + (1 - g_ir) * ir_op
        ir = self.out_proj[1](ir)

        rgb = rgb_st + self.drop_path(rgb)
        ir = ir_st + self.drop_path(ir)
        
        rgb = self.downsample_rgb(rgb.permute(0, 3, 1, 2))
        ir = self.downsample_ir(ir.permute(0, 3, 1, 2))
        rgb = self.downsample_rgb(rgb)
        ir = self.downsample_ir(ir)


        # fused = self.fusion_proj(rgb.permute(0, 3, 1, 2), ir.permute(0, 3, 1, 2))
        fused = self.fusion_proj(rgb, ir)
        # fused = self.fusion_proj(torch.cat([rgb, ir], dim=-1).permute(0, 3, 1, 2))
        # fused = self.fusion_mamba(rgb_ir.view(B, -1, C), rgb.view(B, -1, C), ir.view(B, -1, C))
        fused = F.interpolate(fused, [H, W], mode='nearest')
        return fused #rgb, ir

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class FFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.
    """

    def __init__(self,
                 embed_dims,
                 ffn_dims,
                 kernel_size=3,
                 act=nn.GELU(),
                 ffn_drop=0.):
        super(FFN, self).__init__()

        self.embed_dims = embed_dims
        self.ffn_dims = ffn_dims

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.ffn_dims,
            kernel_size=1
        )
        
        self.dwconv = nn.Conv2d(
            in_channels=self.ffn_dims,
            out_channels=self.ffn_dims,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.ffn_dims
        )
        
        self.act = act
        self.fc2 = nn.Conv2d(
            in_channels=self.ffn_dims,
            out_channels=embed_dims,
            kernel_size=1
        )

        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.ffn_dims,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.ffn_dims, init_value=1e-5, requires_grad=True
        )
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


class CrossFusionBlock(nn.Module):
    def __init__(
        self,
        modals: int = 2,
        hidden_dim: int = 0,
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
        '''
        self.modals = modals
        d_model_rate = self.modals
        
        self.norm1 = nn.ModuleList()
        self.FFN = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.op = nn.ModuleList()
        self.norm3 = nn.ModuleList()
        self.conv_blk = nn.ModuleList()

        for _ in range(self.modals):
            # self.norm1.append(norm_layer(hidden_dim))            
            # self.FFN.append(
            #     nn.Sequential(
            #         Permute(0, 3, 1, 2),
            #         Mlp(in_features=hidden_dim, out_fratures=hidden_dim, ffn_expansion_factor=4),
            #         Permute(0, 2, 3, 1),
            #     )
            # )

            self.norm2.append(norm_layer(hidden_dim))
            self.op.append(
                # SS2D(
                #     d_model=hidden_dim,
                #     d_model_rate=1,
                #     dropout=attn_drop_rate,
                #     d_state=d_state,
                #     ssm_ratio=ssm_ratio,
                #     dt_rank=dt_rank,
                #     shared_ssm=shared_ssm,
                #     softmax_version=softmax_version,
                #     **kwargs,
                # )
                DenseMambaBlock(dims=hidden_dim),
                
            )

            self.norm3.append(norm_layer(hidden_dim))
            self.conv_blk.append(
                nn.Sequential(
                    Permute(0, 3, 1, 2),
                    CAB(hidden_dim),
                    Permute(0, 2, 3, 1),
                )    
            )
        '''
        self.cim = CIM(dim=hidden_dim)
        self.dynamic_fusion = DynamicFusionModule(embed_size=hidden_dim)
        self.resvm = nn.ModuleList([
            nn.Sequential(
                ResMoEBlock(
                    in_ch=hidden_dim * 2,
                    num_experts=4,
                    use_shuffle=True,
                    lr_space=1,
                    topk=2,
                    recursive=2,
                ),
                VMBlock(
                    hidden_dim=hidden_dim * 2,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="WithBias",
                )
            ) for _ in range(3) 
        ])
        '''
        self.norm = norm_layer(hidden_dim * d_model_rate)  
        self.mlp = nn.Sequential(
            Permute(0, 3, 1, 2),
            Mlp(in_features=hidden_dim * 2, out_fratures=hidden_dim, ffn_expansion_factor=4),
            Permute(0, 2, 3, 1),
        )
        self.norm_share = norm_layer(hidden_dim)            
        self.op_share = SS2D(
            d_model=hidden_dim,
            d_model_rate=d_model_rate,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs,
        )
        # self.conv_blk_share = nn.Sequential(
        #             Permute(0, 3, 1, 2),
        #             CAB(hidden_dim),
        #             Permute(0, 2, 3, 1),
        #         )    
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # self.conv1 = nn.Sequential(
        #     Permute(0, 3, 1, 2),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=1),
        #     Permute(0, 2, 3, 1),
        # )
        # self.conv2 = nn.Sequential(
        #     Permute(0, 3, 1, 2),
        #     nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
        #     Permute(0, 2, 3, 1),
        # )
        '''
        self.norm_layer = LayerNorm(hidden_dim*2, "WithBias") # norm_layer(hidden_dim) #nn.BatchNorm2d(hidden_dim)
        self.dwconv = nn.Sequential(
            # Permute(0, 3, 1, 2),
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True, groups=hidden_dim, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
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
        '''
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)

        x = torch.cat([x1, x2], dim=-1)
        x = self.mlp(self.norm(x))
        x = self.op_share(self.norm_share(x))

        x1_short = x1
        x2_short = x2

        g = self.linear(x) # # F.sigmoid(x) self.conv_blk_share(x)
        # x1 = self.FFN[0](self.norm1[0](x1))
        x1, _ = self.op[0](x1) # self.op[0](self.norm2[0](x1))
        x2, _ = self.op[1](x2)
        x1, x2 = self.cim(x1.permute(0, 3, 1, 2), x2.permute(0, 3, 1, 2))
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = self.conv_blk[0](self.norm3[0](x1))

        # x2 = self.FFN[1](self.norm1[1](x2))
        # x2 = self.op[1](self.norm2[1](x2))
        x2 = self.conv_blk[1](self.norm3[1](x2))

        x = g * x1 + g * x2
        x = self.proj(x) + x1_short + x2_short
        # residual = self.conv2(self.conv1(x_short) * x_short)
        '''
        x1, x2 = self.cim(x1, x2)
        # x = self.dynamic_fusion(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        for i in range(3): 
            x = self.resvm[i](x)
        x = self.dwconv(self.norm_layer(x))
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CrossBlock(
    #     modals=2,
    #     hidden_dim=32,
    #     drop_path=0.1,
    #     shared_ssm=False,
    #     d_state=16,
    #     ssm_ratio=2.0,
    #     dt_rank="auto",
    #     mlp_ratio=4,
    #     norm_layer=nn.LayerNorm,
    # ).to(device)
    model = CrossFusionBlock(
        modals=2,
        hidden_dim=96,
        drop_path=0.1,
        shared_ssm=False,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
    ).to(device)
    rgb = torch.randn(1, 96, 120, 160).to(device)
    ir = torch.randn(1, 96, 120, 160).to(device)
    # rgb, ir = model(rgb.to(device), ir.to(device))
    fused = model(rgb, ir)
    # print(fused.shape)
    # print(rgb.shape)
    # print(ir.shape)
