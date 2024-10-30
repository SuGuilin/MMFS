import sys

sys.path.append("/home/suguilin/zigma/")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from timm.models.layers import DropPath
import math

class Mlp(nn.Module):
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

'''
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding."""

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        self.shape = shape
        channel_dims, feature_dim = self.shape[:-1], self.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat(
            [
                t.unsqueeze(-1) * theta_ks
                for t in torch.meshgrid(
                    [torch.arange(d) for d in channel_dims], indexing="ij"
                )
            ],
            dim=-1,
        )

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer("rotations", rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)
'''
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding."""

    def __init__(self, dim, base=10000):
        super(RoPE, self).__init__()
        self.base = base
        self.dim = dim

    def forward(self, x):
        shape = x.shape[-3:-1]  # 获取最后两个维度作为 shape
        channel_dims = shape
        feature_dim = self.dim
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (self.base ** (torch.arange(k_max) / k_max))
        angles = torch.cat(
            [
                t.unsqueeze(-1) * theta_ks
                for t in torch.meshgrid(
                    [torch.arange(d) for d in channel_dims], indexing="ij"
                )
            ],
            dim=-1,
        ).to(x.device)

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
    r"""Linear Attention with LePE and RoPE. 旋转位置嵌入和

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.rope = RoPE(shape=(self.input_resolution[0], self.input_resolution[1], dim))
        self.rope = RoPE(dim=dim)

    def forward(self, x, h, w):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        # h = int(3 / 4 * ((4 / 3 * n) ** (1/2)))
        # w = int((4 / 3 * n) ** (1/2))

        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = (
            self.rope(q.reshape(b, h, w, c))
            .reshape(b, n, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        k_rope = (
            self.rope(k.reshape(b, h, w, c))
            .reshape(b, n, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n**-0.5)) @ (v * (n**-0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"

class MixBlock(nn.Module):
    def __init__(
        self,   
        input_resolution,
        num_heads,
        d_model,
        d_state=16,
        d_conv=4,
        ssm_ratio=3,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        dtype=None,
        device=None,
        ################
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        mlp_ratio=4., 
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner, bias=bias, **factory_kwargs
        )
        self.d_inner = int(self.expand * self.d_model / 3 * 2)
        self.x_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs
        )
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        #######################################
        self.dim = d_model
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        # self.conv_in = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)
        self.norm_in = norm_layer(self.dim)
        # self.proj_in = nn.Linear(self.dim, self.dim)
        self.dw_conv = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)
        self.act = nn.SiLU()
        self.linear_attn = LinearAttention(
            dim=self.dim,
            input_resolution=self.input_resolution,
            num_heads=self.num_heads,
            qkv_bias=True,
        )
        self.linear_attn.input_resolution = self.input_resolution

        self.proj_out = nn.Linear(self.dim, self.dim)

        self.out = nn.Linear(self.dim*2, self.dim)

        self.norm_mlp = norm_layer(self.dim)
        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim*mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, hidden_states):
        H, W = self.input_resolution
        B, seqlen, C = hidden_states.shape

        shortcut = hidden_states
        hidden_states = self.norm_in(hidden_states)
        xzw = self.in_proj(hidden_states)
        xzw = rearrange(xzw, "b l d -> b d l")
        x, z, w = xzw.chunk(3, dim=1)

        lin_x = self.act(self.dw_conv(w.view(B, C, H, W))).permute(0, 2, 3, 1).view(B, seqlen, C)
        # print(lin_x.shape)
        lin_x = self.linear_attn(lin_x)

        A = -torch.exp(self.A_log.float())
        x = F.silu(
            F.conv1d(
                input=x,
                weight=self.conv1d_x.weight,
                bias=self.conv1d_x.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )
        z = F.silu(
            F.conv1d(
                input=z,
                weight=self.conv1d_z.weight,
                bias=self.conv1d_z.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")

        out1 = self.out_proj(y)
        z = rearrange(z, "b d l -> b l d")
        out2 = self.proj_out(lin_x * z)
        out = self.out(torch.cat([out1, out2], dim=-1))
        out = shortcut + self.drop_path(out)
        out = out + self.drop_path(self.mlp(self.norm_mlp(out)))
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 120 * 160, 32).to(device)
    model = MixBlock(
        input_resolution=(120, 160),
        num_heads=4,
        d_model=32,
        d_state=16,
        d_conv=4,
        ssm_ratio=3,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        dtype=None,
        # device=device,
    ).to(device)
    output = model(input)
    print(output.shape)
