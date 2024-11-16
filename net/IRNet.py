import sys

sys.path.append("/home/suguilin/MMFS/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from typing import Callable
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils.utils import LayerNorm, Permute
from utils.modules import SS2D, CAB
from encoder.dense_mamba import DenseMambaBlock, TokenSwapMamba
from net.SparseMoEBlock import SparseMoEBlock
from net.FreqLearning import FreqLearning


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Taskprompt(nn.Module):
    def __init__(self, in_dim, atom_num=32, atom_dim=256):
        super(Taskprompt, self).__init__()
        hidden_dim = 64
        self.CondNet = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, 32, 1),
        )
        self.lastOut = nn.Linear(32, atom_num)
        self.act = nn.GELU()
        self.dictionary = nn.Parameter(
            torch.randn(atom_num, atom_dim), requires_grad=True
        )

    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        logits = F.softmax(out, -1)
        out = logits @ self.dictionary
        out = self.act(out)

        return out


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert (
            x.shape == y.shape
        ), "The shape of feature maps from image and features are not equal!"

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_head)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_head)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_head, h=h, w=w
        )

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x, 1, keepdim=True)[0]
        mean = torch.mean(x, 1, keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale = self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale


##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1, bias=False),
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale


##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out


##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FreModule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(
            dim, num_head=num_heads, bias=bias
        )
        self.channel_cross_h = Chanel_Cross_Attention(
            dim, num_head=num_heads, bias=bias
        )
        self.channel_cross_agg = Chanel_Cross_Attention(
            dim, num_head=num_heads, bias=bias
        )

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 8, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode="bilinear")

        high_feature, low_feature = self.fft(x)

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def shift(self, x):
        """shift FFT feature map to center"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h // n * threshold[i, 0, :, :]).int()
            w_ = (w // n * threshold[i, 1, :, :]).int()

            mask[i, :, h // 2 - h_ : h // 2 + h_, w // 2 - w_ : w // 2 + w_] = 1

        fft = torch.fft.fft2(x, norm="forward", dim=(-2, -1))
        fft = self.shift(fft)

        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm="forward", dim=(-2, -1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm="forward", dim=(-2, -1))
        low = torch.abs(low)

        return high, low


class IRNet(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 4, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super(IRNet, self).__init__()
        atom_dim = 256
        atom_num = 32
        ffn_expansion_factor = 2.66
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.block1 = FreModule(dim * 2**3, num_heads=heads[2], bias=bias)
        self.block2 = FreModule(dim * 2**2, num_heads=heads[2], bias=bias)
        self.block3 = FreModule(dim * 2**1, num_heads=heads[2], bias=bias)
        self.encoder_level1 = nn.Sequential(
            *[
                VMBlock(
                    hidden_dim=dim,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                VMBlock(
                    hidden_dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                VMBlock(
                    hidden_dim=int(dim * 2**2),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )
        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                VMBlock(
                    hidden_dim=int(dim * 2**3),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )
        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                VMBlock(
                    hidden_dim=int(dim * 2**2),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                VMBlock(
                    hidden_dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.up2_1 = Upsample(int(dim * 2**1))
        self.decoder_level1 = nn.Sequential(
            *[
                VMBlock(
                    hidden_dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.refinement = nn.Sequential(
            *[
                VMBlock(
                    hidden_dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )
        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.task_prompt = Taskprompt(in_dim=3, atom_num=atom_num, atom_dim=atom_dim)
        self.SMB1 = SparseMoEBlock(
            atom_dim=atom_dim, dim=dim, ffn_expansion_factor=ffn_expansion_factor
        )
        self.SMB2 = SparseMoEBlock(
            atom_dim=atom_dim,
            dim=int(dim * 2**1),
            ffn_expansion_factor=ffn_expansion_factor,
        )
        self.SMB3 = SparseMoEBlock(
            atom_dim=atom_dim,
            dim=int(dim * 2**2),
            ffn_expansion_factor=ffn_expansion_factor,
        )
        self.FreqL1 = FreqLearning(in_ch=dim * 2**2, num_experts=5)
        self.FreqL2 = FreqLearning(in_ch=dim * 2**1, num_experts=5)
        self.FreqL3 = FreqLearning(in_ch=dim * 2**1, num_experts=5)

    def forward(self, inp_img):
        task_prompt = self.task_prompt(inp_img)
        inp_enc_level1 = self.patch_embed(inp_img)

        task_harmonization_output1, loss_tmp = self.SMB1(inp_enc_level1, task_prompt)
        # print(task_harmonization_output1.shape)
        loss_importance = loss_tmp
        out_enc_level1 = self.encoder_level1(task_harmonization_output1)
        # print(out_enc_level1.shape)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        task_harmonization_output2, loss_tmp = self.SMB2(inp_enc_level2, task_prompt)
        # print(task_harmonization_output2.shape)
        loss_importance = loss_importance + loss_tmp
        out_enc_level2 = self.encoder_level2(task_harmonization_output2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        task_harmonization_output3, loss_tmp = self.SMB3(inp_enc_level3, task_prompt)
        loss_importance = loss_importance + loss_tmp
        out_enc_level3 = self.encoder_level3(task_harmonization_output3)
        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)
        latent = self.block1(inp_img, latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = self.FreqL1(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        out_dec_level3 = self.block2(inp_img, out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = self.FreqL2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.block3(inp_img, out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.FreqL3(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        if self.training:
            return out_dec_level1, loss_importance
        else:
            return out_dec_level1


if __name__ == "__main__":
    import cv2
    # from encoder.dense_mamba import DenseMambaBlock
    from decoder.seghead import DecoderHead

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 120, 160, 32).to(device)
    input_low = torch.randn(4, 3, 128, 128).to(device)
    input_sg = [torch.randn(2, 96, 64, 64).to(device), torch.randn(2, 192, 32, 32).to(device), torch.randn(2, 384, 16, 16).to(device), torch.randn(2, 768, 8, 8).to(device),]

    # GP_example = GatedPercept(in_channels=32, out_channels=64).to(device)
    # VM_example = VMBlock(hidden_dim=32).to(device)
    # IR_example = IRNet(
    #     inp_channels=3,
    #     out_channels=3,
    #     dim=48,
    #     num_blocks=[4, 4, 6, 8],
    #     num_refinement_blocks=1,
    #     heads=[1, 2, 4, 8],
    # ).to(device)
    # DM_example = DenseMambaBlock(dims=32).to(device)
    SD_example = DecoderHead(in_channels=[96, 192, 384, 768], num_classes=9, embed_dim=768).to(device)
    # GP_output = GP_example(input)
    # VM_output = VM_example(input)
    # IR_output, _ = IR_example(input_low)
    # DM_output, _ = DM_example(input)
    SD_output = SD_example(input_sg)

    # print(GP_output.shape)
    # print(VM_output.shape)
    # print(IR_output.shape)
    # print(DM_output.shape)
    print(SD_output.shape)
    # cv2.imwrite('outputD.png', IR_output.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
