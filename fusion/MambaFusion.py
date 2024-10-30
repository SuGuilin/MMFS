import sys
sys.path.append("/home/suguilin/myfusion/")
from torch import nn
import torch

from utils.utils import PatchUnEmbed, PatchEmbed
from utils.modules import M3, TokenSwapMamba, SEBlock, LocalAttentionFusion, SelfAttention

class MambaFusion(nn.Module):
    def __init__(self, dims=16, num_fusion_layer=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_fusion_layer = num_fusion_layer
        self.patch_embed = nn.ModuleList(
            PatchEmbed(embed_dim=dims * (2**i), norm_layer=norm_layer)
            for i in range(num_fusion_layer)
        )

        self.patch_unembed =nn.ModuleList(
            PatchUnEmbed(embed_dim=dims * (2**i), norm_layer=norm_layer)
            for i in range(num_fusion_layer)
        )

        self.channel_exchange1 = nn.ModuleList(
            TokenSwapMamba(dims * (2 ** i), reduction=(4 if dims * (2 ** i) < 100 else 16)) for i in range(num_fusion_layer))
        
        self.channel_exchange2 = nn.ModuleList(
            TokenSwapMamba(dims * (2 ** i), reduction=(4 if dims * (2 ** i) < 100 else 16)) for i in range(num_fusion_layer))
        
        self.shallow_fusion1 = nn.ModuleList(
            nn.Conv2d(dims * (2 ** i) * 2, dims * (2 ** i), 3, 1, 1)
            for i in range(num_fusion_layer)
        )

        self.shallow_fusion2 = nn.ModuleList(
            nn.Conv2d(dims * (2 ** i) * 2, dims * (2 ** i), 3, 1, 1, padding_mode="reflect")
            for i in range(num_fusion_layer)
        )
        self.deep_fusion = nn.ModuleList([
            nn.ModuleList([M3(dims * (2 ** i)) for j in range(3)])
            for i in range(num_fusion_layer)
        ])
        self.skip_connections = nn.ModuleList(
            nn.Conv2d(dims * (2 ** i), dims * (2 ** i), 1)
            for i in range(num_fusion_layer)
        )
        self.local_attention = nn.ModuleList(
            LocalAttentionFusion(dims * (2 ** i))
            for i in range(num_fusion_layer)
        )
        # self.self_attention = nn.ModuleList(
        #     SelfAttention(dims * (2 ** i))
        #     for i in range(num_fusion_layer)
        # )

    def forward(self, outs_rgb, outs_ir):
        outs_fused = []
        for i in range(self.num_fusion_layer):
            out_rgb = outs_rgb[i].permute(0, 3, 1, 2).contiguous()
            out_ir = outs_ir[i].permute(0, 3, 1, 2).contiguous()
            b, c, h, w = out_rgb.shape
            out_rgb = out_rgb.view(b, c, -1).permute(0, 2, 1)
            out_ir = out_ir.view(b, c, -1).permute(0, 2, 1)

            I_rgb, I_ir = self.channel_exchange1[i](out_rgb, out_ir)
            I_rgb, I_ir = self.channel_exchange2[i](I_rgb, I_ir)

            I_rgb = self.patch_unembed[i](I_rgb, (h, w))
            I_ir = self.patch_unembed[i](I_ir, (h, w))

            I_rgb = self.shallow_fusion1[i](torch.concat([I_rgb, I_ir], dim=1)) + I_rgb
            I_ir = self.shallow_fusion2[i](torch.concat([I_ir, I_rgb], dim=1)) + I_ir

            fusion_f = self.local_attention[i](I_rgb, I_ir)#(I_rgb + I_ir) / 2#
            skip_connection = self.skip_connections[i](fusion_f)

            test_h, test_w = I_rgb.shape[2], I_rgb.shape[3]

            I_rgb = self.patch_embed[i](I_rgb)
            I_ir = self.patch_embed[i](I_ir)
            fusion_f = self.patch_embed[i](fusion_f)

            residual_fusion_f = 0

            for j in range(3):  # 使用改进的深层融合模块
                fusion_f, residual_fusion_f = self.deep_fusion[i][j](
                    I_rgb, residual_fusion_f, I_ir, fusion_f, h, w
                )
            # fusion_f, residual_fusion_f = self.deep_fusion[i][0](
            #     I_rgb, residual_fusion_f, I_ir, fusion_f, test_h, test_w
            # )
            # fusion_f, residual_fusion_f = self.deep_fusion[i][1](
            #     I_rgb, residual_fusion_f, I_ir, fusion_f, test_h, test_w
            # )
            # fusion_f, residual_fusion_f = self.deep_fusion[i][2](
            #     I_rgb, residual_fusion_f, I_ir, fusion_f, test_h, test_w
            # )
            # fusion_f, residual_fusion_f = self.deep_fusion[i][3](
            #     I_rgb, residual_fusion_f, I_ir, fusion_f, test_h, test_w
            # )
            # fusion_f, residual_fusion_f = self.deep_fusion[i][4](
            #     I_rgb, residual_fusion_f, I_ir, fusion_f, test_h, test_w
            # )
            # fusion_f = self.self_attention[i](fusion_f)
            x_fuse = self.patch_unembed[i](fusion_f, (h, w)) + skip_connection
            outs_fused.append(x_fuse.permute(0, 2, 3, 1).contiguous())
        return outs_fused

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_sizes = [[1, 96, 120, 160], [1, 192, 60, 80], [1, 384, 30, 40], [1, 768, 15, 20]]
    rgb_outs = []
    ir_outs = []
    for feature_size in feature_sizes:
        rgb_outs.append(torch.randn(*feature_size).to(device))
        ir_outs.append(torch.randn(*feature_size).to(device))
    model = MambaFusion().to(device)
    output = model(rgb_outs, ir_outs)
    for i in range(4):
        print(output[i].shape)

