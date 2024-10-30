import sys

sys.path.append("/home/suguilin/Graduation/myfusion/")
from torch import nn
import torch
from utils.vssm import VSSM


class RGBXTransformer(nn.Module):
    def __init__(
        self,
        in_chans=1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        downsample_version="v1",
        patch_size=4,
        d_state=16,
        drop_rate=0., 
        drop_path_rate=0.2,
        upsample_option=False,
        **kwargs
    ):
        super().__init__()
        self.in_chans = in_chans
        self.vssm_rgb = VSSM(
            in_chans=in_chans,#*3,
            dims=dims,
            patch_size=patch_size,
            patch_norm=patch_norm,
            depths=depths,
            drop_path_rate=drop_path_rate,
            d_state=d_state,
            drop_rate=drop_rate, 
            norm_layer=norm_layer,
            dowansample_version=downsample_version,
            upsample_option=upsample_option
        )

        self.vssm_ir = VSSM(
            in_chans=in_chans,
            dims=dims,
            patch_size=patch_size,
            patch_norm=patch_norm,
            depths=depths,
            drop_path_rate=drop_path_rate,
            d_state=d_state,
            drop_rate=drop_rate, 
            norm_layer=norm_layer,
            dowansample_version=downsample_version,
            upsample_option=upsample_option
        )
        self.conv = nn.Conv2d(dims[-1]*2, dims[-1], 3, 1, 1, padding_mode="reflect")
    def forward(self, rgb, ir):
        rgb, rgb_outs = self.vssm_rgb(rgb)
        ir, ir_outs = self.vssm_ir(ir)
        rgb_ir = self.conv(torch.cat([rgb, ir], dim=3).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        return rgb_outs, ir_outs, rgb_ir