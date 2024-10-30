import torch
import torch.nn as nn
from decoder.MambaDecoder import MambaDecoder
from encoder.backbone import RGBXTransformer
from fusion.MambaFusion import MambaFusion
from fusion.CrossFusion import CrossFusion
from fusion.TransformerFusion import TransformerFusion
from fusion.AttentionFusion import AttentionFusion

class MYFusion(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        patch_size=4,
        # dims_en=[48, 96, 192, 384],
        # dims_de=[384, 192, 96, 48],
        dims_en=[16, 32, 64, 128],
        dims_de=[128, 64, 32, 16],
        depths_en=[2, 2, 2, 2],
        depths_de=[2, 2, 2, 2],
        downsample_version="v2",
        d_state=16,
        drop_rate=0.0,
        drop_path_rate=0.2,
    ):
        super().__init__()
        self.encoder = RGBXTransformer(
            in_chans=in_chans,
            patch_size=patch_size,
            dims=dims_en,
            depths=depths_en,
            downsample_version=downsample_version,
            d_state=d_state,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # self.fusion = MambaFusion()
        # self.fusion = CrossFusion()
        # self.fusion = TransformerFusion()
        self.fusion = AttentionFusion()

        self.decoder = MambaDecoder(
            out_chans=out_chans,
            dims=dims_de,
            depths=depths_de,
            patch_size=patch_size,
            d_state=d_state,
        )

    def forward(self, rgb, ir):
        outs_rgb, outs_ir, rgb_ir = self.encoder(rgb, ir)
        outs_fused = self.fusion(outs_rgb, outs_ir)
        res = self.decoder(rgb_ir, outs_fused)

        return res
