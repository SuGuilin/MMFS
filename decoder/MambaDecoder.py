import sys

sys.path.append("/home/suguilin/Graduation/myfusion/")
from torch import nn
import torch


from utils.vssm import VSSM
from utils.utils import Final_PatchExpand2D, FinalUpsample_X4, FinalPatchExpand_X4, UpsampleExpand, Final_UpsampleExpand
from utils.fold import Fold

class MambaDecoder(nn.Module):
    def __init__(
        self,
        out_chans=3,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        depths=[2, 2, 4, 2],
        dims=[768, 384, 192, 96],
        downsample_version="v1",
        patch_size=4,
        d_state=16,
        drop_rate=0., 
        drop_path_rate=0.2,
        upsample_option=True,
        **kwargs
    ):
        super().__init__()
        self.vssm_fused = VSSM(
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
        self.final_up = Final_UpsampleExpand(dim=dims[-1], scale_factor=2, norm_layer=norm_layer)
        # self.final_up = Final_PatchExpand2D(dim=dims[-1], dim_scale=4, norm_layer=norm_layer)
        # self.final_up = FinalUpsample_X4(dim=dims[-1], patch_size=4, norm_layer=norm_layer)
        # self.final_up = FinalPatchExpand_X4(dim=dims[-1], patch_size=4, norm_layer=norm_layer)
        # self.fold = Fold(kernel_size=4, stride=4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_last1 = nn.Conv2d(dims[-1], dims[-1] // 2, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(dims[-1] // 2, dims[-1] // 4, 3, 1, 1)
        self.final_conv = nn.Conv2d(dims[-1] // 4, out_chans, 3, 1, 1)
        # self.final_conv = nn.Conv2d(dims[-1], out_chans, 1)


    def forward(self, rgb_ir, fused_outs):
        res = self.vssm_fused(rgb_ir, fused_outs)
        res = self.final_up(res)
        res = res.permute(0, 3, 1, 2)
        res = self.lrelu(self.conv_last1(res))
        res = self.lrelu(self.conv_last2(res))
        res = self.final_conv(res)

        return res
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_sizes = [[1, 240, 320, 96], [1, 120, 160, 192], [1, 60, 80, 384], [1, 30, 40, 768]]
    fused_outs = []
    for feature_size in feature_sizes:
        fused_outs.append(torch.randn(*feature_size).to(device))
    rgb_ir = torch.randn(1, 30, 40, 768).to(device)
    model = MambaDecoder(
    out_chans=3,
    dims=[768, 384, 192, 96],
    patch_size=4,
    d_state=16
    ).to(device)
    output = model(rgb_ir, fused_outs)
    print(output.shape)