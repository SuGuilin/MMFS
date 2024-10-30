import sys
sys.path.append("/home/suguilin/myfusion/")
from torch import nn
import torch

from utils.modules import Restormer_CNN_block

class TransformerFusion(nn.Module):
    def __init__(self, dims=48, num_fusion_layer=4, norm_layer=nn.LayerNorm):
        super(TransformerFusion, self).__init__()
        self.num_fusion_layer = num_fusion_layer
        self.fusion_blocks = nn.ModuleList(
            Restormer_CNN_block(in_dim=dims * (2**i) * 2, out_dim=dims * (2**i)) 
            for i in range(num_fusion_layer)
        )
    def forward(self, outs_rgb, outs_ir):
        outs_fused = []
        for i in range(self.num_fusion_layer):
            out_rgb = outs_rgb[i].permute(0, 3, 1, 2).contiguous()
            out_ir = outs_ir[i].permute(0, 3, 1, 2).contiguous()
            x_fuse = self.fusion_blocks[i](torch.cat((out_rgb, out_ir), 1))
            outs_fused.append(x_fuse.permute(0, 2, 3, 1).contiguous())
        return outs_fused

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_sizes = [[1, 48, 120, 160], [1, 96, 60, 80], [1, 192, 30, 40], [1, 384, 15, 20]]
    rgb_outs = []
    ir_outs = []
    for feature_size in feature_sizes:
        rgb_outs.append(torch.randn(*feature_size).to(device))
        ir_outs.append(torch.randn(*feature_size).to(device))
    model = TransformerFusion().to(device)
    output = model(rgb_outs, ir_outs)
    for i in range(4):
        print(output[i].shape)