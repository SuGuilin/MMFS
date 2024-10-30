import sys
sys.path.append("/home/suguilin/myfusion/")
from torch import nn
import torch

from utils.modules import FusionBlock

class CrossFusion(nn.Module):
    def __init__(self, dims=[96, 192, 384, 768], drop_rate=0., num_fusion_layer=4, d_state=24):
        super().__init__()
        self.num_fusion_layer = num_fusion_layer
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([FusionBlock(
                    d_model=dims[j],
                    dropout=drop_rate,
                    d_state=d_state
                ) for i in range(2)])
                for j in range(self.num_fusion_layer)
            ]
        )

    def forward(self, outs_rgb, outs_ir):
        outs_fused = []
        for i in range(self.num_fusion_layer):
            for layer in self.layers[i]:
                x, y = layer(outs_rgb[i], outs_ir[i])
                outs_rgb[i], outs_ir[i] = outs_rgb[i] + x, outs_ir[i] + y
            outs_fused.append(outs_rgb[i] + outs_ir[i])
        return outs_fused

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_sizes = [[1, 120, 160, 96], [1, 60, 80, 192], [1, 30, 40, 384], [1, 15, 20, 768]]
    rgb_outs = []
    ir_outs = []
    for feature_size in feature_sizes:
        rgb_outs.append(torch.randn(*feature_size).to(device))
        ir_outs.append(torch.randn(*feature_size).to(device))
    model = CrossFusion().to(device)
    output = model(rgb_outs, ir_outs)
    for i in range(4):
        print(output[i].shape)
