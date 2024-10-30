import sys
sys.path.append("/home/suguilin/Graduation/myfusion/")
import torch
import torch.nn as nn
from utils.utils import PatchEmbed, PatchUnEmbed
from utils.MixBlock import MixBlock
import torch.nn.functional as F
import cv2

class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super(Conv2dNormActivation, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class MixAttention(nn.Module):
    def __init__(self, 
        input_resolution,
        num_heads=8,
        d_model=64,
        ):
        super(MixAttention, self).__init__()
        self.input_resolution = input_resolution
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            PatchEmbed(embed_dim=64, norm_layer=nn.LayerNorm),
            MixBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=4,
                input_resolution=input_resolution),
            PatchUnEmbed(embed_dim=64),
            Conv2dNormActivation(64, 128, 3, 1, 1),
        )
        self.fpn = nn.ModuleList([
            Conv2dNormActivation(64, 32),
            Conv2dNormActivation(128, 32),
        ])

        self.fpn_out = Conv2dNormActivation(32, 32, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # (B, 1, H, W)
        c1 = self.body[0](x)
        c1 = self.body[1](c1)
        c1 = self.body[2](c1)
        c1 = self.body[3](c1)
        c1 = self.body[4](c1)
        c1 = self.body[5](c1)

        c2 = self.body[6](c1, self.input_resolution)  # (B, 64, H/4, W/4)
        c3 = self.body[7](c2)  # (B, 128, H/4, W/4)

        p2 = self.fpn[1](c3)
        p1 = self.fpn[0](c2) + F.interpolate(p2, size=c2.shape[2:], mode="nearest")
        
        p1 = self.fpn_out(p1)
        p1 = self.sigmoid(p1)
        return p1

class CombineNet(nn.Module):
    def __init__(self, 
        num_heads=8,
        d_model=64,
        dim=32):
        super(CombineNet, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.combine_attention = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim*2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, rgb, ir):
        B, C, H, W = rgb.shape
        self.att_module_rgb = MixAttention(input_resolution=(int(H/4), int(W/4)), num_heads=self.num_heads, d_model=self.d_model).to(ir.device)
        self.att_module_ir = MixAttention(input_resolution=(int(H/4), int(W/4)), num_heads=self.num_heads, d_model=self.d_model).to(ir.device)
        rgb_attention = self.att_module_rgb(rgb)
        ir_attention = self.att_module_ir(ir)
        Att = torch.cat((rgb_attention, ir_attention), dim=1)
        Att = self.combine_attention(Att)
        return torch.max(Att, dim=1, keepdim=True)[0]


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ir = torch.from_numpy(cv2.imread('00003N_ir.png', cv2.IMREAD_GRAYSCALE)/255.).float().unsqueeze(0).unsqueeze(0).to(device)
    vi = torch.from_numpy(cv2.imread('00003N_vi.png', cv2.IMREAD_GRAYSCALE)/255.).float().unsqueeze(0).unsqueeze(0).to(device)
    xV = torch.randn(1, 1, 384, 512).to(device)  # 可见光输入
    xI = torch.randn(1, 1, 384, 512).to(device)  # 红外输入
    model = CombineNet().to(device)
    # att = model(xI, xV)
    att = model(vi, ir)
    print(att)
    print(att.shape)