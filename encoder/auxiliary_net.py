import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super(Conv2dNormActivation, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BackboneWithFPN(nn.Module):
    def __init__(self):
        super(BackboneWithFPN, self).__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Identity(),
            self._make_layer(64, 64, 256, 2),
            self._make_layer(256, 128, 512, 3, stride=1),
            self._make_layer(512, 256, 1024, 4, stride=1),
            # self._make_layer(64, 64, 256, 3),
            # self._make_layer(256, 128, 512, 4, stride=1),
            # self._make_layer(512, 256, 1024, 6, stride=1),
            # self._make_layer(1024, 512, 2048, 3, stride=1)
        )
        
        self.fpn = nn.ModuleList([
            Conv2dNormActivation(256, 256),
            Conv2dNormActivation(512, 256),
            Conv2dNormActivation(1024, 256),
            # Conv2dNormActivation(2048, 256)
        ])
        
        self.fpn_out = Conv2dNormActivation(256, 32, kernel_size=3, stride=1, padding=1)
        

    def _make_layer(self, in_channels, mid_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Bottleneck(in_channels, mid_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, mid_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.body[0](x)
        c1 = self.body[1](c1)
        c1 = self.body[2](c1)
        c1 = self.body[3](c1)
        
        c2 = self.body[4](c1)
        c3 = self.body[5](c2)
        c4 = self.body[6](c3)
        # c5 = self.body[7](c4)
        
        # p5 = self.fpn[3](c5)
        p4 = self.fpn[2](c4) #+ F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        # print(c3.shape)
        # print(p4.shape)
        p3 = self.fpn[1](c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.fpn[0](c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        
        p2 = self.fpn_out(p2)
        
        return p2

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attention(x)

class AuxiliaryNet(nn.Module):
    def __init__(self, dim=32):
        super(AuxiliaryNet, self).__init__()
        self.backbone_rgb = BackboneWithFPN()
        self.att_module_rgb = AttentionModule(dim)
        self.backbone_ir = BackboneWithFPN()
        self.att_module_ir = AttentionModule(dim)
        self.combine_attention = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim*2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, rgb, ir):
        rgb = self.backbone_rgb(rgb)
        ir = self.backbone_ir(ir)

        rgb_attention = self.att_module_rgb(rgb)
        ir_attention = self.att_module_ir(ir)

        Att = torch.cat((rgb_attention, ir_attention), dim=1)
        Att = self.combine_attention(Att)

        return torch.max(Att, dim=1, keepdim=True)[0]

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xV = torch.randn(1, 1, 480, 640)  # 可见光输入
    xI = torch.randn(1, 1, 480, 640)  # 红外输入
    ir = torch.from_numpy(cv2.imread('00003N_ir.png', cv2.IMREAD_GRAYSCALE)/255.).float().unsqueeze(0).unsqueeze(0).to(device)
    vi = torch.from_numpy(cv2.imread('00003N_vi.png', cv2.IMREAD_GRAYSCALE)/255.).float().unsqueeze(0).unsqueeze(0).to(device)
    model = AuxiliaryNet().to(device)
    # att = model(xV, xI)
    att = model(vi, ir)
    print(att)
    print(att.shape)
