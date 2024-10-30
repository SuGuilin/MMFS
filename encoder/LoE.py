import sys

sys.path.append("/home/suguilin/Graduation/myfusion/")
import torch
import torch.nn as nn
from GateRouter import GateNetwork_Local


class Local_Expert(nn.Module):
    def __init__(self, in_channels):
        super(Local_Expert, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=True, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1, bias=True, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.attention(x)
        return x

class MoLE(nn.Module):
    def __init__(self, dim, batch_size, num_experts, topk):
        super(MoLE, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.topk = topk
        self.batch_size = batch_size
        self.experts_rgb = nn.ModuleList([
            nn.ModuleList(
                [Local_Expert(dim) for i in range(num_experts)])
                for j in range(batch_size)
        ])

        self.experts_ir = nn.ModuleList([
            nn.ModuleList(
                [Local_Expert(dim) for i in range(num_experts)])
                for j in range(batch_size)
        ])

        self.router = GateNetwork_Local(dim*480*640, num_experts, topk)

    def forward(self, rgb_local, ir_local):
        weights, index_rgb, index_ir = self.router(rgb_local, ir_local)
        print(weights, index_rgb, index_ir)
        rgb_local = torch.stack([
            sum(self.experts_rgb[b][i](rgb_local) * weights[b, 0]
                for i in index_rgb[b])
            for b in range(self.batch_size)
        ], dim=0).sum(dim=1)
        ir_local = torch.stack([
            sum(self.experts_ir[b][i](ir_local) * weights[b, 1]
                for i in index_ir[b])
            for b in range(self.batch_size)
        ], dim=0).sum(dim=1)

        local = rgb_local + ir_local

        return local
    

if __name__ == "__main__":
    xV = torch.randn(4, 32, 480, 640)  # 可见光局部特征
    xI = torch.randn(4, 32, 480, 640)  # 红外局部特征
    model = MoLE(dim=32, batch_size=4, num_experts=2, topk=1)
    att = model(xV, xI)
    print(att.shape)

