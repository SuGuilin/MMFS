import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.ReLU(),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=3, padding=1, stride=1, groups=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*4, dim, 1), 
            nn.Sigmoid(),
        )
        self.last_conv = nn.Conv2d(dim*2, dim, kernel_size=1)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        x_mean_out = torch.mean(x, dim=1, keepdim=True)
        x_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s1 = self.conv(torch.cat([x_mean_out, x_max_out], dim=1))
        s2 = self.mlp(self.avg_pool(x))
        out += self.last_conv(torch.cat([s1*y, s2*y], dim=1))

        out = self.project_out(out)
        return out
    
if __name__ == '__main__':
    seg_input = torch.randn(2, 96, 120, 160)
    fusion_input = torch.randn(2, 96, 120, 160)
    model = Chanel_Cross_Attention(dim=96, num_head=8, bias=True)

    output = model(seg_input, fusion_input)
    print(output.shape)