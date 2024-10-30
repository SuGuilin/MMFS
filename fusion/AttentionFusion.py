import sys
sys.path.append('/home/suguilin/zigma/')
sys.path.append("/home/suguilin/myfusion/")
from mamba_ssm.modules.mamba_simple import Mamba
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import copy
from utils.utils import LayerNorm

def activateFunc(x):
    x = torch.tanh(x)
    return F.relu(x)


class Router(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.conv_pool = nn.Linear(embed_size*36*36, embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.ReLU(True),
            nn.Linear(embed_size, num_out_path),
        )
        self.init_weights()

    def init_weights(self):
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):
        # x = x.reshape(x.shape[0],-1)
        avg_out = self.avgpool(x)
        max_out = self.maxpool(x)
        f_out = torch.cat([max_out, avg_out], 1)
        x = f_out.contiguous().view(f_out.size(0), -1)
        x = self.mlp(x)
        soft_g = activateFunc(x)
        return soft_g


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
        )
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


class CoTAttention(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=4,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1),
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2

# class MutualAttention(nn.Module):
#     def __init__(self, dim, num_heads=4, bias=False):
#         super(MutualAttention, self).__init__()
#         self.Mamba = Mamba(dim, bimamba_type="m3")
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.kv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.normx = LayerNorm(dim, "with_bias")
#         self.normy = LayerNorm(dim, "with_bias")
#         self.out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
#     def forward(self, x, y):
#         # print(x.shape)
#         assert (x.shape == y.shape), "The shape of feature maps from image and event branch are not equal!"
#         b, c, h, w = x.shape
#         x = self.q(x)
#         y = self.kv(y)
#         x = x.view(b, c, -1).permute(0, 2, 1)
#         y = y.view(b, c, -1).permute(0, 2, 1)
#         out = self.Mamba(self.normx(x), extra_emb1=self.normy(y), extra_emb2=self.normy(y))

#         out = out.permute(0, 2, 1).contiguous().view(b, c, h, w)
#         out = self.out(out)
#         return out


class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(MutualAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # print("init x: ", x.shape)
        assert (
            x.shape == y.shape
        ), "The shape of feature maps from image and event branch are not equal!"

        b, c, h, w = x.shape

        q = self.q(x)  # image
        k = self.k(y)  # event
        v = self.v(y)  # event

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        # print("att:", out.shape)
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        # print("before pro:", out.shape)
        out = self.project_out(out)
        # print("after pro:", out.shape)
        return out


class ChannelCell(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(ChannelCell, self).__init__()
        self.router = Router(num_out_path, embed_size * 2)
        self.channel_att_1 = ECAAttention(kernel_size=3)  # .cuda()
        self.channel_att_2 = ECAAttention(kernel_size=3)  # .cuda()

    def forward(self, x1, x2):
        x12 = torch.cat([x1, x2], 1)
        path_prob = self.router(x12)
        esa_emb1 = self.channel_att_1(x1)
        esa_emb2 = self.channel_att_2(x2)

        return [esa_emb1, esa_emb2], path_prob


class SpatailCell(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(SpatailCell, self).__init__()
        self.router = Router(num_out_path, embed_size * 2)
        self.spatial_att_1 = SpatialGroupEnhance(groups=8)
        self.spatial_att_2 = SpatialGroupEnhance(groups=8)

    def forward(self, x1, x2):
        x12 = torch.cat([x1, x2], 1)
        path_prob = self.router(x12)
        sge_emb1 = self.spatial_att_1(x1)
        sge_emb2 = self.spatial_att_1(x2)
        return [sge_emb1, sge_emb2], path_prob


class T2RCell(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(T2RCell, self).__init__()
        self.router = Router(num_out_path, embed_size * 2)
        self.cross_att = MutualAttention(dim=embed_size)

    def forward(self, x1, x2):
        x12 = torch.cat([x1, x2], 1)
        path_prob = self.router(x12)

        cross_emb = x1 + self.cross_att(x1, x2)

        return [cross_emb, x2], path_prob


class R2TCell(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(R2TCell, self).__init__()
        self.router = Router(num_out_path, embed_size * 2)
        self.cross_att = MutualAttention(dim=embed_size)

    def forward(self, x1, x2):
        x12 = torch.cat([x1, x2], 1)
        path_prob = self.router(x12)
        cross_emb = x2 + self.cross_att(x2, x1)

        return [x1, cross_emb], path_prob


def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)


def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DynamicFusion_Layer0(nn.Module):
    def __init__(self, num_cell, num_out_path, embed_size):
        super(DynamicFusion_Layer0, self).__init__()
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.Channel = ChannelCell(num_out_path, embed_size)
        self.Spatial = SpatailCell(num_out_path, embed_size)

        self.R2T_cross = R2TCell(num_out_path, embed_size)
        self.T2R_cross = T2RCell(num_out_path, embed_size)

    def forward(self, x1, x2):
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.Channel(x1, x2)
        emb_lst[1], path_prob[1] = self.Spatial(x1, x2)
        emb_lst[2], path_prob[2] = self.R2T_cross(x1, x2)
        emb_lst[3], path_prob[3] = self.T2R_cross(x1, x2)

        # gate_mask = (sum(path_prob) < self.threshold).float()
        all_path_prob = torch.stack(path_prob, dim=2)

        all_path_prob = all_path_prob / (
            all_path_prob.sum(dim=-1, keepdim=True) + self.eps
        )
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]
        # print('path_prob',path_prob[0].shape) # 6, 4

        aggr_res_lst = []
        for i in range(self.num_out_path):
            # skip_emb = unsqueeze3d(gate_mask[:, i]) * emb_lst[0]
            res1 = 0
            res2 = 0

            for j in range(self.num_cell):
                cur_path = unsqueeze3d(path_prob[j][:, i])

                if emb_lst[j][0].dim() == 3:
                    cur_emb_1 = emb_lst[j][0].unsqueeze(1)
                    cur_emb_2 = emb_lst[j][1].unsqueeze(1)
                else:  # 4
                    cur_emb_1 = emb_lst[j][0]
                    cur_emb_2 = emb_lst[j][1]

                res1 = res1 + cur_path * cur_emb_1
                res2 = res2 + cur_path * cur_emb_2
                # print('res',res.shape)
            # res = res + skip_emb#.unsqueeze(1)
            aggr_res_lst.append([res1, res2])

        return aggr_res_lst


class DynamicFusion_Layer(nn.Module):
    def __init__(self, num_cell, num_out_path, embed_size):
        super(DynamicFusion_Layer, self).__init__()
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.Channel = ChannelCell(num_out_path, embed_size)
        self.Spatial = SpatailCell(num_out_path, embed_size)

        self.R2T_cross = R2TCell(num_out_path, embed_size)
        self.T2R_cross = T2RCell(num_out_path, embed_size)

    def forward(self, aggr_embed):
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell

        emb_lst[0], path_prob[0] = self.Channel(aggr_embed[0][0], aggr_embed[0][1])
        emb_lst[1], path_prob[1] = self.Spatial(aggr_embed[1][0], aggr_embed[1][1])
        emb_lst[2], path_prob[2] = self.R2T_cross(aggr_embed[2][0], aggr_embed[2][1])
        emb_lst[3], path_prob[3] = self.T2R_cross(aggr_embed[3][0], aggr_embed[3][1])

        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res1 = 0
            res2 = 0
            for j in range(self.num_cell):
                # gate_mask = (path_prob[j] < self.threshold / self.num_cell).float()
                # gate_mask_lst.append(gate_mask)
                # skip_emb = gate_mask.unsqueeze(-1).unsqueeze(-1) * aggr_embed[j]
                res1 += path_prob[j].unsqueeze(-1).unsqueeze(-1) * emb_lst[j][0]
                res2 += path_prob[j].unsqueeze(-1).unsqueeze(-1) * emb_lst[j][1]
                # res += skip_emb
            # res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1).unsqueeze(-1)
            # all_path_prob = torch.stack(path_prob, dim=2)
            # res_fusion = torch.cat([res1,res2],1)
            aggr_res_lst.append([res1, res2])
        else:
            # gate_mask = (sum(path_prob) < self.threshold).float()
            all_path_prob = torch.stack(path_prob, dim=2)

            all_path_prob = all_path_prob / (
                all_path_prob.sum(dim=-1, keepdim=True) + self.eps
            )

            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]
            aggr_res_lst = []
            for i in range(self.num_out_path):
                # skip_emb = unsqueeze3d(gate_mask[:, i]) * emb_lst[0]
                res1 = 0
                res2 = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze3d(path_prob[j][:, i])
                    res1 = res1 + cur_path * emb_lst[j][0]
                    res2 = res2 + cur_path * emb_lst[j][1]
                # res = res + skip_emb
                aggr_res_lst.append([res1, res2])

        return aggr_res_lst


class DynamicFusionModule(nn.Module):
    def __init__(self, num_layer_routing=3, num_cells=4, embed_size=1024):
        super(DynamicFusionModule, self).__init__()
        self.num_cells = num_cells = 4
        self.dynamic_fusion_l0 = DynamicFusion_Layer0(num_cells, num_cells, embed_size)
        self.dynamic_fusion_l1 = DynamicFusion_Layer(num_cells, num_cells, embed_size)
        self.dynamic_fusion_l2 = DynamicFusion_Layer(num_cells, 1, embed_size)
        # total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        # self.path_mapping = nn.Linear(total_paths, path_hid)
        # self.bn = nn.BatchNorm1d(opt.embed_size)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_size*2, embed_size, 1),
            nn.BatchNorm2d(embed_size),
            nn.ReLU(),
            nn.Conv2d(embed_size, embed_size, 3, padding=1),
            nn.BatchNorm2d(embed_size),
            nn.ReLU(),
            )

    def forward(self, x1, x2):
        # input_cat = torch.cat([input_feat_v,input_feat_i],1)
        pairs_emb_lst1 = self.dynamic_fusion_l0(x1, x2)
        pairs_emb_lst2 = self.dynamic_fusion_l1(pairs_emb_lst1)
        pairs_emb_lst3 = self.dynamic_fusion_l2(pairs_emb_lst2)

        feat1 = pairs_emb_lst3[0][0]   + x1
        feat2 = pairs_emb_lst3[0][1]   + x2
        fusion_feat = torch.cat([feat1, feat2], 1)
        fusion_feat = self.conv(fusion_feat)
        # lad123
        return fusion_feat

class AttentionFusion(nn.Module):
    def __init__(self, dims=16, num_fusion_layer=4, norm_layer=nn.LayerNorm):
        super(AttentionFusion, self).__init__()
        self.num_fusion_layer = num_fusion_layer
        self.fusion_blocks = nn.ModuleList(
            DynamicFusionModule(embed_size=dims * (2 ** i)) 
            for i in range(num_fusion_layer)
        )
    def forward(self, outs_rgb, outs_ir):
        outs_fused = []
        for i in range(self.num_fusion_layer):
            out_rgb = outs_rgb[i].permute(0, 3, 1, 2).contiguous()
            out_ir = outs_ir[i].permute(0, 3, 1, 2).contiguous()
            x_fuse = self.fusion_blocks[i](out_rgb, out_ir)
            outs_fused.append(x_fuse.permute(0, 2, 3, 1).contiguous())
        return outs_fused

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x1 = torch.randn(1, 16, 240, 320).to(device)
    x2 = torch.randn(1, 16, 240, 320).to(device)

    model = DynamicFusionModule(embed_size=16).to(device)
    fusion_feat = model(x1, x2)
    print(fusion_feat.shape)
    feature_sizes = [[1, 120, 160, 16], [1, 60, 80, 32], [1, 30, 40, 64], [1, 15, 20, 128]]
    rgb_outs = []
    ir_outs = []
    for feature_size in feature_sizes:
        rgb_outs.append(torch.randn(*feature_size).to(device))
        ir_outs.append(torch.randn(*feature_size).to(device))
    model1 = AttentionFusion().to(device)
    output = model1(rgb_outs, ir_outs)
    for i in range(4):
        print(output[i].shape)
