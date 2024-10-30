import torch
import torch.nn as nn
import torch.nn.functional as F

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

class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k):
        super(GateNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(input_dim, num_experts)
    
    def forward(self, x_local):
        batch_size = x_local.size(0)
        scores = self.linear(x_local.view(batch_size, -1))
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        probs = F.softmax(top_k_scores, dim=-1)
        return probs, top_k_indices

class MoLE(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(MoLE, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Define the experts
        self.experts = nn.ModuleList([Local_Expert(dim) for _ in range(num_experts)])
        
        # Define the gate networks for RGB and IR
        self.gate_rgb = GateNetwork(dim * 480 * 640, num_experts // 2, top_k)
        self.gate_ir = GateNetwork(dim * 480 * 640, num_experts // 2, top_k)

    def forward(self, rgb_local, ir_local, rgb_dense, ir_dense):
        # Get gate weights and indices
        weights_rgb, index_rgb = self.gate_rgb(rgb_local)
        weights_ir, index_ir = self.gate_ir(ir_local)

        rgb_expert_outputs = torch.zeros_like(rgb_local)
        ir_expert_outputs = torch.zeros_like(ir_local)

        # 计算 RGB 专家的加权输出
        for k in range(self.top_k):
            idx = index_rgb[:, k].unsqueeze(1)  # 获取第 k 个专家的索引并添加一个维度
            weight = weights_rgb[:, k].view(-1, 1, 1, 1)  # 获取第 k 个专家的权重并添加维度
            for b in range(rgb_local.size(0)):  # 对于每个批次进行处理
                rgb_expert_outputs[b:b+1] += self.experts[idx[b].item()](rgb_local[b:b+1]) * weight[b]

        # 计算 IR 专家的加权输出
        for k in range(self.top_k):
            idx = index_ir[:, k].unsqueeze(1) + self.num_experts // 2  # 获取第 k 个专家的索引并添加偏移量
            weight = weights_ir[:, k].view(-1, 1, 1, 1)  # 获取第 k 个专家的权重并添加维度
            for b in range(ir_local.size(0)):  # 对于每个批次进行处理
                ir_expert_outputs[b:b+1] += self.experts[idx[b].item()](ir_local[b:b+1]) * weight[b]

        # 合并 RGB 和 IR 专家的输出
        local_features = rgb_expert_outputs + ir_expert_outputs

        # 与 dense 特征拼接
        global_features = torch.cat([local_features, rgb_dense, ir_dense], dim=1)

        return global_features

# Example usage
dim = 32
batch_size = 4
num_experts = 4
top_k = 1

model = MoLE(dim, num_experts, top_k)
rgb_local = torch.randn(batch_size, dim, 480, 640)
ir_local = torch.randn(batch_size, dim, 480, 640)
rgb_dense = torch.randn(batch_size, dim, 480, 640)
ir_dense = torch.randn(batch_size, dim, 480, 640)

output = model(rgb_local, ir_local, rgb_dense, ir_dense)
print(output.shape)
