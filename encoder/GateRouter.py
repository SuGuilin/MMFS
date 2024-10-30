import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNetwork_Local(nn.Module):
    def __init__(self, input_dim, num_experts, top_k):
        super(GateNetwork_Local, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear_rgb = nn.Linear(input_dim, num_experts)
        self.linear_ir = nn.Linear(input_dim, num_experts)

    def forward(self, rgb_local, ir_local):
        # Flatten the input: (H, W, C) -> (H*W, C)
        rgb = rgb_local.view(rgb_local.size(0), -1)
        ir = ir_local.view(ir_local.size(0), -1)
        # rgb_ir = torch.cat([rgb, ir], dim=1)

        # Compute scores with the linear layer
        scores_rgb = self.linear_rgb(rgb)  # (batch_size, num_experts)
        scores_ir = self.linear_ir(ir)

        probs_rgb = F.softmax(scores_rgb, dim=-1)
        probs_ir = F.softmax(scores_ir, dim=-1)

        top_k_scores_rgb, top_k_indices_rgb = torch.topk(probs_rgb, self.top_k, dim=-1)
        top_k_scores_ir, top_k_indices_ir = torch.topk(probs_ir, self.top_k, dim=-1)
        # top_k_probs_rgb = torch.gather(probs_rgb, 1, top_k_indices_rgb)

        probs = F.softmax(torch.cat([top_k_scores_rgb, top_k_scores_ir], dim=-1), dim=-1)

        return probs, top_k_indices_rgb, top_k_indices_ir

class GateNetwork_Gobal(nn.Module):
    def __init__(self, input_dim, num_experts, top_k):
        super(GateNetwork_Gobal, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(input_dim, num_experts)
    def forward(self, x_local):
        scores = self.linear(x_local.view(x_local.size(0), -1))
        probs = F.softmax(scores, dim=-1)
        top_k_scores, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        probs = F.softmax(top_k_scores, dim=-1)
        return probs, top_k_indices

if __name__ == "__main__":
    batch_size = 3
    H, W, C = 240, 320, 32
    input_dim = H * W * C
    num_experts = 2
    top_k = 1
    gate_net = GateNetwork_Local(input_dim, num_experts, top_k)
    gobal = GateNetwork_Gobal(input_dim, num_experts, top_k)

    x_local = torch.randn(batch_size, H, W, C)
    y_local = torch.randn(batch_size, H, W, C)
    probs, v = gobal(x_local)#gate_net(x_local, y_local)
    x, y, z = gate_net(x_local, y_local)

    print(x)
    print(y)
    print(z)

