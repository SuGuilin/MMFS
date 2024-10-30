import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts_per_branch, top_k):
        super(GateNetwork, self).__init__()
        self.num_experts_per_branch = num_experts_per_branch
        self.top_k = top_k
        self.linear_rgb = nn.Linear(input_dim, num_experts_per_branch)  # 单独计算RGB得分
        self.linear_ir = nn.Linear(input_dim, num_experts_per_branch)  # 单独计算IR得分
        self.experts_rgb = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_experts_per_branch)])
        self.experts_ir = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_experts_per_branch)])
        self.relu = nn.ReLU()

    def forward(self, rgb_local, ir_local):
        # Flatten the input: (batch_size, H, W, C) -> (batch_size, H*W*C)
        rgb = rgb_local.view(rgb_local.size(0), -1)
        ir = ir_local.view(ir_local.size(0), -1)
        rgb_ir = torch.cat([rgb, ir], dim=1)

        # Compute scores separately for RGB and IR
        scores_rgb = self.linear_rgb(rgb_ir)  # (batch_size, num_experts_per_branch)
        scores_ir = self.linear_ir(rgb_ir)  # (batch_size, num_experts_per_branch)

        # Apply ReLU to ensure non-negative scores
        scores_rgb = self.relu(scores_rgb)
        scores_ir = self.relu(scores_ir)

        # Select top 1 scores from the first two experts and the last two experts for RGB and IR separately
        top1_scores_rgb_1, top1_indices_rgb_1 = torch.topk(scores_rgb[:, :2], 1, dim=-1)
        top1_scores_rgb_2, top1_indices_rgb_2 = torch.topk(scores_rgb[:, 2:], 1, dim=-1)
        top1_indices_rgb_2 += 2  # Adjust indices for the second part

        top1_scores_ir_1, top1_indices_ir_1 = torch.topk(scores_ir[:, :2], 1, dim=-1)
        top1_scores_ir_2, top1_indices_ir_2 = torch.topk(scores_ir[:, 2:], 1, dim=-1)
        top1_indices_ir_2 += 2  # Adjust indices for the second part

        # Concatenate scores and indices
        top1_scores_rgb = torch.cat([top1_scores_rgb_1, top1_scores_rgb_2], dim=-1)
        top1_indices_rgb = torch.cat([top1_indices_rgb_1, top1_indices_rgb_2], dim=-1)
        top1_scores_ir = torch.cat([top1_scores_ir_1, top1_scores_ir_2], dim=-1)
        top1_indices_ir = torch.cat([top1_indices_ir_1, top1_indices_ir_2], dim=-1)

        # Ensure that the sum of the probabilities is 1 for RGB and IR separately
        probs_rgb = F.softmax(top1_scores_rgb, dim=-1)
        probs_ir = F.softmax(top1_scores_ir, dim=-1)

        # Get the indices of the maximum probability
        max_idx_rgb = torch.argmax(probs_rgb, dim=-1)
        max_idx_ir = torch.argmax(probs_ir, dim=-1)

        # Initialize outputs for RGB and IR
        output_rgb = torch.zeros_like(rgb_ir)
        output_ir = torch.zeros_like(rgb_ir)

        # Compute the weighted sum of the expert outputs for RGB
        for i in range(self.top_k):
            output_rgb += probs_rgb[:, i].unsqueeze(-1) * self.experts_rgb[top1_indices_rgb[:, i]](rgb_ir)

        # Compute the weighted sum of the expert outputs for IR
        for i in range(self.top_k):
            output_ir += probs_ir[:, i].unsqueeze(-1) * self.experts_ir[top1_indices_ir[:, i]](rgb_ir)

        # Combine RGB and IR outputs
        combined_output = output_rgb + output_ir

        return combined_output, max_idx_rgb, max_idx_ir

# Example usage
batch_size = 1
H, W, C = 4, 4, 3
input_dim = H * W * C * 2  # Because of concatenation of rgb and ir
num_experts_per_branch = 4
top_k = 2

# Initialize the Gate Network
gate_net = GateNetwork(input_dim, num_experts_per_branch, top_k)

# Example inputs: (batch_size, H, W, C)
rgb_local = torch.randn(batch_size, H, W, C)
ir_local = torch.randn(batch_size, H, W, C)

# Forward pass
combined_output, max_idx_rgb, max_idx_ir = gate_net(rgb_local, ir_local)

print("Combined Output:", combined_output)
print("Max Index RGB:", max_idx_rgb)
print("Max Index IR:", max_idx_ir)
