import sys

sys.path.append("/home/suguilin/Graduation/myfusion/")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from fusion.AttentionFusion import DynamicFusionModule, MutualAttention
from utils.STMBlock import STMBlock
from utils.CTMBlock import CTMBlock
from utils.utils import Permute
from utils.NAFBlock import NAFBlock, NAFNet, Global_Dynamics, Local_Dynamics
from utils.MixBlock import Mlp
from utils.modules import ChannelAttention
from utils.InvertedBlock import DetailFeatureExtraction
from encoder.cross_mamba import Mlp


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp, extra=None):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        if extra is not None:
            extra_exp = extra[self._batch_index].squeeze(1)
            return torch.split(inp_exp, self._part_sizes, dim=0), torch.split(extra_exp, self._part_sizes, dim=0)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MambaMoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        width,
        height,
        local_dim,
        gobal_dim,
        local_num_experts,
        gobal_num_experts,
        noisy_gating=True,
        local_k=2,
        gobal_k=4,
    ):
        super(MambaMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.width = width
        self.height = height
        self.input_size = local_dim * height * width

        # local experts initialization
        self.local_k = local_k
        self.local_dim = local_dim
        self.local_num_experts = local_num_experts
        self.local_experts = nn.ModuleList(
            [Local_Expert(self.local_dim) for i in range(self.local_num_experts * 2)]
        )
        self.local_w_gate = [
            nn.Parameter(
                torch.zeros(
                    self.width * self.height * self.local_dim, self.local_num_experts
                ),
                requires_grad=True,
            )
            for i in range(2)
        ]
        self.local_w_noise = [
            nn.Parameter(
                torch.zeros(
                    self.width * self.height * self.local_dim, self.local_num_experts
                ),
                requires_grad=True,
            )
            for i in range(2)
        ]

        # self.enhance_mutual = NAFNet(
        #     img_channel=64,
        #     embed_dim=64,
        #     out_channels=32,
        #     middle_blk_num=2,
        #     enc_blk_nums=[2, 2, 4],
        #     dec_blk_nums=[4, 2, 2],
        # )
        self.reduce = nn.Conv2d(local_dim * 2, local_dim, 3, 1, 1)

        self.rgb_gobal_in = nn.Sequential(
            ChannelAttention(local_dim * 5),
            nn.Conv2d(local_dim * 5, gobal_dim, kernel_size=1, bias=False),
        )

        self.ir_gobal_in = nn.Sequential(
            ChannelAttention(local_dim * 5),
            nn.Conv2d(local_dim * 5, gobal_dim, kernel_size=1, bias=False),
        )

        # global experts initialization
        self.gobal_k = gobal_k
        self.gobal_dim = gobal_dim
        self.gobal_num_experts = gobal_num_experts
        self.gobal_experts = nn.ModuleList(
            [Global_Expert(self.gobal_dim) for i in range(self.gobal_num_experts)]
        )
        self.gobal_w_gate = nn.Parameter(
            torch.zeros(
                self.width * self.height * self.gobal_dim, self.gobal_num_experts
            ),
            requires_grad=True,
        )
        self.gobal_w_noise = nn.Parameter(
            torch.zeros(
                self.width * self.height * self.gobal_dim, self.gobal_num_experts
            ),
            requires_grad=True,
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.local_k <= self.local_num_experts
        assert self.gobal_k <= self.gobal_num_experts

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, k, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)  # 获取 batch 的大小
        m = noisy_top_values.size(1)  # 获取 top k 的大小
        top_values_flat = noisy_top_values.flatten()  # 将 noisy_top_values 展平

        # 计算如果 clean_values 在 top k 中，获取门限值的位置
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + k
        )
        # 提取门限值
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        # 判断 noisy_values 是否大于门限值
        is_in = torch.gt(noisy_values, threshold_if_in)
        # 计算如果 clean_values 不在 top k 中，门限值的位置
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, locindex, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        # whether is local_experts
        if locindex != 2:
            w_gate = self.local_w_gate[locindex]
            w_noise = self.local_w_noise[locindex]
            k = self.local_k
            num_experts = self.local_num_experts
        else:
            w_gate = self.gobal_w_gate
            w_noise = self.gobal_w_noise
            k = self.gobal_k
            num_experts = self.gobal_num_experts

        w_gate = w_gate.to(x.device)
        w_noise = w_noise.to(x.device)

        clean_logits = x @ w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise
            # softplus(y) = log(1 + e^y)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(k + 1, num_experts), dim=1)
        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]
        top_k_gates = top_k_logits / (
            top_k_logits.sum(1, keepdim=True) + 1e-6
        )  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and k < num_experts and train:
            load = (
                self._prob_in_top_k(
                    k, clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, rgb_local, ir_local, rgb_dense, ir_dense, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        # local rgb branch
        rgb_gates, rgb_load = self.noisy_top_k_gating(
            x=rgb_local, train=self.training, locindex=0
        )
        # calculate importance loss
        rgb_importance = rgb_gates.sum(0)
        rgb_loss = self.cv_squared(rgb_importance) + self.cv_squared(rgb_load)
        rgb_loss *= loss_coef

        rgb_dispatcher = SparseDispatcher(self.local_num_experts, rgb_gates)
        rgb_expert_inputs = rgb_dispatcher.dispatch(rgb_local)
        rgb_gates = rgb_dispatcher.expert_to_gates()
        rgb_expert_outputs = [
            self.local_experts[i](
                rgb_expert_inputs[i].view(-1, self.local_dim, self.height, self.width)
            ).view(-1, self.input_size)
            for i in range(self.local_num_experts)
        ]
        rgb_y = rgb_dispatcher.combine(rgb_expert_outputs).view(
            -1, self.local_dim, self.height, self.width
        )

        # local ir branch
        ir_gates, ir_load = self.noisy_top_k_gating(
            x=ir_local, train=self.training, locindex=1
        )
        # calculate importance loss
        ir_importance = ir_gates.sum(0)
        ir_loss = self.cv_squared(ir_importance) + self.cv_squared(ir_load)
        ir_loss *= loss_coef

        ir_dispatcher = SparseDispatcher(self.local_num_experts, ir_gates)
        ir_expert_inputs = ir_dispatcher.dispatch(ir_local)
        ir_gates = ir_dispatcher.expert_to_gates()
        ir_expert_outputs = [
            self.local_experts[i + self.local_num_experts](
                ir_expert_inputs[i].view(-1, self.local_dim, self.height, self.width)
            ).view(-1, self.input_size)
            for i in range(self.local_num_experts)
        ]
        ir_y = rgb_dispatcher.combine(ir_expert_outputs).view(
            -1, self.local_dim, self.height, self.width
        )
        # print(ir_y)

        # feature interaction enhancement
        y_local = self.reduce(torch.cat([rgb_y, ir_y], dim=1))
        # y_local = self.enhance_mutual(torch.cat([rgb_y, ir_y], dim=1))
        # print(y_local.shape)

        # gobal branch
        gobal_gates, gobal_load = self.noisy_top_k_gating(
            x=y_local.view(-1, self.input_size), train=self.training, locindex=2
        )
        # calculate importance loss
        global_importance = gobal_gates.sum(0)
        global_loss = self.cv_squared(global_importance) + self.cv_squared(gobal_load)
        global_loss *= (loss_coef * 2)

        global_dispatcher = SparseDispatcher(self.gobal_num_experts, gobal_gates)

        global_rgb_in = self.rgb_gobal_in(torch.cat([y_local, rgb_dense], dim=1)).view(-1, self.gobal_dim, self.width, self.height)
        global_ir_in = self.ir_gobal_in(torch.cat([y_local, ir_dense], dim=1)).view(-1, self.gobal_dim, self.width, self.height)

        global_expert_inputs_x, global_expert_inputs_y = global_dispatcher.dispatch(global_rgb_in, global_ir_in)
        # print(global_expert_inputs_x)
        # print(global_expert_inputs_y)
        gobal_gates = global_dispatcher.expert_to_gates()
        global_expert_outputs = [
            self.gobal_experts[i](
                global_expert_inputs_x[i].view(-1, self.gobal_dim, self.height, self.width),
                global_expert_inputs_y[i].view(-1, self.gobal_dim, self.height, self.width)
            ).view(-1, self.input_size)
            for i in range(self.gobal_num_experts)
        ]
        global_y = rgb_dispatcher.combine(global_expert_outputs).view(
            -1, self.local_dim, self.height, self.width
        )
        # print(global_y.shape)
        loss = rgb_loss + ir_loss + global_loss

        return global_y, loss


class Local_Expert(nn.Module):
    def __init__(self, in_channels):
        super(Local_Expert, self).__init__()
        # self.attention = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels,
        #         in_channels // 2,
        #         3,
        #         padding=1,
        #         bias=True,
        #         padding_mode="reflect",
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels // 2,
        #         in_channels,
        #         3,
        #         padding=1,
        #         bias=True,
        #         padding_mode="reflect",
        #     ),
        #     nn.ReLU(inplace=True),
        # )
        # )
        # self.attention = NAFBlock(c=in_channels, drop_out_rate=0.2)
        self.detail_en = DetailFeatureExtraction(dim=in_channels, num_layers=2)
        '''
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)
        self.conv_f = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        '''

    def forward(self, x):
        # print("local:", x.shape)
        # x = self.attention(x)
        # print("local_out:", x.shape)
        '''
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
        '''
        x = self.detail_en(x)
        return x

class Global_Expert(nn.Module):
    def __init__(self, dim):
        super(Global_Expert, self).__init__()
        # self.attention = DynamicFusionModule(embed_size=dim)
        self.vi_att = nn.Sequential(
            Permute(0, 2, 3, 1),
            STMBlock(
                hidden_dim=dim,
                drop_path=0.1,
                ssm_ratio=2.0,
                d_state=16,
                dt_rank="auto",
                mlp_ratio=4.0,
                norm_layer=nn.LayerNorm,
            ),
  
            # STMBlock(
            #     hidden_dim=dim,
            #     drop_path=0.1,
            #     ssm_ratio=2.0,
            #     d_state=16,
            #     dt_rank="auto",
            #     mlp_ratio=4.0,
            #     norm_layer=nn.LayerNorm,
            # ),
        )

        self.mlp_rgb = Mlp(
                in_features=dim, 
                hidden_features=int(dim * 4.0), 
                out_features=dim
        )

        self.mlp_ir = Mlp(
                in_features=dim, 
                hidden_features=int(dim * 4.0), 
                out_features=dim
        )

        self.ir_att = nn.Sequential(
            Permute(0, 2, 3, 1),
            STMBlock(
                hidden_dim=dim,
                drop_path=0.1,
                ssm_ratio=2.0,
                d_state=16,
                dt_rank="auto",
                mlp_ratio=4.0,
                norm_layer=nn.LayerNorm,
            ),
            # STMBlock(
            #     hidden_dim=dim,
            #     drop_path=0.1,
            #     ssm_ratio=2.0,
            #     d_state=16,
            #     dt_rank="auto",
            #     mlp_ratio=4.0,
            #     norm_layer=nn.LayerNorm,
            # ),
        )

        # self.cross_att = CTMBlock(
        #     modals=2,
        #     hidden_dim=dim,
        #     drop_path=0.1,
        #     shared_ssm=False,
        #     d_state=16,
        #     ssm_ratio=2.0,
        #     dt_rank="auto",
        #     mlp_ratio=4,
        #     norm_layer=nn.LayerNorm,
        # )

        self.mutal_att1 = MutualAttention(dim=dim)
        self.mutal_att2 = MutualAttention(dim=dim)
        self.output = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 3, padding=1, bias=True, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True, padding_mode="reflect"),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        B, C, H, W = x.shape
        if B == 0:
            return (x + y) / 2.0
        
        x = self.vi_att(x) + x.permute(0, 2, 3, 1)
        z1 = self.mlp_rgb(x.reshape(B, H * W, C), H, W).transpose(1, 2).view(B, -1, H, W) + x.permute(0, 3, 1, 2)
        y = self.ir_att(y) + y.permute(0, 2, 3, 1)
        z2 = self.mlp_ir(y.reshape(B, H * W, C), H, W).transpose(1, 2).view(B, -1, H, W) + y.permute(0, 3, 1, 2)

        # a, b = x, y # self.cross_att(x, y)
        a, b = z1, z2
        # z1 = self.mutal_att1(a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)) + a.permute(0, 3, 1, 2)
        # z1 = self.mutal_att1(a, b) + a
        # z1 = self.mlp_ir(z1.flatten(2).transpose(1, 2), H, W).transpose(1, 2).view(B, -1, H, W) + z1
        # z2 = self.mutal_att2(b.permute(0, 3, 1, 2), a.permute(0, 3, 1, 2)) + b.permute(0, 3, 1, 2)
        # z2 = self.mutal_att2(b, a) + b
        # z2 = self.mlp_ir(z2.flatten(2).transpose(1, 2), H, W).transpose(1, 2).view(B, -1, H, W) + z2
        # z = (z1 + z2) / 2.0
        # print(z1.shape)  # (b, c, h, w)
        z = self.output(torch.cat([z1, z2], dim=1))
        z = self.mutal_att1(z, z) + z
        # z = self.output(torch.cat([a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)], dim=1))
        # print(z.shape)
        return z  # (a + b)

class Global_Expert_init(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Global_Expert_init, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class GateNetwork(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(GateNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(dim, num_experts)

    def forward(self, x_local):
        batch_size = x_local.size(0)
        # print(batch_size)
        # print(x_local.shape)
        scores = self.linear(x_local.reshape(batch_size, -1))
        probs = F.softmax(scores, dim=-1)
        max_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        return max_probs, top_k_indices


class MoLE(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(MoLE, self).__init__()
        '''
        self.area = 120 * 160
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.conv_rgb = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.conv_ir = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.experts = nn.ModuleList([Local_Expert(dim) for _ in range(num_experts)])
        '''
        self.expert = Local_Expert(dim)
        self.att = nn.Sequential(
            Permute(0, 2, 3, 1),
            STMBlock(
                hidden_dim=dim,
                drop_path=0.1,
                ssm_ratio=2.0,
                d_state=16,
                dt_rank="auto",
                mlp_ratio=4.0,
                norm_layer=nn.LayerNorm,
            ),
            Permute(0, 3, 1, 2),
        )          
        self.mlp = Mlp(
                in_features=dim, 
                hidden_features=int(dim * 4.0), 
                out_features=dim
            )
        # self.gate_rgb = GateNetwork(dim * self.area, num_experts // 2, top_k)
        # self.gate_ir = GateNetwork(dim * self.area, num_experts // 2, top_k)
        # self.rgb_local = nn.Conv2d(dim * 5, dim, kernel_size=1, bias=False)
        # self.ir_local = nn.Conv2d(dim * 5, dim, kernel_size=1, bias=False)
        # self.local = nn.Conv2d(dim * 9, dim, kernel_size=1, bias=False)
        # self.refine_expert = NAFBlock(c=dim)
        # self.gmm = Global_Dynamics(dim*4)
        # self.lmm = Local_Dynamics(dim*4)
        # self.enhance_mutual = NAFNet(
        #     img_channel=dim * 9,
        #     embed_dim=dim * 9,
        #     out_channels=dim * 5,
        #     middle_blk_num=2,
        #     enc_blk_nums=[2, 4, 4],
        #     dec_blk_nums=[4, 4, 2],
        # )
        # self.gobal_in = nn.Sequential(
        #     ChannelAttention(dim * 5),
        #     nn.Conv2d(dim * 5, dim, kernel_size=1, bias=False),
        # )
        '''
        self.gate_rgb = GateNetwork(self.dim * self.area, self.num_experts // 2, self.top_k)
        self.gate_ir = GateNetwork(self.dim * self.area, self.num_experts // 2, self.top_k)
        '''

    def forward(self, rgb_local, ir_local, rgb_dense=None, ir_dense=None):
        B, C, H, W = rgb_local.shape
        # print(rgb_local.shape)
        # print(rgb_dense.shape)
        # self.area = H * W
        # device = rgb_local.device
        # self.gate_rgb = GateNetwork(self.dim * self.area, self.num_experts // 2, self.top_k).to(device)
        # self.gate_ir = GateNetwork(self.dim * self.area, self.num_experts // 2, self.top_k).to(device)

        '''
        weights_rgb, index_rgb = self.gate_rgb(rgb_local)
        weights_ir, index_ir = self.gate_ir(ir_local)
        weights = F.softmax(torch.cat([weights_rgb, weights_ir], dim=-1), dim=-1)

        weights_rgb = weights[:, 0 : self.top_k]
        weights_ir = weights[:, self.top_k :]

        rgb_up = self.conv_rgb(rgb_local)
        ir_up = self.conv_ir(ir_local)
        rgb_expert_outputs = torch.zeros_like(rgb_local)
        ir_expert_outputs = torch.zeros_like(ir_local)

        # 计算RGB专家的加权输出
        for k in range(self.top_k):
            idx = index_rgb[:, k].unsqueeze(1)  # 获取第 k 个专家的索引并添加一个维度
            weight = weights_rgb[:, k].view(-1, 1, 1, 1)  # 获取第 k 个专家的权重并添加维度
            for b in range(rgb_local.size(0)):  # 对于每个批次进行处理
                rgb_expert_outputs[b : b + 1] += (
                    # self.experts[idx[b].item()](rgb_local[b : b + 1]) * weight[b]
                    self.experts[idx[b].item()](rgb_up[b : b + 1]) * weight[b]
                )

        # 计算IR专家的加权输出
        for k in range(self.top_k):
            idx = (index_ir[:, k].unsqueeze(1) + self.num_experts // 2)  # 获取第 k 个专家的索引并添加偏移量
            weight = weights_ir[:, k].view(-1, 1, 1, 1)
            for b in range(ir_local.size(0)):
                ir_expert_outputs[b : b + 1] += (
                    # self.experts[idx[b].item()](ir_local[b : b + 1]) * weight[b]
                    self.experts[idx[b].item()](ir_up[b : b + 1]) * weight[b]
                )

        local_features = rgb_expert_outputs + ir_expert_outputs
        '''
        # local_features = self.refine_expert(local_features)
        # rgb_dense, ir_dense = self.gmm(rgb_dense, ir_dense)
        # rgb_dense, ir_dense = self.lmm(rgb_dense, ir_dense)
        # y_local = self.att(local_features) + local_features

        y_local = self.expert(torch.cat([rgb_local, ir_local], dim=1))
        y_local = self.att(y_local) + y_local
        y_local = self.mlp(y_local.flatten(2).transpose(1, 2), H, W).transpose(1, 2).view(B, -1, H, W) + y_local
        '''
        y_local = torch.cat([local_features, rgb_dense, ir_dense], dim=1)
        # print(y_local.shape)
        y_local = self.enhance_mutual(y_local)
        # print(y_local.shape)
        y_local = self.gobal_in(y_local)
        '''
        # print(y_local.shape)
        # rgb_local = torch.cat([rgb_dense, local_features], dim=1)
        # ir_local = torch.cat([ir_dense, local_features], dim=1)

        # y_local = self.local(y_local)
        # rgb_local = self.rgb_local(rgb_local)
        # ir_local = self.ir_local(ir_local)

        return y_local#, rgb_local, ir_local


class MoGE(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(MoGE, self).__init__()
        '''
        self.area = 120 * 160
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Global_Expert(dim) for _ in range(num_experts)])
        # self.experts = nn.ModuleList([Global_Expert_init(input_channels=dim, output_channels=dim) for _ in range(num_experts)])
        self.gate_global = GateNetwork(dim * self.area, num_experts, top_k)
        '''
        self.att = Global_Expert(dim)

    def forward(self, local_features, rgb_local, ir_local):
        B, C, H, W = local_features.shape
        # print(local_features.shape)
        # print(rgb_local.shape)
        # self.area = H * W
        # device = local_features.device
        # self.gate_global = GateNetwork(self.dim * self.area, self.num_experts, self.top_k).to(device)
        '''
        weights, index = self.gate_global(local_features)
        weights = F.softmax(weights, dim=-1)
        # print(weights)
        # print(index)

        outputs = torch.zeros_like(local_features)
        # outputs = torch.zeros_like(local_features[:, 0:32, :, :])

        # print(outputs.shape)
        for k in range(self.top_k):
            idx = index[:, k].unsqueeze(1)
            # print(idx)
            weight = weights[:, k].view(-1, 1, 1, 1)
            # print(weight)
            for b in range(local_features.size(0)):
                # print(weight[b])
                # print(rgb_local[b:b+1].shape)
                outputs[b : b + 1] += (
                    self.experts[idx[b].item()](
                        rgb_local[b : b + 1], ir_local[b : b + 1]
                        # local_features[b : b + 1]
                    )* weight[b])
        '''
        # outputs = F.relu(outputs)
        outputs = self.att(rgb_local, ir_local)
        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # Example usage
    # dim = 32
    # batch_size = 4
    # num_experts = 4
    # top_k = 2

    # model_l = MoLE(dim, num_experts, top_k).to(device)
    # model_g = MoGE(dim, num_experts, top_k).to(device)
    # rgb_local = torch.randn(batch_size, dim, 120, 160).to(device)
    # ir_local = torch.randn(batch_size, dim, 120, 160).to(device)
    # rgb_dense = torch.randn(batch_size, 4*dim, 120, 160).to(device)
    # ir_dense = torch.randn(batch_size, 4*dim, 120, 160).to(device)
    # local = torch.randn(batch_size, dim, 120, 160).to(device)

    # output = model_g(local, local, local)
    # o1, o2, o3 = model_l(rgb_local, ir_local, rgb_dense,  ir_dense)
    # print(output.shape)
    # print(o1.shape)
    # print(o2.shape)
    # print(o3.shape)

    #########################################################################
    # batch_size = 10
    # num_experts = 5
    # input_size = 8

    # gates = torch.rand(batch_size, num_experts)
    # gates[gates < 0.5] = 0
    # inputs = torch.rand(batch_size, input_size)
    # dispatcher = SparseDispatcher(num_experts, gates)
    # expert_inputs = dispatcher.dispatch(inputs)
    # assert len(expert_inputs) == num_experts
    # for expert_input in expert_inputs:
    #     assert expert_input.size(0) == dispatcher._part_sizes[expert_inputs.index(expert_input)]

    # # 测试 combine 方法
    # expert_outputs = [torch.rand(dispatcher._part_sizes[i], input_size) for i in range(num_experts)]
    # combined_output = dispatcher.combine(expert_outputs)
    # assert combined_output.size(0) == batch_size
    # assert combined_output.size(1) == input_size

    # # 测试 expert_to_gates 方法
    # expert_gates = dispatcher.expert_to_gates()
    # assert len(expert_gates) == num_experts
    # for i, gate in enumerate(expert_gates):
    #     assert gate.size(0) == dispatcher._part_sizes[i]

    # print("All tests passed.")

    ####################################################################################
    rgb_local = torch.randn(4, 32, 120, 160).to(device)
    ir_local = torch.randn(4, 32, 120, 160).to(device)
    mole = MoLE(dim=32, num_experts=2, top_k=1).to(device)
    output = mole(rgb_local, ir_local)
    print(output.shape)
    # rgb_dense = torch.randn(4, 4 * 32, 120, 160).to(device)
    # ir_dense = torch.randn(4, 4 * 32, 120, 160).to(device)
    # model = MambaMoE(160, 120, 32, 32, 4, 4, noisy_gating=True, local_k=2, gobal_k=2).to(device)
    # y, loss = model(rgb_local.view(4, -1), ir_local.view(4, -1), rgb_dense, ir_dense)
    # print(y.shape)
    # print(loss)
