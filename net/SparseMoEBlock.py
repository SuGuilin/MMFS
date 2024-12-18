import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal
import numpy as np


class SparseDispatcher(nn.Module):
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

    def __init__(self, gate_size, out_dim, device, num_experts, experts, multiply_by_gates=True):
        super().__init__()
        """Create a SparseDispatcher."""
        self._num_experts = num_experts
        self.experts = experts
        self.multiply_by_gates = multiply_by_gates
        self.zeros = torch.zeros(
            int(gate_size),
            int(out_dim),
            requires_grad=True,
            device=device,
        )
    '''
    def dispatch(self, inp, extra=None):
        inp_exp = inp[self._batch_index].squeeze(1)
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
        stitched = torch.cat(expert_out, 0).exp()

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
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
            a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
            and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    '''
    def forward(self, gates, inp):
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, _expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        _part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[_batch_index.flatten()]
        _nonzero_gates = torch.gather(gates_exp, 1, _expert_index)

        inp_exp = inp[_batch_index].squeeze(1)
        expert_inputs = torch.split(inp_exp, _part_sizes, dim=0)
        # split nonzero gates for each expert
        gates_outs = torch.split(_nonzero_gates, _part_sizes, dim=0)
        
        expert_out = [
            self.experts[i](expert_inputs[i]) for i in range(self._num_experts)
        ]

        stitched = torch.cat(expert_out, 0).exp()

        if self.multiply_by_gates:
            stitched = stitched.mul(_nonzero_gates)
        self.zeros.to(stitched.device)
        # print(gates.size(0), expert_out[-1].size(1))
        # zeros = torch.zeros(
        #     gates.size(0),
        #     expert_out[-1].size(1),
        #     # requires_grad=True,
        #     device=stitched.device,
        # )
        # combine samples that have been processed by the same k experts
        combined = self.zeros.index_add(0, _batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()



class Mlp(nn.Module):
    def __init__(self, in_feat, h_feat=None, out_feat=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_feat = out_feat or in_feat
        h_feat = h_feat or in_feat
        self.fc1 = nn.Linear(in_feat, h_feat)
        self.act = act_layer()
        self.fc2 = nn.Linear(h_feat, out_feat)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return x


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
        input_size: integer - size of the input
        output_size: integer - size of the input
        num_experts: an integer - number of experts
        noisy_gating: a boolean
        k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        gate_size,
        device,
        input_size,
        output_size,
        mlp_ratio,
        num_experts,
        noisy_gating=True,
        use_experts=2,
    ):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = use_experts
        # instantiate experts
        self.experts = nn.ModuleList([
            Mlp(input_size, h_feat=int(input_size * mlp_ratio), out_feat=output_size)
            for i in range(self.num_experts)
        ])
        self.w_gate = nn.Parameter(
            torch.randn(2 * input_size, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(2 * input_size, num_experts), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.beta  = nn.Parameter(torch.zeros((1, input_size, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, input_size, 1, 1)), requires_grad=True)
        self.dispatcher = SparseDispatcher(gate_size, output_size, device, self.num_experts, self.experts)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

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

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
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
            noisy_top_values: a `Tensor` of shape [batch, m]. "values" Output of 
                tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
            a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        # if clean_values in topk, get the position of threshold
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        # extract threshold
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        # judge noisy_valued if greater than threshold
        is_in = torch.gt(noisy_values, threshold_if_in)
        # if clean_values not in topk, get the position of threshold
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the topk.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
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
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, prompt):
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
        # import pdb
        # pdb.set_trace()
        x = x * self.gamma + self.beta

        B, C, H, W = x.shape
        prompt = prompt.unsqueeze(-1).unsqueeze(-1).expand_as(x)

        x = rearrange(x, "b c h w -> (b h w) c")
        prompt = rearrange(prompt, "b c h w -> (b h w) c")
        # print(x.shape, prompt.shape)

        # prompt and image content work together to improve expert understanding
        x_gating = torch.cat((x, prompt), dim=1)  # [B, 2C, H, W]

        gates, load = self.noisy_top_k_gating(x_gating, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        # one for the index and one for the weight value
        loss = self.cv_squared(importance) + self.cv_squared(load)

        # dispatcher = SparseDispatcher(self.num_experts, gates)
        # expert_inputs = dispatcher.dispatch(x)
        # gates = dispatcher.expert_to_gates()
        # expert_outputs = [
        #     self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        # ]
        # y = dispatcher.combine(expert_outputs)
        y = self.dispatcher(gates, x)
        print(y.shape)
        print(B)
        print(H)
        print(W)
        y = rearrange(y, "(b h w) c -> b c h w", b=B, h=H, w=W)
        
        return y, loss


class SparseMoEBlock(nn.Module):
    def __init__(self, gate_size, device, atom_dim, dim, ffn_expansion_factor):
        super(SparseMoEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(atom_dim, dim)
        self.moe = MoE(
            gate_size=gate_size,
            device=device,
            input_size=dim,
            output_size=dim,
            mlp_ratio=ffn_expansion_factor,
            num_experts=4,
            noisy_gating=True,
            use_experts=2,
        )

    def forward(self, x, prompt):
        prompt = self.avg_pool(prompt.permute(0, 2, 1)).squeeze(-1)
        d = self.fc(prompt)
        out, loss = self.moe(x, d)
        return out + x, loss
    

class Channel_Routing(nn.Module):
    def __init__(self, atom_dim, dim):
        super(Channel_Routing, self).__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True) 

    def forward(self, x, prompt):
        gating_factors = torch.sigmoid(self.fc(prompt))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        out = x * self.gamma + self.beta  
        out = out * gating_factors 
        return x + out


class Spatial_Routing(nn.Module):
    def __init__(self, atom_dim, dim, ffn_expansion_factor):
        super(Spatial_Routing, self).__init__() 
        self.fc = nn.Linear(atom_dim, dim) 
        self.moe = MoE(dim, dim, mlp_ratio=ffn_expansion_factor, num_experts=4, noisy_gating=True, use_experts=2) 

    def forward(self, x, prompt): 
        d = self.fc(prompt) 
        out, loss = self.moe(x, d) 
        return out + x, loss


if __name__ == "__main__":
    x = torch.randn(1, 96, 120, 160)
    prompt = torch.randn(1, 256, 768)
    # prompt = prompt.view(1, -1)
    # b, c = prompt.shape
    # print(c)

    model = SparseMoEBlock(atom_dim=768, dim=96, ffn_expansion_factor=2)
    out, loss = model(x, prompt)
    print(out.shape)