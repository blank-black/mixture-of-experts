# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


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

    def dispatch(self, inp):
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
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
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


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    hierarchical: a boolean - whether to use hierarchical MoE
    num_groups: an integer - number of expert groups for hierarchical MoE
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4, 
                 hierarchical=False, num_groups=None, experts_per_group=None):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.hierarchical = hierarchical
        
        # Setup for hierarchical MoE
        if hierarchical:
            assert num_groups is not None, "num_groups must be specified for hierarchical MoE"
            assert experts_per_group is not None, "experts_per_group must be specified for hierarchical MoE"
            assert num_groups * experts_per_group == num_experts, "num_groups * experts_per_group must equal num_experts"
            
            self.num_groups = num_groups
            self.experts_per_group = experts_per_group
            self.k_groups = min(2, num_groups)  # Number of groups to select
            self.k_experts_per_group = min(2, experts_per_group)  # Number of experts to select within each group
            
            # Primary gating network (selects groups)
            self.w_gate_primary = nn.Parameter(torch.zeros(input_size, num_groups), requires_grad=True)
            self.w_noise_primary = nn.Parameter(torch.zeros(input_size, num_groups), requires_grad=True)
            
            # Secondary gating networks (one per group, selects experts within group)
            self.w_gate_secondary = nn.ParameterList([
                nn.Parameter(torch.zeros(input_size, experts_per_group), requires_grad=True)
                for _ in range(num_groups)
            ])
            self.w_noise_secondary = nn.ParameterList([
                nn.Parameter(torch.zeros(input_size, experts_per_group), requires_grad=True)
                for _ in range(num_groups)
            ])
            
            # Instantiate experts (grouped)
            self.experts = nn.ModuleList([
                nn.ModuleList([
                    MLP(self.input_size, self.output_size, self.hidden_size) 
                    for _ in range(experts_per_group)
                ])
                for _ in range(num_groups)
            ])
        else:
            # Flat MoE (original implementation)
            # instantiate experts
            self.experts = nn.ModuleList([
                MLP(self.input_size, self.output_size, self.hidden_size) 
                for _ in range(self.num_experts)
            ])
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        if not hierarchical:
            assert(self.k <= self.num_experts)

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
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values, k):
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
        k: integer - number of elements to select
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, k=None, w_gate=None, w_noise=None):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
            k: number of experts to select
            w_gate: gating weights
            w_noise: noise weights
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        if k is None:
            k = self.k
            
        if w_gate is None:
            w_gate = self.w_gate
            
        if w_noise is None:
            w_noise = self.w_noise
            
        clean_logits = x @ w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(k + 1, logits.size(1)), dim=1)
        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and k < logits.size(1) and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits, k)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def hierarchical_gating(self, x, train, noise_epsilon=1e-2):
        """Hierarchical gating for MoE.
        First selects groups of experts, then selects experts within each group.
        
        Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
            
        Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        batch_size = x.size(0)
        
        # Primary gating - select groups
        primary_gates, primary_load = self.noisy_top_k_gating(
            x, train, noise_epsilon, 
            k=self.k_groups, 
            w_gate=self.w_gate_primary, 
            w_noise=self.w_noise_primary
        )
        
        # Get the indices of the selected groups for each batch element
        _, primary_indices = primary_gates.topk(self.k_groups, dim=1)
        
        # Initialize the final gates tensor - without requires_grad=True
        final_gates = torch.zeros(batch_size, self.num_experts, device=x.device)
        
        # For each batch element, select experts from the chosen groups
        for batch_idx in range(batch_size):
            # Get the groups selected for this batch element
            selected_groups = primary_indices[batch_idx]
            
            # For each selected group
            for group_idx in selected_groups:
                # Get the group weight from primary gating
                group_weight = primary_gates[batch_idx, group_idx]
                
                # Secondary gating - select experts within the group
                secondary_gates, _ = self.noisy_top_k_gating(
                    x[batch_idx:batch_idx+1], 
                    train, 
                    noise_epsilon,
                    k=self.k_experts_per_group,
                    w_gate=self.w_gate_secondary[group_idx],
                    w_noise=self.w_noise_secondary[group_idx]
                )
                
                # Scale the expert gates by the group weight
                scaled_gates = secondary_gates * group_weight
                
                # Map from local expert indices to global expert indices
                global_expert_offset = group_idx * self.experts_per_group
                for local_expert_idx in range(self.experts_per_group):
                    global_expert_idx = global_expert_offset + local_expert_idx
                    final_gates[batch_idx, global_expert_idx] = scaled_gates[0, local_expert_idx]
        
        # Calculate load balancing
        load = self._gates_to_load(final_gates)
        
        return final_gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        if self.hierarchical:
            gates, load = self.hierarchical_gating(x, self.training)
            
            # Calculate importance loss
            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef
            
            # Dispatch to experts
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            gates = dispatcher.expert_to_gates()
            
            # Process inputs through experts
            expert_outputs = []
            for i in range(self.num_experts):
                # Map from flat expert index to group and local expert indices
                group_idx = i // self.experts_per_group
                local_expert_idx = i % self.experts_per_group
                
                # Only process if this expert received any inputs
                if expert_inputs[i].size(0) > 0:
                    expert_output = self.experts[group_idx][local_expert_idx](expert_inputs[i])
                    expert_outputs.append(expert_output)
                else:
                    # Create empty tensor with correct shape for this expert
                    expert_outputs.append(torch.zeros((0, self.output_size), device=x.device))
            
            y = dispatcher.combine(expert_outputs)
        else:
            # Original flat MoE implementation
            gates, load = self.noisy_top_k_gating(x, self.training)
            
            # Calculate importance loss
            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef
            
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            gates = dispatcher.expert_to_gates()
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            y = dispatcher.combine(expert_outputs)
            
        return y, loss


class LSTMWithMoE(nn.Module):
    """LSTM with Mixture of Experts layer as described in the paper.
    
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers
        moe_input_size: Size of input to MoE layer
        moe_output_size: Size of output from MoE layer
        num_experts: Number of experts in MoE
        moe_hidden_size: Hidden size of each expert
        noisy_gating: Whether to use noisy gating
        k: How many experts to use for each batch element
        hierarchical: Whether to use hierarchical MoE
        num_groups: Number of expert groups for hierarchical MoE
        experts_per_group: Number of experts per group for hierarchical MoE
        dropout: Dropout probability (0 means no dropout)
        bidirectional: If True, becomes a bidirectional LSTM
    """
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 moe_input_size=None, moe_output_size=None, num_experts=8, 
                 moe_hidden_size=128, noisy_gating=True, k=4,
                 hierarchical=False, num_groups=None, experts_per_group=None,
                 dropout=0, bidirectional=False):
        super(LSTMWithMoE, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Default MoE input/output sizes if not specified
        if moe_input_size is None:
            moe_input_size = hidden_size * (2 if bidirectional else 1)
        if moe_output_size is None:
            moe_output_size = hidden_size
            
        self.moe_input_size = moe_input_size
        self.moe_output_size = moe_output_size
        
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(input_size, hidden_size, 1, 
                   dropout=0, bidirectional=bidirectional, batch_first=True)
        )
        
        # Create MoE layers between LSTM layers
        self.moe_layers = nn.ModuleList()
        
        # Number of MoE layers = num_layers - 1
        for i in range(num_layers - 1):
            # Create MoE layer
            moe = MoE(
                input_size=moe_input_size,
                output_size=moe_output_size,
                num_experts=num_experts,
                hidden_size=moe_hidden_size,
                noisy_gating=noisy_gating,
                k=k,
                hierarchical=hierarchical,
                num_groups=num_groups,
                experts_per_group=experts_per_group
            )
            self.moe_layers.append(moe)
            
            # Create next LSTM layer
            lstm_input_size = moe_output_size
            self.lstm_layers.append(
                nn.LSTM(lstm_input_size, hidden_size, 1, 
                       dropout=0, bidirectional=bidirectional, batch_first=True)
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            hidden: Initial hidden state
            
        Returns:
            output: Output features (h_t) from the last layer of the LSTM, for each t
            hidden: Tuple containing the final hidden state and cell state
            moe_losses: List of MoE auxiliary losses
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        moe_losses = []
        
        # Initialize hidden state if not provided
        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(1 * num_directions, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(1 * num_directions, batch_size, self.hidden_size, device=x.device)
            hidden = (h0, c0)
        
        # Process through stacked LSTM and MoE layers
        output = x
        current_hidden = hidden
        
        for i in range(self.num_layers):
            # Process through LSTM layer
            output, current_hidden = self.lstm_layers[i](output, current_hidden)
            
            # Apply dropout
            output = self.dropout(output)
            
            # If not the last layer, process through MoE
            if i < self.num_layers - 1:
                # Reshape for MoE: [batch_size * seq_len, hidden_size]
                moe_input = output.contiguous().view(-1, output.size(2))
                
                # Process through MoE
                moe_output, moe_loss = self.moe_layers[i](moe_input)
                moe_losses.append(moe_loss)
                
                # Reshape back: [batch_size, seq_len, moe_output_size]
                output = moe_output.view(batch_size, seq_len, -1)
        
        return output, current_hidden, moe_losses


class MoEModel(nn.Module):
    """A model that combines LSTM and MoE layers for sequence processing tasks.
    
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        output_size: The number of output features
        num_layers: Number of recurrent layers
        num_experts: Number of experts in MoE
        moe_hidden_size: Hidden size of each expert
        lstm_dropout: Dropout probability for LSTM (0 means no dropout)
        bidirectional: If True, becomes a bidirectional LSTM
        hierarchical: Whether to use hierarchical MoE
        num_groups: Number of expert groups for hierarchical MoE
        experts_per_group: Number of experts per group for hierarchical MoE
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, 
                 num_experts=8, moe_hidden_size=128, lstm_dropout=0.1, 
                 bidirectional=False, hierarchical=False, num_groups=None, 
                 experts_per_group=None):
        super(MoEModel, self).__init__()
        
        # LSTM with MoE layers
        self.lstm_moe = LSTMWithMoE(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_experts=num_experts,
            moe_hidden_size=moe_hidden_size,
            noisy_gating=True,
            k=4,
            hierarchical=hierarchical,
            num_groups=num_groups,
            experts_per_group=experts_per_group,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.output_layer = nn.Linear(lstm_output_size, output_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            output: Output tensor of shape [batch_size, output_size]
            moe_losses: List of MoE auxiliary losses
        """
        # Process through LSTM with MoE
        lstm_output, _, moe_losses = self.lstm_moe(x)
        
        # Take the output of the last time step
        last_output = lstm_output[:, -1, :]
        
        # Process through output layer
        output = self.output_layer(last_output)
        
        # Sum all MoE losses
        total_moe_loss = sum(moe_losses) if moe_losses else torch.tensor(0.0, device=x.device)
        
        return output, total_moe_loss
