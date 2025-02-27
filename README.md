# Mixture of Experts (MoE) Implementation

This repository contains an implementation of the Sparsely-Gated Mixture-of-Experts (MoE) layer as described in the paper ["Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538) by Shazeer et al.

## Features

- **Flat MoE**: The original Mixture of Experts implementation with a single gating network
- **Hierarchical MoE**: Two-level gating network as described in the paper
- **LSTM Integration**: LSTM layers with MoE layers in between, as used in the paper's language models
- **Noisy Top-K Gating**: Implementation of the noisy gating mechanism for load balancing

## Requirements

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
```

## Usage

### Basic Usage

The repository includes examples to demonstrate the usage of the MoE implementation:

#### CIFAR10 Example

```bash
# Run with flat MoE (original implementation)
python cifar10_example.py

# Run with hierarchical MoE
python cifar10_example.py --hierarchical --num_groups 5 --experts_per_group 2

# Run with LSTM + MoE
python cifar10_example.py --lstm --num_layers 2

# Run with LSTM + hierarchical MoE
python cifar10_example.py --lstm --hierarchical --num_groups 5 --experts_per_group 2 --num_layers 2
```

#### Language Model Example

```bash
# Run with flat MoE
python language_model_example.py

# Run with hierarchical MoE
python language_model_example.py --hierarchical --num_groups 4 --experts_per_group 2

# Run with bidirectional LSTM
python language_model_example.py --bidirectional

# Run with more LSTM layers
python language_model_example.py --num_layers 3
```

### Command Line Arguments

#### CIFAR10 Example

- `--hierarchical`: Use hierarchical MoE instead of flat MoE
- `--lstm`: Use LSTM with MoE layers
- `--num_experts`: Number of experts (default: 10)
- `--hidden_size`: Hidden size of experts (default: 128)
- `--num_groups`: Number of groups for hierarchical MoE (default: 5)
- `--experts_per_group`: Number of experts per group for hierarchical MoE (default: 2)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--bidirectional`: Use bidirectional LSTM
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)

#### Language Model Example

- `--hierarchical`: Use hierarchical MoE instead of flat MoE
- `--num_experts`: Number of experts (default: 8)
- `--hidden_size`: Hidden size of LSTM and experts (default: 128)
- `--num_groups`: Number of groups for hierarchical MoE (default: 4)
- `--experts_per_group`: Number of experts per group for hierarchical MoE (default: 2)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--bidirectional`: Use bidirectional LSTM
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--seq_length`: Sequence length for language modeling (default: 50)

## Implementation Details

### Flat MoE

The flat MoE implementation uses a single gating network to select which experts to use for each input. The gating network outputs a sparse vector of weights, and only the top-k experts are used for each input.

### Hierarchical MoE

The hierarchical MoE implementation uses a two-level gating network:
1. The primary gating network selects which groups of experts to use
2. The secondary gating networks (one per group) select which experts within each selected group to use

This allows for more efficient routing of inputs to experts, especially when the number of experts is large.

### LSTM with MoE

The LSTM with MoE implementation places MoE layers between stacked LSTM layers, as described in the paper. This allows the model to use different experts for different types of inputs, while still maintaining the sequential processing capabilities of LSTMs.

## Paper Implementation Notes

This implementation follows the architecture described in the paper:

1. For language models, the paper uses stacked LSTM layers with MoE layers in between
2. The MoE layer is applied to the output of each LSTM layer (except the last)
3. The hierarchical MoE is used when the number of experts is large (thousands)
4. The paper uses noisy top-k gating to encourage load balancing among experts

## Examples

### CIFAR10 Example

The CIFAR10 example demonstrates how to use the MoE layer for image classification. It can be configured to use either a flat MoE, a hierarchical MoE, or an LSTM with MoE layers.

### Language Model Example

The language model example demonstrates how to use the LSTM with MoE layers for character-level language modeling. It includes a simple character-level dataset and text generation capabilities.

## Citation

If you use this code, please cite the original paper:

```
@article{shazeer2017outrageously,
  title={Outrageously large neural networks: The sparsely-gated mixture-of-experts layer},
  author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
  journal={arXiv preprint arXiv:1701.06538},
  year={2017}
}
```






# Requirements

To install the requirements run:

```pip install -r requirements.py```