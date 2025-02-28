# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

from moe import MoE, LSTMWithMoE, MoEModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='CIFAR10 with Mixture of Experts')
parser.add_argument('--hierarchical', action='store_true', help='Use hierarchical MoE')
parser.add_argument('--lstm', action='store_true', help='Use LSTM with MoE')
parser.add_argument('--num_experts', type=int, default=10, help='Number of experts')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of experts')
parser.add_argument('--num_groups', type=int, default=5, help='Number of groups for hierarchical MoE')
parser.add_argument('--experts_per_group', type=int, default=2, help='Number of experts per group for hierarchical MoE')
parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--log_interval', type=int, default=100, help='How often to log training stats')
parser.add_argument('--plot_experts', action='store_true', help='Plot expert usage distribution')
parser.add_argument('--loss_coef', type=float, default=1e-2, help='Loss coefficient for load balancing')
args = parser.parse_args()

# Print the loss_coef value to verify it's being read correctly

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def plot_expert_load_over_time(expert_batch_counts, title="Expert Load Over Time"):
    """Plot expert load over time"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot each expert's load over time
        for expert_id, counts in expert_batch_counts.items():
            if len(counts) > 0:  # Only plot if we have data
                plt.plot(counts, label=f"Expert {expert_id.split('_')[-1]}")
        
        plt.xlabel("Batch")
        plt.ylabel("Number of samples")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"expert_load_over_time_{timestamp}.png"
        plt.savefig(filename)
        logger.info(f"Expert load over time plot saved to {filename}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting expert load over time: {e}")

def log_model_structure(model):
    """Log detailed information about the model structure"""
    logger.info("=" * 50)
    logger.info("MODEL STRUCTURE DETAILS")
    logger.info("=" * 50)
    
    if isinstance(model, MoEModel):
        logger.info("Model type: MoEModel (LSTM with MoE)")
        lstm_moe = model.lstm_moe
        logger.info(f"LSTM layers: {lstm_moe.num_layers}")
        logger.info(f"Bidirectional: {lstm_moe.bidirectional}")
        logger.info(f"Hidden size: {lstm_moe.hidden_size}")
        
        # Log MoE layers details
        for i, moe_layer in enumerate(lstm_moe.moe_layers):
            logger.info(f"MoE layer {i}:")
            log_moe_structure(moe_layer)
    
    elif isinstance(model, MoE):
        logger.info("Model type: MoE (Mixture of Experts)")
        log_moe_structure(model)
    
    logger.info("=" * 50)

def log_moe_structure(moe):
    """Log detailed information about an MoE layer"""
    logger.info(f"  Input size: {moe.input_size}")
    logger.info(f"  Output size: {moe.output_size}")
    logger.info(f"  Hidden size: {moe.hidden_size}")
    logger.info(f"  Number of experts: {moe.num_experts}")
    logger.info(f"  Top-k experts per sample: {moe.k}")
    logger.info(f"  Noisy gating: {moe.noisy_gating}")
    
    if moe.hierarchical:
        logger.info(f"  Hierarchical MoE: Yes")
        logger.info(f"  Number of groups: {moe.num_groups}")
        logger.info(f"  Experts per group: {moe.experts_per_group}")
        logger.info(f"  Top-k groups per sample: {moe.k_groups}")
        logger.info(f"  Top-k experts per group: {moe.k_experts_per_group}")
    else:
        logger.info(f"  Hierarchical MoE: No")

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Print configuration
    logger.info("=" * 50)
    logger.info("CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Model: {'LSTM+MoE' if args.lstm else 'MoE only'}")
    logger.info(f"MoE Type: {'Hierarchical' if args.hierarchical else 'Flat'}")
    logger.info(f"Number of experts: {args.num_experts}")
    logger.info(f"Hidden size: {args.hidden_size}")
    if args.hierarchical:
        logger.info(f"Number of groups: {args.num_groups}")
        logger.info(f"Experts per group: {args.experts_per_group}")
    if args.lstm:
        logger.info(f"LSTM layers: {args.num_layers}")
        logger.info(f"Bidirectional: {args.bidirectional}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info("=" * 50)
    
    # Create the model based on arguments
    if args.lstm:
        # Create LSTM with MoE model
        # For CIFAR10, reshape images to sequences: [batch_size, 32, 3*32]
        # Each row of the image becomes a timestep with 3*32 features
        net = MoEModel(
            input_size=3*32,  # 3 channels * 32 width
            hidden_size=args.hidden_size,
            output_size=10,  # 10 classes
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            moe_hidden_size=args.hidden_size,
            lstm_dropout=0.1,
            bidirectional=args.bidirectional,
            hierarchical=args.hierarchical,
            num_groups=args.num_groups if args.hierarchical else None,
            experts_per_group=args.experts_per_group if args.hierarchical else None,
            loss_coef=args.loss_coef
        )
    else:
        # Create flat or hierarchical MoE model
        net = MoE(
            input_size=3072,  # 3*32*32
            output_size=10,   # 10 classes
            num_experts=args.num_experts,
            hidden_size=args.hidden_size,
            noisy_gating=True,
            k=4,
            hierarchical=args.hierarchical,
            num_groups=args.num_groups if args.hierarchical else None,
            experts_per_group=args.experts_per_group if args.hierarchical else None,
            loss_coef=args.loss_coef
        )
    
    # Log model structure details
    log_model_structure(net)
    
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Training loop
    logger.info("Starting training...")
    net.train()
    
    # Track metrics
    epoch_losses = []
    epoch_moe_losses = []
    batch_times = []
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        running_moe_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            batch_start_time = time.time()
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if args.lstm:
                # Reshape for LSTM: [batch_size, seq_len, features]
                # For CIFAR10, we treat each row as a timestep
                batch_size = inputs.shape[0]
                inputs_reshaped = inputs.view(batch_size, 32, 3*32)
                outputs, moe_loss = net(inputs_reshaped)
                loss = criterion(outputs, labels)
                total_loss = loss + moe_loss
            else:
                # Flatten for MoE
                inputs = inputs.view(inputs.shape[0], -1)
                outputs, moe_loss = net(inputs)
                loss = criterion(outputs, labels)
                total_loss = loss + moe_loss
                
            total_loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Track batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            # Update running losses
            running_loss += loss.item()
            running_moe_loss += moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss
            
            # Print statistics
            if i % args.log_interval == args.log_interval - 1:
                accuracy = 100 * correct / total
                avg_batch_time = sum(batch_times[-args.log_interval:]) / args.log_interval
                
                logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] '
                      f'loss: {running_loss / args.log_interval:.4f}, '
                      f'moe_loss: {running_moe_loss / args.log_interval:.4f}, '
                      f'accuracy: {accuracy:.2f}%, '
                      f'batch_time: {avg_batch_time:.4f}s')
                
                running_loss = 0.0
                running_moe_loss = 0.0
                correct = 0
                total = 0
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        logger.info(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s')
        
        # Plot expert usage at the end of each epoch if requested
        if args.plot_experts:
            if isinstance(net, MoEModel):
                for i, moe_layer in enumerate(net.lstm_moe.moe_layers):
                    moe_layer.plot_expert_usage(f"Expert Usage - LSTM Layer {i} - Epoch {epoch+1}")
            elif isinstance(net, MoE):
                net.plot_expert_usage(f"Expert Usage - Epoch {epoch+1}")

    logger.info('Finished Training')
    
    # Plot expert load over time
    from moe import expert_batch_counts
    if args.plot_experts and expert_batch_counts:
        plot_expert_load_over_time(expert_batch_counts)

    # Evaluation
    logger.info("Starting evaluation...")
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            if args.lstm:
                # Reshape for LSTM
                batch_size = images.shape[0]
                images_reshaped = images.view(batch_size, 32, 3*32)
                outputs, _ = net(images_reshaped)
            else:
                # Flatten for MoE
                outputs, _ = net(images.view(images.shape[0], -1))
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print overall accuracy
    overall_accuracy = 100 * correct / total
    logger.info(f'Overall accuracy on test set: {overall_accuracy:.2f}%')

    # Print per-class accuracy
    logger.info("Per-class accuracy:")
    for i in range(10):
        class_accuracy = 100 * class_correct[i] / class_total[i]
        logger.info(f'  {classes[i]}: {class_accuracy:.2f}%')
    
    # Final expert usage statistics
    logger.info("=" * 50)
    logger.info("FINAL EXPERT USAGE STATISTICS")
    logger.info("=" * 50)
    
    if isinstance(net, MoEModel):
        for i, moe_layer in enumerate(net.lstm_moe.moe_layers):
            logger.info(f"LSTM Layer {i} MoE:")
            moe_layer.log_expert_usage()
    elif isinstance(net, MoE):
        net.log_expert_usage()
    
    logger.info("=" * 50)

