# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

from moe import MoE, LSTMWithMoE, MoEModel

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
args = parser.parse_args()

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

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Model: {'LSTM+MoE' if args.lstm else 'MoE only'}")
    print(f"  MoE Type: {'Hierarchical' if args.hierarchical else 'Flat'}")
    print(f"  Number of experts: {args.num_experts}")
    print(f"  Hidden size: {args.hidden_size}")
    if args.hierarchical:
        print(f"  Number of groups: {args.num_groups}")
        print(f"  Experts per group: {args.experts_per_group}")
    if args.lstm:
        print(f"  LSTM layers: {args.num_layers}")
        print(f"  Bidirectional: {args.bidirectional}")
    print(f"  Device: {device}")
    
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
            experts_per_group=args.experts_per_group if args.hierarchical else None
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
            experts_per_group=args.experts_per_group if args.hierarchical else None
        )
    
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Training loop
    net.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    # Evaluation
    correct = 0
    total = 0
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

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # Per-class accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
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
                
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

