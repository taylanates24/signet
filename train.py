from model import SigNet, ContrastiveLoss
import os
from data import get_data_loader
from PIL import ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from metrics import accuracy
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import numpy as np

def train(model, optimizer, criterion, dataloader, writer, epoch, log_interval=50):
    model.train()
    running_loss = 0
    number_samples = 0

    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()
        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        loss.backward()
        optimizer.step()

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)
        
        # Log batch loss to TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
        
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            avg_loss = running_loss / number_samples
            print('{}/{}: Loss: {:.4f}'.format(batch_idx+1, len(dataloader), avg_loss))
            writer.add_scalar('Loss/Train_Running', avg_loss, global_step)
            running_loss = 0
            number_samples = 0

@torch.no_grad()
def eval(model, criterion, dataloader, writer, epoch, exp_dir, log_interval=50):
    model.eval()
    running_loss = 0
    number_samples = 0

    distances = []

    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        distances.extend(zip(torch.pairwise_distance(x1, x2, 2).cpu().tolist(), y.cpu().tolist()))

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx+1, len(dataloader), running_loss / number_samples))

    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy, best_threshold = accuracy(distances, y)
    avg_loss = running_loss / number_samples
    
    print(f'Max accuracy: {max_accuracy:.4f} at threshold: {best_threshold:.4f}')
    
    # Log evaluation metrics to TensorBoard
    writer.add_scalar('Loss/Validation', avg_loss, epoch)
    writer.add_scalar('Accuracy/Validation', max_accuracy, epoch)
    writer.add_scalar('Threshold/Best', best_threshold, epoch)
    
    # Generate and log scatter plot
    fig = plt.figure()
    
    # Separate genuine and forged distances
    genuine_distances = distances[y == 1]
    forged_distances = distances[y == 0]
    
    # Plot genuine and forged distances
    plt.scatter(np.arange(len(genuine_distances)), genuine_distances, color='green', label='Genuine', alpha=0.5)
    plt.scatter(np.arange(len(forged_distances)), forged_distances, color='red', label='Forged', alpha=0.5)
    
    # Add threshold line
    plt.axhline(y=best_threshold, color='blue', linestyle='--', label=f'Threshold ({best_threshold:.4f})')
    
    plt.title('Distances of Genuine and Forged Signatures')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('Evaluation/Distances', fig, epoch)
    
    figures_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, f'distances_epoch_{epoch:03d}.png')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f'Saved distance plot to {fig_path}')
    
    return avg_loss, max_accuracy

if __name__ == "__main__":
    parser = ArgumentParser(description='SigNet Training Script')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed for reproducibility')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, choices=['data/CEDAR', 'sign_data'], default='data/CEDAR', help='Dataset directory path')
    parser.add_argument('--img_height', type=int, default=155, help='Image height for resizing')
    parser.add_argument('--img_width', type=int, default=220, help='Image width for resizing')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, choices=['rmsprop', 'adam', 'sgd'], default='adam', help='Optimizer type')
    parser.add_argument('--eps', type=float, default=1e-8, help='Optimizer epsilon (for RMSprop and Adam)')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for RMSprop and SGD)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_step', type=int, default=5, help='Learning rate scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Learning rate scheduler gamma')
    
    # Loss function parameters
    parser.add_argument('--alpha', type=float, default=1.0, help='Contrastive loss alpha parameter')
    parser.add_argument('--beta', type=float, default=1.0, help='Contrastive loss beta parameter')
    parser.add_argument('--margin', type=float, default=1.0, help='Contrastive loss margin parameter')
    
    # Dropout parameters
    parser.add_argument('--dropout_conv_p', type=float, default=0.2, help='Dropout probability for convolutional layers')
    parser.add_argument('--dropout_fc_p', type=float, default=0.3, help='Dropout probability for fully connected layers')
    
    # Experiment organization parameters
    parser.add_argument('--exp_name', type=str, default='signet_exp', help='Experiment name for organizing outputs')
    parser.add_argument('--exp_root', type=str, default='experiments', help='Root directory for all experiments')
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval for training progress')
    
    # Model graph logging
    parser.add_argument('--log_model_graph', action='store_true', default=True, help='Log model architecture to TensorBoard')
    
    args = parser.parse_args()
    print(args)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # Create timestamped experiment directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.exp_root, f'{args.exp_name}_{timestamp}')
    
    # Create subdirectories for this experiment
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    
    # Create all necessary directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    print(f'Experiment directory: {exp_dir}')
    print(f'Checkpoints will be saved to: {checkpoint_dir}')
    print(f'TensorBoard logs will be saved to: {tensorboard_dir}')

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and criterion
    model = SigNet(dropout_conv_p=args.dropout_conv_p, dropout_fc_p=args.dropout_fc_p).to(device)
    criterion = ContrastiveLoss(alpha=args.alpha, beta=args.beta, margin=args.margin).to(device)
    
    # Initialize optimizer
    if args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=args.eps, 
                                weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps, 
                             weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                            weight_decay=args.weight_decay, momentum=args.momentum)
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.scheduler_step, args.scheduler_gamma)

    # Save experiment configuration
    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write(f"Experiment: {args.exp_name}_{timestamp}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {device}\n")
        f.write("="*50 + "\n")
        f.write("Configuration:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print(f'Configuration saved to: {config_path}')

    # Log hyperparameters to TensorBoard
    writer.add_hparams({
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'seed': args.seed,
        'dataset': args.dataset,
        'img_height': args.img_height,
        'img_width': args.img_width,
        'optimizer': args.optimizer,
        'eps': args.eps,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'scheduler_step': args.scheduler_step,
        'scheduler_gamma': args.scheduler_gamma,
        'criterion_alpha': args.alpha,
        'criterion_beta': args.beta,
        'criterion_margin': args.margin,
        'dropout_conv_p': args.dropout_conv_p,
        'dropout_fc_p': args.dropout_fc_p,
    }, {})

    # Image transformations
    image_transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        ImageOps.invert,
        transforms.ToTensor(),
        # TODO: add normalize
    ])

    # Data loaders
    trainloader = get_data_loader(is_train=True, batch_size=args.batch_size, 
                                image_transform=image_transform, dataset_dir=args.dataset)
    testloader = get_data_loader(is_train=False, batch_size=args.batch_size, 
                               image_transform=image_transform, dataset_dir=args.dataset)

    model.train()
    print(model)
    
    # Log model architecture to TensorBoard (optional)
    if args.log_model_graph:
        dummy_input = torch.randn(1, 1, args.img_height, args.img_width).to(device)
        writer.add_graph(model, (dummy_input, dummy_input))
    
    # Training loop
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs))
        print('Training', '-'*20)
        train(model, optimizer, criterion, trainloader, writer, epoch, args.log_interval)
        print('Evaluating', '-'*20)
        loss, acc = eval(model, criterion, testloader, writer, epoch, exp_dir, args.log_interval)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        scheduler.step()

        # Save checkpoint with timestamp in filename
        to_save = {
            'model': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optim': optimizer.state_dict(),
            'args': args,
            'epoch': epoch,
            'timestamp': timestamp,
        }

        checkpoint_filename = f'{timestamp}_epoch_{epoch:03d}_loss_{loss:.4f}_acc_{acc:.4f}.pt'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        print(f'Saving checkpoint: {checkpoint_path}')
        torch.save(to_save, checkpoint_path)

    # Save final summary
    summary_path = os.path.join(exp_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Experiment Summary: {args.exp_name}_{timestamp}\n")
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {args.num_epochs}\n")
        f.write(f"Final validation loss: {loss:.4f}\n")
        f.write(f"Final validation accuracy: {acc:.4f}\n")
        f.write(f"Device used: {device}\n")
        f.write("="*50 + "\n")
        f.write("Final model checkpoint: " + checkpoint_filename + "\n")

    # Close the TensorBoard writer
    writer.close()
    print('Done')
    print(f'Experiment completed: {exp_dir}')
    print(f'To view TensorBoard logs, run: tensorboard --logdir={tensorboard_dir}')
    print(f'To view all experiments, run: tensorboard --logdir={args.exp_root}')
