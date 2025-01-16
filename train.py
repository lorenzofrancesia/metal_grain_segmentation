import argparse
import logging 
import torch
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader # Unless I write my own data loader
from torch import optim
import sys
from pathlib import Path

# Add the project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

#Models
from models.unet import UNet
from models.u2net import U2Net

def get_args():
    parser = argparse.ArgumentParser(description='Train a U-Net model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model', type=str, default='unet', help='Model to train')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model checkpoints')
    parser.add_argument('--log_file', type=str, default='train.log', help='File to save training logs')
    
    return parser.parse_args()


def get_model(args):
    if args.model == 'unet':
        model = UNet()
    elif args.model == 'u2net':
        model = U2Net()
    else:
        raise ValueError('Model type not recognized')
        
    return model
