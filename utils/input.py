import argparse


#Models
from models.unet import UNet
from models.u2net import U2Net
from models.unetpp import UNetPP

def get_args_train():
    
    parser = argparse.ArgumentParser(description='Train a U-Net model')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--model', type=str, default='unet', help='Model to train')
    parser.add_argument('--output_dir', type=str, default='../output', help='Directory to save model checkpoints')
    parser.add_argument("--resume", default=False, action="store_true", help="Resume traing from last model saved")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for resuming training')
    
    return parser.parse_args()

def get_args_test():
    
    parser = argparse.ArgumentParser(description='Test a U-Net model')
    parser.add_argument('--model_path', type=str, help='Path to the model to test')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the model to test')
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset')
    
    return parser.parse_args()

def get_model(args):
    
    if args.model == 'unet':
        model = UNet()
    elif args.model == 'u2net':
        model = U2Net()
    elif args.model == 'unet++':
        model = UNetPP()
    else:
        raise ValueError('Model type not recognized')
        
    return model

