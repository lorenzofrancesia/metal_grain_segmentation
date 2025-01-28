import argparse
import torchseg
import torch


#Models
from loss.tversky import TverskyLoss, FocalTverskyLoss
from loss.iou import IoULoss

def get_args_train():
    
    parser = argparse.ArgumentParser(description='Train a U-Net model')
    
    # Model parameters 
    parser.add_argument('--model', type=str, default='Unet', help='Model to train')
    parser.add_argument("--dropout", type=float, default=0, help='Dropout probability.')
    
    # Encoder parameters
    parser.add_argument('--encoder', type=str, default='resnet152', help='Model to train.')
    parser.add_argument('--weights', default=None, action="store_true", help="Utilize pretrained weights.")
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer if supported.')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler.')
    parser.add_argument('--start_factor', type=float, default=0.03, help='Start factor for LinearLR.')
    parser.add_argument('--end_factor', type=float, default=1.0, help='End factor for LinearLR.')
    parser.add_argument('--iterations', type=int, default=10, help='Iterations for LinearLR.')
    parser.add_argument('--t_max', type=int, default=10, help='T-max for CosineAnnealing.')
    parser.add_argument('--eta_min', type=float, default=0, help='Eta-min for CosineAnnealing.')

    # Loss function parameters
    parser.add_argument('--loss_function', type=str, default='FocalTversky', help='Loss Function.')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha for FocalTversky and Tversky.')
    parser.add_argument('--beta', type=float, default=0.3, help='Beta for FocalTversky and Tversky.')
    parser.add_argument('--gamma', type=float, default=1.3333, help='Gamma for FocalTversky.')
    
    # Directories
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing the dataset.')
    parser.add_argument('--output_dir', type=str, default='../runs', help='Directory to save model checkpoints.')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    
    # Dataset parameters 
    parser.add_argument("--normalize", default=False, action="store_true", help="Activate normalization.")
    parser.add_argument('--transform', type=str, default='transforms.ToTensor', help='Transform to apply to the dataset.')
    
    # To implement
    parser.add_argument("--resume", default=False, action="store_true", help="Resume traing from last model saved")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for resuming training')

    
    return parser.parse_args()



def get_model(args, aux_params=None):

    weights = "imagenet" if bool(args.weights) else None
        
    try:
        if args.model == 'U-Net' or args.model == 'Unet':
            model = torchseg.Unet(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                )
            
        elif args.model == 'U-Net++':
            model = torchseg.UnetPlusPlus(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                )
            
        elif args.model == 'MAnet':
            model = torchseg.MAnet(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                )   
            
        elif args.model == 'LinkNet':
            model = torchseg.Linknet(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                )  
        
        elif args.model == 'FPN':
            model = torchseg.FPN(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                ) 
            
        elif args.model == 'PSPNet':
            model = torchseg.PSPNet(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                )    
            
        elif args.model == 'PAN':
            model = torchseg.PAN(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                )    
            
        elif args.model == 'DeepLabV3':
            model = torchseg.DeepLabV3(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                )  
            
        elif args.model == 'DeepLabV3+':
            model = torchseg.DeepLabV3Plus(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params
                )          
        
        else:
            raise ValueError('Model type not recognized')

        return model
    
    except Exception as e:
        print(e)
        raise ValueError

def get_optimizer(args, model):
    try:
        if args.optimizer == "Adam":
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
        
        elif args.optimizer == "SGD":
            
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            
        else:
            raise ValueError('Optimizer type not recognized')
            
        return optimizer

    except Exception as e:
        print(e)
        raise ValueError

def get_scheduler(args, optimizer):

    if args.scheduler is None or args.scheduler == "None":
        return None
    
    elif args.scheduler == "LinearLR":
        
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.start_factor, end_factor=args.end_factor, total_iters=args.iterations )
    
    elif args.scheduler == "CosineAnnealingLR":
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
        
    else:
        raise ValueError('Scheduler type not recognized')
        
    return scheduler

def get_loss_function(args):
    
    if args.loss_function == "FocalTversky":
        
        loss_function = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        
    elif args.loss_function == "Tversky":
        
        loss_function = TverskyLoss(alpha=args.alpha, beta=args.beta)
        
    elif args.loss_function == "IoU":
        
        loss_function = IoULoss()
        
    else:
        raise ValueError('Loss function type not recognized')
        
    return loss_function
         

def get_args_test():
    
    parser = argparse.ArgumentParser(description='Test a U-Net model')
    parser.add_argument('--model', type=str, default='unet', help='Model for testing')
    parser.add_argument('--model_path', type=str, help='Path to the model to test')
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument("--normalize", default=False, action="store_true", help="Activate normalization")
    
    return parser.parse_args()
