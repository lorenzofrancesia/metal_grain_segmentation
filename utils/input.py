import argparse
import torchseg
import torch
from torchvision.transforms import transforms
import re
import ast


#Models
from loss.tversky import TverskyLoss, FocalTverskyLoss
from loss.iou import IoULoss
from loss.topoloss import TopologicalLoss
from loss.focal import FocalLoss
from loss.dice import DiceLoss, LCDiceLoss

def get_args_train():
    
    parser = argparse.ArgumentParser(description='Train a U-Net model')
    
    # Model parameters 
    parser.add_argument_group(title="Model Parameters", description="Settings related to the model architecture")
    parser.add_argument('--model', type=str, default='Unet', help='Model to train')
    parser.add_argument('--attention', type=str, default='None', help='Attention type')
    parser.add_argument('--batchnorm', type=str, default='True', help='Batchnorm')
    
    # Encoder parameters
    parser.add_argument_group(title="Encoder Parameters", description="Settings for the encoder part of the model")
    parser.add_argument('--encoder', type=str, default='resnet152', help='Model to train.')
    parser.add_argument('--pretrained_weights', default=None, action="store_true", help="Utilize pretrained weights.")
    parser.add_argument('--freeze_backbone', default=None, action="store_true", help="Freezes encoder weights.")
    
    # Optimizer parameters
    parser.add_argument_group(title="Optimizer Parameters", description="Settings for the optimizer")
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer if supported.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer if supported.')
    
    # Warmup parameters
    parser.add_argument_group(title="Warmup Parameters", description="Settings for learning rate warmup")
    parser.add_argument('--warmup_scheduler', type=str, default=None, help='Warmup scheduler.')
    parser.add_argument('--warmup_steps', type=int, default=3, help='Warmup steps for LinearLR.')

    # Scheduler parameters
    parser.add_argument_group(title="Scheduler Parameters", description="Settings for learning rate scheduling")
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler.')
    parser.add_argument('--start_factor', type=float, default=0.03, help='Start factor for LinearLR.')
    parser.add_argument('--end_factor', type=float, default=1.0, help='End factor for LinearLR.')
    parser.add_argument('--iterations', type=int, default=10, help='Iterations for LinearLR.')
    parser.add_argument('--t_max', type=int, default=10, help='T-max for CosineAnnealing.')
    parser.add_argument('--eta_min', type=float, default=0, help='Eta-min for CosineAnnealing.')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for StepLR.')
    parser.add_argument('--gamma_lr', type=float, default=0.5, help='Gamma for StepLR.')

    # Loss function parameters
    parser.add_argument_group(title="Loss Function Parameters", description="Settings for the loss function(s)")
    parser.add_argument('--loss_function', type=str, default='FocalTversky', help='Loss Function.')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha for FocalTversky and Tversky.')
    parser.add_argument('--beta', type=float, default=0.3, help='Beta for FocalTversky and Tversky.')
    parser.add_argument('--gamma', type=float, default=1.3333, help='Gamma for FocalTversky.')
    parser.add_argument('--topoloss_patch', type=int, default=64, help='Patch size for Topoloss.')
    parser.add_argument('--positive_weight', type=float, default=1.0, help='Weight for positive example in BCE.')
    parser.add_argument('--alpha_focal', type=float, default=0.8, help='Alpha for Focal.')
    parser.add_argument('--gamma_focal', type=float, default=2, help='Gamma for Focal.')
    
    parser.add_argument('--loss_function1', type=str, default='FocalTversky', help='Loss Function 1 for combo loss.')
    parser.add_argument('--loss_function1_weight', type=float, default=0.5, help='Weight of loss Function 1 for combo loss.')
    parser.add_argument('--loss_function2', type=str, default='FocalTversky', help='Loss Function 2 for combo loss.')
    parser.add_argument('--loss_function2_weight', type=float, default=0.5, help='Weight of loss Function 2 for combo loss.')
    
    # Directories
    parser.add_argument_group(title="Directory Settings", description="Paths to data and output directories")
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing the dataset.')
    parser.add_argument('--output_dir', type=str, default='../runs', help='Directory to save model checkpoints.')
    
    # training parameters
    parser.add_argument_group(title="Training Parameters", description="Settings for the training process")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    
    # Dataset parameters 
    parser.add_argument_group(title="Dataset Parameters", description="Settings related to data loading and preprocessing")
    parser.add_argument("--normalize", default=False, action="store_true", help="Activate normalization.")
    parser.add_argument("--negative", default=False, action="store_true", help="Images are inverted.")
    parser.add_argument("--augment", default=False, action="store_true", help="Online augmentation.")
    parser.add_argument('--transform', type=str, default='transforms.ToTensor', help='Transform to apply to the dataset.')
    
    # # To implement
    # parser.add_argument("--resume", default=False, action="store_true", help="Resume traing from last model saved")
    # parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for resuming training')

    return parser.parse_args()

def freeze_encoder(model):
    for child in model.encoder.children():
        for param in child.parameters():
            param.requires_grad = False
    return

def get_model(args, aux_params=None):

    weights = "imagenet" if bool(args.pretrained_weights) else None
    freeze = bool(args.freeze_backbone)
    attention = None if args.attention == "None" else args.attention
    batchnorm = args.batchnorm

    try:
        if args.model == 'U-Net' or args.model == 'Unet':
            model = torchseg.Unet(
                encoder_name=args.encoder,
                encoder_weights=weights,
                decoder_attention_type=attention,
                decoder_use_batchnorm=batchnorm, 
                aux_params=aux_params,
                )
            
        elif args.model == 'U-Net++':
            model = torchseg.UnetPlusPlus(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params,
                decoder_attention_type=attention,
                decoder_use_batchnorm=batchnorm
                )
            
        elif args.model == 'MAnet':
            model = torchseg.MAnet(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params,
                decoder_use_batchnorm=batchnorm
                )   
            
        elif args.model == 'LinkNet':
            model = torchseg.Linknet(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params,
                decoder_use_batchnorm=batchnorm
                )  
        
        elif args.model == 'FPN':
            model = torchseg.FPN(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params,
                decoder_use_batchnorm=batchnorm
                ) 
            
        elif args.model == 'PSPNet':
            model = torchseg.PSPNet(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params,
                decoder_use_batchnorm=batchnorm
                )    
            
        elif args.model == 'PAN':
            model = torchseg.PAN(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params,
                decoder_use_batchnorm=batchnorm
                )    
            
        elif args.model == 'DeepLabV3':
            model = torchseg.DeepLabV3(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params,
                decoder_use_batchnorm=batchnorm
                )  
            
        elif args.model == 'DeepLabV3+':
            model = torchseg.DeepLabV3Plus(
                encoder_name=args.encoder,
                encoder_weights=weights,
                aux_params=aux_params,
                decoder_use_batchnorm=batchnorm
                )          
        
        else:
            raise ValueError('Model type not recognized')

        if freeze:
            freeze_encoder(model)

        return model
    
    except Exception as e:
        print(e)
        raise ValueError

def get_optimizer(args, model):
    try:
        if args.optimizer == "Adam":
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
            
        elif args.optimizer == "AdamW":
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
        
        elif args.optimizer == "SGD":
            
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            
        else:
            raise ValueError('Optimizer type not recognized')
            
        return optimizer

    except Exception as e:
        print(e)
        raise ValueError

def get_warmup_scheduler(args, optimizer):
    
    if args.warmup_scheduler in [None,"None"]:
        return None
    
    elif args.warmup_scheduler == "Linear":
        
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                   start_factor=0.001,
                                                   end_factor=1.0,
                                                   total_iters=args.warmup_steps)
        
    else:
            raise ValueError('Warmup scheduler type not recognized')
        
    return warmup

def get_scheduler(args, optimizer, warmup=None):
    
    if args.scheduler in [None,"None"]:
        return None
    
    elif args.scheduler == "LinearLR":
        
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                      start_factor=args.start_factor, 
                                                      end_factor=args.end_factor, 
                                                      total_iters=args.iterations )
    
    elif args.scheduler == "CosineAnnealingLR":
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.t_max, 
                                                               eta_min=args.eta_min)
        
    elif args.scheduler == "StepLR":
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=args.step_size,
                                                    gamma=args.gamma_lr)
        
    else:
        raise ValueError('Scheduler type not recognized')
    
    if warmup is not None:
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                          schedulers=[warmup, scheduler],
                                                          milestones=[args.warmup_steps])
    
    
    return scheduler

def get_loss_function(args):
    """
    Retrieves the loss function based on the configuration in args.

    Args:
        args: The parsed command-line arguments or configuration object.

    Returns:
        A loss function object or a tuple of (loss function object, weight) 
        in the case of the "Combo" loss.
    """

    if args.loss_function == "FocalTversky":
        return FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    elif args.loss_function == "Tversky":
        return TverskyLoss(alpha=args.alpha, beta=args.beta)
    elif args.loss_function == "IoU":
        return IoULoss()
    elif args.loss_function == "Dice":
        return DiceLoss()
    elif args.loss_function == "LCDice":
        return LCDiceLoss()
    elif args.loss_function == "Topoloss":
        return TopologicalLoss()
    elif args.loss_function == "BCE":
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.positive_weight])).to("cuda" if torch.cuda.is_available() else "cpu")
    elif args.loss_function == "Focal":
        return FocalLoss(alpha=args.alpha_focal, gamma=args.gamma_focal)
    elif args.loss_function == "Combo":
        loss_func1 = get_loss_function_by_name(args.loss_function1, args)
        loss_func2 = get_loss_function_by_name(args.loss_function2, args)
        weight1 = float(args.loss_function1_weight)
        weight2 = float(args.loss_function2_weight)
        return [loss_func1, loss_func2, weight1, weight2]
    else:
        raise ValueError('Loss function type not recognized')

def get_loss_function_by_name(loss_func_name, args):
    """
    Retrieves a specific loss function by name using the provided arguments.
    """
    if loss_func_name == "FocalTversky":
        return FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    elif loss_func_name == "Tversky":
        return TverskyLoss(alpha=args.alpha, beta=args.beta)
    elif loss_func_name == "IoU":
        return IoULoss()
    elif loss_func_name == "Dice":
        return DiceLoss()
    elif loss_func_name == "LCDice":
        return LCDiceLoss()
    elif loss_func_name == "Topoloss":
        return TopologicalLoss()
    elif loss_func_name == "BCE":
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.positive_weight]))
    elif loss_func_name == "Focal":
        return FocalLoss(alpha=args.alpha_focal, gamma=args.gamma_focal)
    else:
        raise ValueError(f"Invalid loss function name for Combo option: {loss_func_name}")

def get_args_test():
    
    parser = argparse.ArgumentParser(description='Test a U-Net model')
    # Model parameters 
    parser.add_argument('--model', type=str, default='Unet', help='Model to train')
    parser.add_argument('--attention', type=str, default='None', help='Attention type')
    parser.add_argument('--batchnorm', type=str, default='True', help='Batchnorm')
    parser.add_argument('--test_model_path', type=str, help='Path to the model to test')
    
    # Encoder parameters
    parser.add_argument('--encoder', type=str, default='resnet152', help='Model to train.')
    parser.add_argument('--pretrained_weights', default=None, action="store_true", help="Utilize pretrained weights.")
    parser.add_argument('--freeze_backbone', default=None, action="store_true", help="Freezes encoder weights.")
    
    # Loss function parameters
    parser.add_argument('--loss_function', type=str, default='FocalTversky', help='Loss Function.')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha for FocalTversky and Tversky.')
    parser.add_argument('--beta', type=float, default=0.3, help='Beta for FocalTversky and Tversky.')
    parser.add_argument('--gamma', type=float, default=1.3333, help='Gamma for FocalTversky.')
    parser.add_argument('--topoloss_patch', type=int, default=64, help='Patch size for Topoloss.')
    parser.add_argument('--positive_weight', type=float, default=1.0, help='Weight for positive example in BCE.')
    
    parser.add_argument('--loss_function1', type=str, default='FocalTversky', help='Loss Function 1 for combo loss.')
    parser.add_argument('--loss_function1_weight', type=float, default=0.5, help='Weight of loss Function 1 for combo loss.')
    parser.add_argument('--loss_function2', type=str, default='FocalTversky', help='Loss Function 2 for combo loss.')
    parser.add_argument('--loss_function2_weight', type=float, default=0.5, help='Weight of loss Function 2 for combo loss.')
    
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument("--normalize", default=False, action="store_true", help="Activate normalization")
    parser.add_argument("--negative", default=False, action="store_true", help="Invert images")
    parser.add_argument('--transform', type=str, default='transforms.ToTensor', help='Transform to apply to the dataset.')
    
    return parser.parse_args()

def parse_transforms(transform_strings_str):
    """
    Parses a string representing a list of transform strings into a 
    torchvision.transforms.Compose object.

    Args:
        transform_strings_str: A string representing a list of transform strings.
                               e.g., "['transforms.Resize((512, 512))', 'transforms.ToTensor()']"

    Returns:
        A transforms.Compose object or None if an error occurred.
    """
    try:
        # Safely evaluate the input string as a Python list
        transform_strings = ast.literal_eval(transform_strings_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error: Input string is not a valid list: {e}")
        return None

    if not isinstance(transform_strings, list):
        print("Error: Input is not a list of strings.")
        return None

    transform_list = []
    for transform_str in transform_strings:
        try:
            transform = parse_single_transform(transform_str)
            transform_list.append(transform)
        except ValueError as e:
            print(f"Error processing transform '{transform_str}': {e}")
            return None

    return transforms.Compose(transform_list)

def parse_single_transform(transform_str):
    """
    Parses a single transform string and returns the corresponding transform object.

    Args:
        transform_str: A string representing a single transform.
                       e.g., "transforms.Resize((512, 512))"

    Returns:
        A transform object.
    """
    # Match transforms with or without arguments
    match = re.match(r"transforms\.(\w+)(?:\((.*)\))?", transform_str)
    if not match:
        raise ValueError(f"Invalid transform format: {transform_str}")

    transform_name, transform_args_str = match.groups()
    transform_args = {}

    if transform_args_str:
        # Parse arguments using eval within a safe context
        try:
            # Create a safe dictionary for evaluation
            safe_dict = {'__builtins__': None}  # Restrict built-in functions
            # Allow specific functions if needed, like 'tuple'
            safe_dict['tuple'] = tuple

            # Check if the argument string represents a tuple
            if transform_args_str.startswith('(') and transform_args_str.endswith(')'):
                # Evaluate the entire argument string as a tuple
                try:
                    args_tuple = ast.literal_eval(transform_args_str)
                    if isinstance(args_tuple, tuple):
                        transform_args = {'size': args_tuple}  # Resize expects a 'size' argument
                    else:
                        raise ValueError("Argument is not a tuple")
                except (SyntaxError, ValueError):
                    raise ValueError(f"Invalid transform arguments (tuple parsing failed): {transform_args_str}")
            else:
                # Parse arguments as key-value pairs or positional arguments
                for arg_str in transform_args_str.split(','):
                    arg_str = arg_str.strip()
                    if '=' in arg_str:
                        key, value = arg_str.split('=', 1)
                        # Evaluate the value in a safe context
                        transform_args[key.strip()] = eval(value.strip(), safe_dict)
                    else:
                        # Evaluate the value in a safe context
                        transform_args[len(transform_args)] = eval(arg_str, safe_dict)
        except (SyntaxError, NameError, ValueError) as e:
            raise ValueError(f"Invalid transform arguments: {transform_args_str} - Error: {e}")

    try:
        transform = getattr(transforms, transform_name)(**transform_args)
    except AttributeError:
        raise ValueError(f"Invalid transform name: {transform_name}")
    except TypeError as e:
        raise ValueError(f"Error creating transform {transform_name}: {e}")

    return transform