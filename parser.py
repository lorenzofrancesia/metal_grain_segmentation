import argparse
import yaml
import os
# --- Model Parameters ---
parser = argparse.ArgumentParser(description='Train a U-Net model')
model_group = parser.add_argument_group(title="Model Parameters", description="Settings related to the model architecture")
model_group.add_argument('--model', type=str, default='Unet', help='Model to train')
model_group.add_argument("--dropout", type=float, default=0, help='Dropout probability.')
model_group.add_argument("--pooling",  type=str, default='max', help='Type of pooling used.')

# --- Encoder Parameters ---
encoder_group = parser.add_argument_group(title="Encoder Parameters", description="Settings for the encoder part of the model")
encoder_group.add_argument('--encoder', type=str, default='resnet152', help='Model to train.')
encoder_group.add_argument('--weights', default=None, action="store_true", help="Utilize pretrained weights.")

# --- Optimizer Parameters ---
optimizer_group = parser.add_argument_group(title="Optimizer Parameters", description="Settings for the optimizer")
optimizer_group.add_argument('--optimizer', type=str, default='Adam', help='Optimizer.')
optimizer_group.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer.')
optimizer_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer if supported.')
optimizer_group.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer if supported.')

# --- Warmup Parameters ---
warmup_group = parser.add_argument_group(title="Warmup Parameters", description="Settings for learning rate warmup")
warmup_group.add_argument('--warmup_scheduler', type=str, default=None, help='Warmup scheduler.')
warmup_group.add_argument('--warmup_steps', type=int, default=3, help='Warmup steps for LinearLR.')

# --- Scheduler Parameters ---
scheduler_group = parser.add_argument_group(title="Scheduler Parameters", description="Settings for learning rate scheduling")
scheduler_group.add_argument('--scheduler', type=str, default=None, help='Scheduler.')
scheduler_group.add_argument('--start_factor', type=float, default=0.03, help='Start factor for LinearLR.')
scheduler_group.add_argument('--end_factor', type=float, default=1.0, help='End factor for LinearLR.')
scheduler_group.add_argument('--iterations', type=int, default=10, help='Iterations for LinearLR.')
scheduler_group.add_argument('--t_max', type=int, default=10, help='T-max for CosineAnnealing.')
scheduler_group.add_argument('--eta_min', type=float, default=0, help='Eta-min for CosineAnnealing.')
scheduler_group.add_argument('--step_size', type=int, default=5, help='Step size for StepLR.')
scheduler_group.add_argument('--gamma_lr', type=float, default=0.5, help='Gamma for StepLR.')

# --- Loss Function Parameters ---
loss_group = parser.add_argument_group(title="Loss Function Parameters", description="Settings for the loss function(s)")
loss_group.add_argument('--loss_function', type=str, default='FocalTversky', help='Loss Function.')
loss_group.add_argument('--alpha', type=float, default=0.7, help='Alpha for FocalTversky and Tversky.')
loss_group.add_argument('--beta', type=float, default=0.3, help='Beta for FocalTversky and Tversky.')
loss_group.add_argument('--gamma', type=float, default=1.3333, help='Gamma for FocalTversky.')
loss_group.add_argument('--topoloss_patch', type=int, default=64, help='Patch size for Topoloss.')

# --- Combo Loss Parameters ---
combo_loss_group = parser.add_argument_group(title="Combo Loss Parameters", description="Settings for combining multiple loss functions")
combo_loss_group.add_argument('--loss_function1', type=str, default='FocalTversky', help='Loss Function 1 for combo loss.')
combo_loss_group.add_argument('--loss_function1_weight', type=float, default=0.5, help='Weight of loss Function 1 for combo loss.')
combo_loss_group.add_argument('--loss_function2', type=str, default='FocalTversky', help='Loss Function 2 for combo loss.')
combo_loss_group.add_argument('--loss_function2_weight', type=float, default=0.5, help='Weight of loss Function 2 for combo loss.')

# --- Directories ---
dir_group = parser.add_argument_group(title="Directory Settings", description="Paths to data and output directories")
dir_group.add_argument('--data_dir', type=str, default='../data', help='Directory containing the dataset.')
dir_group.add_argument('--output_dir', type=str, default='../runs', help='Directory to save model checkpoints.')

# --- Training Parameters ---
training_group = parser.add_argument_group(title="Training Parameters", description="Settings for the training process")
training_group.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
training_group.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')

# --- Dataset Parameters ---
dataset_group = parser.add_argument_group(title="Dataset Parameters", description="Settings related to data loading and preprocessing")
dataset_group.add_argument("--normalize", default=False, action="store_true", help="Activate normalization.")
dataset_group.add_argument('--transform', type=str, default='transforms.ToTensor', help='Transform to apply to the dataset.')

def create_nested_dict(args, groups):
    """
    Creates a nested dictionary based on argparse groups.

    Args:
      args: The Namespace object returned by parser.parse_args().
      groups: A list of argument groups created using parser.add_argument_group().

    Returns:
      A nested dictionary where keys are group titles (converted to
      lowercase and spaces replaced with underscores) and values are
      dictionaries of arguments within those groups.
    """
    nested_dict = {}
    for group in groups:
        group_key = group.title.lower().replace(" ", "_")
        group_dict = {}
        for action in group._group_actions:
            # Get the destination (variable name) of the argument
            dest = action.dest
            # Get the value from the Namespace object
            value = getattr(args, dest)
            group_dict[dest] = value
        nested_dict[group_key] = group_dict

    # Add arguments that aren't in any group to a top-level 'ungrouped' category.
    ungrouped_dict = {}
    all_group_args = set()
    for group in groups:
      for action in group._group_actions:
        all_group_args.add(action.dest)

    for dest in vars(args):
       if dest not in all_group_args:
         ungrouped_dict[dest] = getattr(args, dest)
    if ungrouped_dict:  # Only add if there are ungrouped arguments
      nested_dict['ungrouped'] = ungrouped_dict

    return nested_dict


if __name__ == '__main__':
    args = parser.parse_args()

    # Collect all argument groups.  This is important!
    all_groups = [model_group, encoder_group, optimizer_group, warmup_group,
                  scheduler_group, loss_group, combo_loss_group, dir_group,
                  training_group, dataset_group]

    config_dict = create_nested_dict(args, all_groups)
    
    # Use a relative path for better portability and demonstration
    output_file_path = os.path.join("C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop", "config.yml")

    with open(output_file_path, "w+") as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {output_file_path}")

    # --- Example of running the script ---
    # To test this, run the script with various arguments:
    # python your_script.py --model UNet --encoder resnet34 --lr 0.001 --epochs 20 --batch_size 4
    # Then inspect the generated config.yaml file.
    
    # --- Dummy arguments for testing ---
    dummy_args = [
        '--model', 'UNet++',
        '--encoder', 'resnet34',
        '--optimizer', 'AdamW',
        '--loss_function', 'DiceLoss',
        '--epochs', '20',
        '--batch_size', '4',
        '--data_dir', './my_data',
        '--output_dir', './my_runs'
    ]

    args = parser.parse_args(dummy_args)  # Parse dummy arguments.

    all_groups = [model_group, encoder_group, optimizer_group, warmup_group,
                  scheduler_group, loss_group, combo_loss_group, dir_group,
                  training_group, dataset_group]

    config_dict = create_nested_dict(args, all_groups)
    output_file_path = os.path.join("C:\\Users\lorenzo.francesia\\OneDrive - Swerim\\Desktop", "dummy.yml")
    
    with open(output_file_path, "w+") as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False, sort_keys=False)
        
    print(f"Configuration saved to {output_file_path}")