import os
import torch
import segmentation_models_pytorch as smp
from utils.hyperparameter_optimizer import HyperparameterOptimizer
from PIL import Image
import numpy as np
import torchvision.utils as vutils

def main():
    # Use a highly reduced hyperparameter space for quick testing
    hyperparameter_space = {
        "model_params": {
            "attention": [None],               # Reduced options
            "encoder_name": ["resnet18"],      # Only smallest model
            "encoder_weights": [None],         # Skip pretrained weights
            "in_channels": [3],
            "classes": [1],
        },
        "optimizer_params": {
            "optimizer": ["Adam"],             # Just one optimizer
            "lr": {"low": 1e-4, "high": 1e-2, "log": True},
            "momentum": [0.9],                 # Fixed value
            "weight_decay": [1e-4],            # Fixed value
        },
        "loss_params": {
            "loss_function": ["BCE"],          # Simplest loss
            "loss_function1": ["Tversky"],
            "loss_function2": ["Tversky"],
            "loss_function1_weight": [0.5],
            "loss_function2_weight": [0.5],
            "alpha": [0.7],
            "beta": [0.3],
            "gamma": [1.3333],
            "positive_weight": [1],
        },
        "warmup_params": {
            "warmup_scheduler": ["None"],      # No warmup
            "warmup_steps": [0],               # Fixed value
        },
        "scheduler_params": {
            "scheduler": ["None"],             # No scheduler
            "start_factor": [1.0],
            "end_factor": [0.3],
            "iterations": [10],
            "t_max": [10],
            "eta_min": [0],
        },
        "other_params": {
            "batch_size": [2],                 # Very small batch size
            "epochs": [2],                     # Just 2 epochs
            "normalize": [False],
            "negative": [True],
            "transform": [
                "['transforms.Resize((64,64))','transforms.ToTensor()']"  # Small images
            ],
        },
    }
    
    # Set up directories for testing
    # Use temporary or local directories
    import tempfile
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    data_dir = r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Documents\Project\data"
    output_dir = os.path.join(temp_dir, "test_output")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy data files for testing
    # This assumes your Trainer can handle the absence of real data or provides appropriate error messages
    
    # Get the model class
    model_class = smp.Unet
    
    # Force CPU usage for testing
    device = 'cpu'
    
    # Create the Hyperparameter Optimizer
    optimizer = HyperparameterOptimizer(
        data_dir=data_dir,
        model_class=model_class,
        hyperparameter_space=hyperparameter_space,
        study_name="test_improvements",
        output_dir=output_dir,
        device=device,
    )
    
    # Run a very small optimization
    print("Starting optimization test...")
    try:
        optimizer.optimize(n_trials=2)  # Just 2 trials for quick testing
        
        # Check if output files were created
        check_output_files(output_dir)
        
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error during optimization: {e}")
    
    print(f"Results saved to: {output_dir}")

def check_output_files(output_dir):
    """Check if the expected output files were created."""
    expected_files = [
        "best_hyperparameters.yml", 
        "study_summary.yml",
    ]
    
    # Check for main files
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ Found output file: {filename}")
        else:
            print(f"✗ Missing expected file: {filename}")
    
    # Check for trial directories
    trial_dirs = [d for d in os.listdir(output_dir) if d.startswith("trial_")]
    print(f"Found {len(trial_dirs)} trial directories")
    
    if trial_dirs:
        # Check contents of the first trial directory
        trial_dir = os.path.join(output_dir, trial_dirs[0])
        trial_files = os.listdir(trial_dir)
        print(f"Files in {trial_dirs[0]}: {', '.join(trial_files)}")
    
    # Check for visualization files
    vis_files = [f for f in os.listdir(output_dir) if f.endswith(".png") or f.endswith(".csv")]
    print(f"Found {len(vis_files)} visualization/data files: {', '.join(vis_files)}")

if __name__ == "__main__":
    main()