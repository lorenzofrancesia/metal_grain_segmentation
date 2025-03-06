import optuna
import torch.nn as nn
from torch import optim
import segmentation_models_pytorch as smp
from utils.hyperparameter_optimizer import HyperparameterOptimizer 

def main():
    # 1. Define the Hyperparameter Space
    hyperparameter_space = {
        "model_params": {
            "attention": [None],
            "batchnorm": [False],
            "encoder_name": ["resnet18"],
            "encoder_weights": ["imagenet"],
            "in_channels": [3],
            "classes": [1],
        },
        "optimizer_params": {
            "optimizer": ["AdamW"],
            "lr": {"low": 1e-5, "high": 1e-1, "log": True},
            "momentum": {"low": 0.2, "high": 0.99, "log": False},
            "weight_decay": [1e-2]
        },
        "loss_params": {
            "loss_function": ["Tversky"],
            "loss_function1": ["Tversky"],
            "loss_function2": ["Tversky"],
            "loss_function1_weight": [0.5],
            "loss_function2_weight": [0.5],
            "alpha": [0.7],
            "beta": [0.3],
            "gamma": [1.3333],
            "topoloss_patch": [64],
            "positive_weight": [1],
            "alpha_focal": [0.2],
            "gamma_focal": [0.8]
        },
        "warmup_params": {
            "warmup_scheduler": ["None"],
            "warmup_steps": [0]
        },
        "scheduler_params": {
            "scheduler": ["LinearLR"],
            "start_factor": [1.0],
            "end_factor": [0.3],
            "iterations": [10],
            "t_max": [10],
            "eta_min": [0],
            "step_size": [5],
            "gamma_lr": [0.5],
        },
        "other_params": {
            "batch_size": [12],
            "epochs": [3],
            "normalize": [True],
            "negative": [True], 
            "transform": [
                "['transforms.Resize((512,512))','transforms.ToTensor()']"
            ],
        },
    }

    # 2. Define the Data Directory and Output Directory
    data_dir = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\data"  # Replace with your data directory
    output_dir = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\optim"  # Replace with your desired output directory

    # 3. Get the Model Class
    model_class = smp.Unet

    # 4. Create the Hyperparameter Optimizer
    optimizer = HyperparameterOptimizer(
        data_dir=data_dir,
        model_class=model_class,
        hyperparameter_space=hyperparameter_space,
        study_name="test4",  # Choose a study name
        output_dir=output_dir,
    )

    # 5. Run the Optimization
    optimizer.optimize(n_trials=10)  # Adjust the number of trials as needed


if __name__ == "__main__":
    main()
    