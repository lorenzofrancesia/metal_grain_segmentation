import optuna
import torch.nn as nn
from torch import optim
import segmentation_models_pytorch as smp
from utils.hyperparameter_optimizer import HyperparameterOptimizer 

def main():
    # 1. Define the Hyperparameter Space
    hyperparameter_space = {
        "model_params": {
            "attention": [None, "scse"],
            "batchnorm": [False, True],
            "encoder_name": ["resnet18", "resnet50", "resnet100", "resnet200", "lambda_resnet50ts",
                             'regnetv_064', 'regnetx_064', "resnest200e", "efficientnet_b8", "effcientnetv2_xl" ],
            "encoder_weights": [None, "imagenet"],
            "in_channels": [3],
            "classes": [1],
        },
        "optimizer_params": {
            "optimizer": ["AdamW", "Adam", "SGD"],
            "lr": {"low": 1e-5, "high": 1e-1, "log": True},
            "momentum": {"low": 0.2, "high": 0.99, "log": False},
            "weight_decay": {"low": 1e-5, "high": 1e-2, "log": False}
        },
        "loss_params": {
            "loss_function": ["Focal"],
            "loss_function1": ["Tversky"],
            "loss_function2": ["Tversky"],
            "loss_function1_weight": [0.5],
            "loss_function2_weight": [0.5],
            "alpha": [0.7],
            "beta": [0.3],
            "gamma": [1.3333],
            "topoloss_patch": [64],
            "positive_weight": [1],
            "alpha_focal": {"low": 0.01, "high": 0.99, "log": False},
            "gamma_focal": {"low": 0.01, "high": 0.99, "log": False}
        },
        "warmup_params": {
            "warmup_scheduler": ["None", "Linear"],
            "warmup_steps": {"low": 1, "high": 10, "log": False}
        },
        "scheduler_params": {
            "scheduler": ["StepLR"],
            "start_factor": [1.0],
            "end_factor": [0.3],
            "iterations": [10],
            "t_max": [10],
            "eta_min": [0],
            "step_size": {"low": 5, "high": 20, "log": False},
            "gamma_lr": {"low": 0.05, "high": 0.95, "log": False},
        },
        "other_params": {
            "batch_size": [6, 12, 14, 48],
            "epochs": [20],
            "normalize": [True, False],
            "negative": [True], 
            "transform": [
                "['transforms.Resize((512,512))','transforms.ToTensor()']"
            ],
        },
    }

    # 2. Define the Data Directory and Output Directory
    data_dir = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\github\\data"  # Replace with your data directory
    output_dir = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\github\\runs"  # Replace with your desired output directory

    # 3. Get the Model Class
    model_class = smp.Unet

    # 4. Create the Hyperparameter Optimizer
    optimizer = HyperparameterOptimizer(
        data_dir=data_dir,
        model_class=model_class,
        hyperparameter_space=hyperparameter_space,
        study_name="test_focal",  # Choose a study name
        output_dir=output_dir,
    )

    # 5. Run the Optimization
    optimizer.optimize(n_trials=10)  # Adjust the number of trials as needed


if __name__ == "__main__":
    main()
    