import optuna
import torch.nn as nn
from torch import optim
import torchseg
from utils.hyperparameter_optimizer import HyperparameterOptimizer 

def main():
    # 1. Define the Hyperparameter Space
    hyperparameter_space = {
        "model_params": {
            "decoder_attention_type": [None, "scse"],
            "decoder_use_batchnorm": [True, False],
            "encoder_name": ["resnet34", "resnet50", "resnet101", "resnet200"],
            "encoder_weights": [None],
        },
        "optimizer_params": {
            "optimizer": ["AdamW", "Adam"],
            "lr": {"low": 1e-4, "high": 1e-3, "log": True},
            "momentum": {"low": 0.9, "high": 0.99, "log": False},
            "weight_decay": {"low": 1e-5, "high": 1e-3, "log": False}
        },
        "loss_params": {
            "loss_function": ["BCE"],
            "loss_function1": ["Tversky"],
            "loss_function2": ["Tversky"],
            "loss_function1_weight": [0.5],
            "loss_function2_weight": [0.5],
            "alpha": [0.7],
            "beta": [0.3],
            "gamma": [1.3333],
            "topoloss_patch": [64],
            "positive_weight": {"low": 0.1, "high": 99, "log": False},
            "alpha_focal": [0.8], #{"low": 0.01, "high": 0.99, "log": False},
            "gamma_focal": [0.2], # {"low": 0.01, "high": 0.99, "log": False}
        },
        "warmup_params": {
            "warmup_scheduler": ["None", "Linear"],
            "warmup_steps": [1, 2, 3],
        },
        "scheduler_params": {
            "scheduler": ["None", "StepLR"],
            "start_factor": [1.0],
            "end_factor": [0.3],
            "iterations": [10],
            "t_max": [10],
            "eta_min": [0],
            "step_size": {"low": 5, "high": 20, "log": False},
            "gamma_lr": {"low": 0.05, "high": 0.95, "log": False},
        },
        "other_params": {
            "batch_size": [24, 48, 72],
            "epochs": [20],
            "normalize": [False, True],
            "negative": [True], 
            "transform": [
                "['transforms.Resize((256,256))','transforms.ToTensor()']"
            ],
        },
    }

    # 2. Define the Data Directory and Output Directory
    data_dir = "C:\\Users\\lorenzo.francesia\\Documents\\github\\data"  # Replace with your data directory
    output_dir = "C:\\Users\\lorenzo.francesia\\Documents\\github\\runs"  # Replace with your desired output directory

    # 3. Get the Model Class
    model_class = torchseg.Unet

    # 4. Create the Hyperparameter Optimizer
    optimizer = HyperparameterOptimizer(
        data_dir=data_dir,
        model_class=model_class,
        hyperparameter_space=hyperparameter_space,
        study_name="Focal_opt_1",  # Choose a study name
        output_dir=output_dir,
    )

    # 5. Run the Optimization
    optimizer.optimize(n_trials=100)  # Adjust the number of trials as needed


if __name__ == "__main__":
    main()
    
