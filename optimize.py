import optuna
import torch.nn as nn
from torch import optim
import torchseg
from utils.hyperparameter_optimizer import HyperparameterOptimizer 

def main():
    # 1. Define the Hyperparameter Space
    hyperparameter_space = {
        "model_params": {
            "decoder_attention_type": [None], #scse
            "decoder_use_batchnorm": [True],
            "encoder_name": ["resnet50"],
            "encoder_weights": [None],
        },
        "optimizer_params": {
            "optimizer": ["Adam"],
            "lr": {"low": 5e-5, "high": 3e-3, "log": True},
            "momentum": {"low": 0.6, "high": 0.99, "log": False},
            "weight_decay": {"low": 5e-5, "high": 3e-3, "log": True},
        },
        "loss_params": {
            "loss_function": ["BCE"],
            #"loss_function1": ["BCE"],
            #"loss_function2": ["Topoloss"],
            #"loss_function1_weight": [1],
            #"loss_function2_weight": {"low": 1e-5, "high": 1e-1, "log": False},
            #"alpha": {"low": 0.1, "high": .99, "log": False},
            #"beta": {"low": 0.1, "high": 0.99, "log": False},
            #"gamma": {"low": 0.1, "high": 2, "log": False},
            #"topoloss_patch": [32, 64, 128],
            "positive_weight": {"low": 0.1, "high": 10, "log": False},
            # "alpha_focal": [0.8],
            # "gamma_focal": [0.2],
        },
        "warmup_params": {
            "warmup_scheduler": [None],
            "warmup_steps": [1], # [1, 2, 3, 4, 5],
        },
        "scheduler_params": {
            "scheduler": [None, "CosineAnnealingLR"],
            # "start_factor": [1.0],
            # "end_factor": [0.3],
            #"iterations": [10],
            "t_max": [5, 10],
            "eta_min": [0, 0.1, 0.2, 0.3],
            #"step_size": [5, 10, 15],
            #"gamma_lr": {"low": 0.6, "high": 0.8, "log": False},
        },
        "other_params": {
            "batch_size": [64],
            "epochs": [30],
            "normalize": [False, True],
            "negative": [True], 
            "transform": [
                "['transforms.Resize((128, 128))','transforms.ToTensor()']"
            ],
        },
    }

    # 2. Define the Data Directory and Output Directory
    data_dir = "C:\\Users\\lorenzo.francesia\\Documents\\github\\data_plus" 
    output_dir = "C:\\Users\\lorenzo.francesia\\Documents\\github\\runs" 

    # 3. Get the Model Class
    model_class = torchseg.Unet

    # 4. Create the Hyperparameter Optimizer
    optimizer = HyperparameterOptimizer(
        data_dir=data_dir,
        model_class=model_class,
        hyperparameter_space=hyperparameter_space,
        study_name="unet_wasserstein_bce_plusdata_adam_30epochs",  # Choose a study name
        output_dir=output_dir,
        objective_names=("wasserstein_similarity",),
        objective_directions=("maximize",)
    )

    # 5. Run the Optimization
    optimizer.optimize(n_trials=300) 


if __name__ == "__main__":
    main()
    
    #optuna.delete_study(study_name="unet_wasserstein_bce_plusdata_adam", storage="sqlite:///C://Users//lorenzo.francesia//Documents//github//runs//optimization_studies.db")
