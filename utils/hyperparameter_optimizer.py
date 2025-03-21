import optuna
import os
import torch
import torch.nn as nn
from torch import optim
import ast 
import re  
import yaml
import numpy as np
import pandas as pd

from utils.trainer import Trainer  # Correct relative import
from loss.tversky import TverskyLoss, FocalTverskyLoss  # Correct relative import
from loss.iou import IoULoss
from loss.dice import DiceLoss
from loss.topoloss import TopologicalLoss
from loss.focal import FocalLoss
import torchvision.transforms as transforms  # Import transforms

class HyperparameterOptimizer:
    def __init__(self,
                 data_dir,
                 model_class,
                 hyperparameter_space,
                 study_name="segmentation_study",
                 storage=None,
                 output_dir="../optimization_runs",
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.data_dir = data_dir
        self.model_class = model_class
        self.hyperparameter_space = hyperparameter_space
        self.study_name = study_name
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        # Database
        if storage is None:
            db_name = "optimization_studies.db"  # Use a common database file
            self.storage = f"sqlite:///{os.path.join(self.output_dir, db_name)}"
        else:
            self.storage = storage

        self.study = optuna.create_study(study_name=self.study_name, storage=self.storage, direction="maximize",
                                         load_if_exists=True)

    def _objective(self, trial):
        # Get hyperparameter suggestions from Optuna
        model_params = self.get_model_params(trial)
        optimizer_params = self.get_optimizer_params(trial)
        loss_params = self.get_loss_params(trial)
        warmup_params = self.get_warmup_params(trial)  # Get warmup parameters
        scheduler_params = self.get_scheduler_params(trial)
        other_params = self.get_other_params(trial)

        # Get model
        model = self.model_class(**model_params)

        # Create optimizer
        optimizer = self.get_optimizer(optimizer_params, model)

        # Handle loss function selection (single or combo)
        loss_function = self.get_loss_function(loss_params)

        # Create scheduler (with optional warmup)
        lr_scheduler = self.get_scheduler(scheduler_params, optimizer, warmup_params)  # Pass warmup_params


        # Create a Trainer instance with the suggested hyperparameters
        trainer = Trainer(
            data_dir=self.data_dir,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            lr_scheduler=lr_scheduler,
            device=self.device,
            output_dir=self.output_dir,
            config=None,  # Pass config=None
            batch_size=other_params["batch_size"],  # Pass batch_size
            epochs=other_params["epochs"],  # Pass epochs
            warmup=warmup_params["warmup_steps"] if warmup_params["warmup_scheduler"] != "None" else 0, 
            train_transform=self.parse_transforms(other_params["transform"]),
            normalize=other_params["normalize"],
            negative=other_params["negative"],
            save_output=False
        )

        # Train and evaluate the model
        trainer.train()

        # Get the validation loss
        val_loss = trainer.last_loss
        dice = trainer.last_dice
        
        # Store all relevant metrics for this trial
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("optimizer_params", optimizer_params)
        trial.set_user_attr("loss_params", loss_params)
        trial.set_user_attr("warmup_params", warmup_params)
        trial.set_user_attr("scheduler_params", scheduler_params)
        trial.set_user_attr("other_params", other_params)
        
        # Store training metrics if available
        if hasattr(trainer, "train_losses"):
            trial.set_user_attr("train_losses", trainer.train_losses)
        if hasattr(trainer, "val_losses"):
            trial.set_user_attr("val_losses", trainer.val_losses)
        
        # Log additional metrics if available from trainer
        if hasattr(trainer, "metrics"):
            for metric_name, metric_value in trainer.metrics.items():
                trial.set_user_attr(f"metric_{metric_name}", metric_value)
        
        if not np.isfinite(val_loss):
            print(f"[WARNING] Trial {trial.number} resulted in infinite or NaN loss. Skipping.")
            del trainer, model, optimizer, lr_scheduler, loss_function  # Clean up before pruning
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        del trainer, model, optimizer, lr_scheduler, loss_function
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._save_trial_results(trial)
        return dice  


    def optimize(self, n_trials=100):
        """Run the optimization process with enhanced logging."""
        self.study.optimize(self._objective, n_trials=n_trials)
        
        # Print additional information if all trials were pruned.
        if len(self.study.trials) == 0:
            print("All trials were pruned due to infinite loss. Check your setup.")
        elif all(trial.state == optuna.trial.TrialState.PRUNED for trial in self.study.trials):
            print("All trials were pruned. Consider adjusting your hyperparameter search space or checking for issues.")
        else:
            print(f"Optimization complete.")
            print(f"Best trial: {self.study.best_trial.number}")
            print(f"Best hyperparameters: {self.study.best_params}")
            print(f"Best loss: {self.study.best_value}")
            
            # Print parameter importance if available
            try:
                importance = optuna.importance.get_param_importances(self.study)
                print("\nParameter importance:")
                for param, score in importance.items():
                    print(f"  {param}: {score:.4f}")
            except:
                pass

        self._save_results()
        
        return self.study.best_trial

    def _save_trial_results(self, trial):
        """Save detailed information about each completed trial in a folder named after the study."""
        # Get the study name from the study object
        study_name = self.study.study_name
        
        # Create a directory for the study if it doesn't exist
        study_dir = os.path.join(self.output_dir, study_name)
        os.makedirs(study_dir, exist_ok=True)
        
        # Create a directory for the trial inside the study directory
        trial_dir = os.path.join(study_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Save trial parameters and results
        trial_data = {
            "number": trial.number,
            "params": trial.params,
        }
        
        # Add user attributes (metrics, parameter groups, etc.)
        for key, value in trial.user_attrs.items():
            if isinstance(value, (list, dict, np.ndarray)):
                # For complex data types like training history
                if key == "train_losses" or key == "val_losses":
                    # Save training curves separately as CSV for easy plotting
                    if isinstance(value, list):
                        df = pd.DataFrame({key: value})
                        df.to_csv(os.path.join(trial_dir, f"{key}.csv"), index_label="epoch")
                        trial_data[key] = f"Saved to {key}.csv"
                    else:
                        trial_data[key] = value
            else:
                trial_data[key] = value
            
        # Save as YAML
        with open(os.path.join(trial_dir, "trial_details.yml"), "w") as f:
            yaml.dump(trial_data, f, default_flow_style=False)
            
            
            
            
    def _save_results(self):
        """Enhanced version to save comprehensive results of the optimization."""
        # Get the study name from the study object
        study_name = self.study.study_name
        
        # Create a directory for the study if it doesn't exist
        study_dir = os.path.join(self.output_dir, study_name)
        os.makedirs(study_dir, exist_ok=True)
        
        # Save best parameters as before
        best_params = self.study.best_params
        with open(os.path.join(study_dir, "best_hyperparameters.yml"), "w") as outfile:
            yaml.dump(best_params, outfile, default_flow_style=False)
        
        # Save overall study summary
        study_info = {
            "study_name": self.study.study_name,
            "direction": self.study.direction,
            "best_trial": self.study.best_trial.number,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "n_completed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }
        
        with open(os.path.join(study_dir, "study_summary.yml"), "w") as outfile:
            yaml.dump(study_info, outfile, default_flow_style=False)
        
        # Generate better visualizations
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_param_importances, plot_optimization_history, plot_slice

            # Create a directory specifically for visualizations within the study directory
            viz_dir = os.path.join(study_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Parameter importance visualization with improved layout
            fig = plot_param_importances(self.study)
            fig.update_layout(
                width=900, 
                height=600,
                margin=dict(l=20, r=20, t=30, b=20),
                title_text="Parameter Importance",
                title_x=0.5,
                font=dict(family="Arial, sans-serif", size=14)  # Fixed font specification
            )
            fig.write_image(os.path.join(viz_dir, "parameter_importance.png"))
            fig.write_html(os.path.join(viz_dir, "parameter_importance.html"))
            
            # Optimization history plot
            history_fig = plot_optimization_history(self.study)
            history_fig.update_layout(
                width=900, 
                height=500,
                margin=dict(l=20, r=20, t=30, b=20),
                title_text="Optimization History",
                title_x=0.5,
                font=dict(family="Arial, sans-serif", size=14)  # Fixed font specification
            )
            history_fig.write_image(os.path.join(viz_dir, "optimization_history.png"))
            history_fig.write_html(os.path.join(viz_dir, "optimization_history.html"))
            
            # Create a CSV with all trials information for easy analysis
            trials_df = self.study.trials_dataframe()
            trials_df.to_csv(os.path.join(study_dir, "all_trials.csv"))
            
            # Generate slice plots for the most important parameters
            # Get parameter importance
            importance = optuna.importance.get_param_importances(self.study)
            # Get top 5 most important parameters or all if less than 5
            top_params = list(importance.keys())[:min(5, len(importance))]
            
            for param in top_params:
                try:
                    slice_fig = plot_slice(self.study, params=[param])
                    slice_fig.update_layout(
                        width=800, 
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                        title_text=f"Slice Plot for {param}",
                        title_x=0.5,
                        font=dict(family="Arial, sans-serif", size=14)  # Fixed font specification
                    )
                    slice_fig.write_image(os.path.join(viz_dir, f"slice_{param}.png"))
                    slice_fig.write_html(os.path.join(viz_dir, f"slice_{param}.html"))
                except Exception as e:
                    print(f"Error generating slice plot for {param}: {e}")
            
            # Generate pairwise correlation plots for top parameters
            from optuna.visualization import plot_contour, plot_parallel_coordinate
            
            # Create contour plots only for the top parameters to avoid too many plots
            for i, param1 in enumerate(top_params):
                for param2 in top_params[i+1:]:  # Avoid duplicates
                    try:
                        contour_fig = plot_contour(self.study, params=[param1, param2])
                        contour_fig.update_layout(
                            width=800, 
                            height=600,
                            margin=dict(l=20, r=20, t=40, b=20),
                            title_text=f"Contour Plot: {param1} vs {param2}",
                            title_x=0.5,
                            font=dict(family="Arial, sans-serif", size=14)  # Fixed font specification
                        )
                        contour_fig.write_image(os.path.join(viz_dir, f"contour_{param1}_vs_{param2}.png"))
                        contour_fig.write_html(os.path.join(viz_dir, f"contour_{param1}_vs_{param2}.html"))
                    except Exception as e:
                        print(f"Error generating contour plot for {param1} vs {param2}: {e}")
            
            # Parallel coordinate plot for top parameters
            try:
                parallel_fig = plot_parallel_coordinate(self.study, params=top_params)
                parallel_fig.update_layout(
                    width=1000, 
                    height=600,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title_text="Parallel Coordinate Plot for Top Parameters",
                    title_x=0.5,
                    font=dict(family="Arial, sans-serif", size=14)  # Fixed font specification
                )
                parallel_fig.write_image(os.path.join(viz_dir, "parallel_coordinate.png"))
                parallel_fig.write_html(os.path.join(viz_dir, "parallel_coordinate.html"))
            except Exception as e:
                print(f"Error generating parallel coordinate plot: {e}")
                
            # Create a summary visualization HTML page that links to all plots
            self._create_visualization_index(viz_dir, top_params)
                
        except ImportError as e:
            print(f"Could not generate visualization plots. Make sure matplotlib and plotly are installed. Error: {e}")
        except Exception as e:
            print(f"Error generating visualization: {e}")
            
    def _create_visualization_index(self, viz_dir, top_params):
        """Create an HTML index page for all visualizations."""
        html_content = """<!DOCTYPE html>
    <html>
    <head>
        <title>Optuna Optimization Results - {study_name}</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
            }}
            h1, h2 {{ 
                color: #333; 
            }}
            .section {{ 
                margin-bottom: 30px; 
            }}
            .plot-container {{ 
                display: flex; 
                flex-wrap: wrap; 
                gap: 20px; 
            }}
            .plot-item {{ 
                margin-bottom: 20px; 
            }}
            .plot-item img {{ 
                max-width: 100%; 
                height: auto; 
                border: 1px solid #ddd; 
            }}
            .plot-item h3 {{ 
                margin: 10px 0; 
            }}
            a {{ 
                color: #0066cc; 
                text-decoration: none; 
            }}
            a:hover {{ 
                text-decoration: underline; 
            }}
        </style>
    </head>
    <body>
        <h1>Optuna Hyperparameter Optimization Results - {study_name}</h1>
        
        <div class="section">
            <h2>Overview</h2>
            <p>Study Name: {study_name}</p>
            <p>Best Trial: #{best_trial} (Loss: {best_value:.6f})</p>
            <p>Total Trials: {n_trials} (Completed: {n_completed}, Pruned: {n_pruned})</p>
        </div>
        
        <div class="section">
            <h2>Key Visualizations</h2>
            <div class="plot-container">
                <div class="plot-item">
                    <h3>Parameter Importance</h3>
                    <a href="parameter_importance.html" target="_blank">
                        <img src="parameter_importance.png" alt="Parameter Importance">
                    </a>
                </div>
                <div class="plot-item">
                    <h3>Optimization History</h3>
                    <a href="optimization_history.html" target="_blank">
                        <img src="optimization_history.png" alt="Optimization History">
                    </a>
                </div>
                <div class="plot-item">
                    <h3>Parallel Coordinate Plot</h3>
                    <a href="parallel_coordinate.html" target="_blank">
                        <img src="parallel_coordinate.png" alt="Parallel Coordinate Plot">
                    </a>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Parameter Slice Plots</h2>
            <div class="plot-container">
                {slice_plots}
            </div>
        </div>
        
        <div class="section">
            <h2>Parameter Contour Plots</h2>
            <div class="plot-container">
                {contour_plots}
            </div>
        </div>
        
        <div class="section">
            <h2>Trial Data</h2>
            <p>Individual trial data is stored in the trial_X directories.</p>
            <p><a href="../all_trials.csv" target="_blank">Download all trials data (CSV)</a></p>
        </div>
    </body>
    </html>
    """
        
        # Generate slice plot HTML
        slice_plots_html = ""
        for param in top_params:
            plot_path = f"slice_{param}.png"
            if os.path.exists(os.path.join(viz_dir, plot_path)):
                slice_plots_html += f"""
                <div class="plot-item">
                    <h3>Slice Plot: {param}</h3>
                    <a href="slice_{param}.html" target="_blank">
                        <img src="{plot_path}" alt="Slice Plot for {param}">
                    </a>
                </div>
                """
        
        # Generate contour plot HTML
        contour_plots_html = ""
        for i, param1 in enumerate(top_params):
            for param2 in top_params[i+1:]:
                plot_path = f"contour_{param1}_vs_{param2}.png"
                if os.path.exists(os.path.join(viz_dir, plot_path)):
                    contour_plots_html += f"""
                    <div class="plot-item">
                        <h3>Contour Plot: {param1} vs {param2}</h3>
                        <a href="contour_{param1}_vs_{param2}.html" target="_blank">
                            <img src="{plot_path}" alt="Contour Plot for {param1} vs {param2}">
                        </a>
                    </div>
                    """
        
        # Fill in the template
        formatted_html = html_content.format(
            study_name=self.study.study_name,
            best_trial=self.study.best_trial.number,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            n_completed=len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            n_pruned=len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            slice_plots=slice_plots_html,
            contour_plots=contour_plots_html
        )
        
        # Write the HTML file
        with open(os.path.join(viz_dir, "index.html"), "w") as f:
            f.write(formatted_html)
                
    def get_model_params(self, trial):
        model_params = {}
        for param, values in self.hyperparameter_space.get("model_params", {}).items():
            if isinstance(values, list):
                model_params[param] = trial.suggest_categorical(f"model_{param}", values)
            elif isinstance(values, dict) and "low" in values and "high" in values:
                if values.get("log", False):
                    model_params[param] = trial.suggest_float(f"model_{param}", values["low"], values["high"], log=True)
                else:
                    model_params[param] = trial.suggest_float(f"model_{param}", values["low"], values["high"])
            else:
                model_params[param] = values  # Directly assign if it's a fixed value
        return model_params

    def get_optimizer_params(self, trial):
        optimizer_params = {}
        for param, values in self.hyperparameter_space.get("optimizer_params", {}).items():
            if param == "optimizer":
                optimizer_params[param] = trial.suggest_categorical(f"optimizer_{param}", values)
            elif isinstance(values, list):
                optimizer_params[param] = trial.suggest_categorical(f"optimizer_{param}", values)
            elif isinstance(values, dict) and "low" in values and "high" in values:
                if values.get("log", False):
                    optimizer_params[param] = trial.suggest_float(f"optimizer_{param}", values["low"], values["high"], log=True)
                else:
                    optimizer_params[param] = trial.suggest_float(f"optimizer_{param}", values["low"], values["high"])
            else:
                optimizer_params[param] = values

        return optimizer_params

    def get_scheduler_params(self, trial):
        scheduler_params = {}
        for param, values in self.hyperparameter_space.get("scheduler_params", {}).items():
            if param == "scheduler":
                scheduler_params[param] = trial.suggest_categorical(f"scheduler_{param}", values)
            elif isinstance(values, list):
                scheduler_params[param] = trial.suggest_categorical(f"scheduler_{param}", values)
            elif isinstance(values, dict) and "low" in values and "high" in values:
                if values.get("log", False):
                    scheduler_params[param] = trial.suggest_float(f"scheduler_{param}", values["low"], values["high"], log=True)
                else:
                    scheduler_params[param] = trial.suggest_float(f"scheduler_{param}", values["low"], values["high"])
            else:
                scheduler_params[param] = values
        return scheduler_params
    
    def get_warmup_params(self, trial):
        warmup_params = {}
        for param, values in self.hyperparameter_space.get("warmup_params", {}).items():
            if param == "warmup_scheduler":
                warmup_params[param] = trial.suggest_categorical(f"warmup_{param}", values)
            elif isinstance(values, list):
                warmup_params[param] = trial.suggest_categorical(f"warmup_{param}", values)
            elif isinstance(values, dict) and "low" in values and "high" in values:
                if values.get("log", False):
                    warmup_params[param] = trial.suggest_float(f"warmup_{param}", values["low"], values["high"], log=True)
                else:
                    warmup_params[param] = trial.suggest_float(f"warmup_{param}", values["low"], values["high"])
        return warmup_params

    def get_loss_params(self, trial):
        loss_params = {}
        for param, values in self.hyperparameter_space.get("loss_params", {}).items():
            if isinstance(values, list):
                loss_params[param] = trial.suggest_categorical(f"loss_{param}", values)
            elif isinstance(values, dict) and "low" in values and "high" in values:
                if values.get("log", False):
                    loss_params[param] = trial.suggest_float(f"loss_{param}", values["low"], values["high"], log=True)
                else:
                    loss_params[param] = trial.suggest_float(f"loss_{param}", values["low"], values["high"])
            else:
                loss_params[param] = values
        return loss_params

    def get_other_params(self, trial):
        other_params = {}
        for param, values in self.hyperparameter_space.get("other_params", {}).items():
            if isinstance(values, list):
                other_params[param] = trial.suggest_categorical(f"other_{param}", values)
            elif isinstance(values, dict) and "low" in values and "high" in values:
                if values.get("log", False):
                    other_params[param] = trial.suggest_int(f"other_{param}", values["low"], values["high"], log=True)
                else:
                    other_params[param] = trial.suggest_int(f"other_{param}", values["low"], values["high"])
            else:
                other_params[param] = values
        return other_params

    def get_optimizer(self, optimizer_params, model):
        optimizer_name = optimizer_params['optimizer']
        optimizer_args = {k: v for k, v in optimizer_params.items() if k != 'optimizer'}

        if optimizer_name == "Adam":
            betas = optimizer_args.pop('momentum', 0.9)
            if not isinstance(betas, tuple):
                betas = (betas, 0.999)
            optimizer_args['betas'] = betas
            optimizer = optim.Adam(model.parameters(), **optimizer_args)

        elif optimizer_name == "AdamW":
            betas = optimizer_args.pop('momentum', 0.9)
            if not isinstance(betas, tuple):
                betas = (betas, 0.999)
            optimizer_args['betas'] = betas
            optimizer = optim.AdamW(model.parameters(), **optimizer_args)

        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), **optimizer_args)
        else:
            raise ValueError("Optimizer not recognized")

        return optimizer

    def get_warmup_scheduler(self, warmup_params, optimizer):
        if warmup_params["warmup_scheduler"] in [None, "None"]:
            return None

        elif warmup_params["warmup_scheduler"] == "Linear":
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                       start_factor=0.001,
                                                       end_factor=1.0,
                                                       total_iters=warmup_params["warmup_steps"])
            return warmup

        else:
            raise ValueError('Warmup scheduler type not recognized')

    def get_loss_function(self, loss_params):
        loss_name = loss_params["loss_function"]

        if loss_name == "Combo":
            loss_function = [
                self.get_loss_function_by_name(loss_params["loss_function1"], loss_params),
                self.get_loss_function_by_name(loss_params["loss_function2"], loss_params),
                loss_params["loss_function1_weight"],
                loss_params["loss_function2_weight"],
            ]
        else:
            loss_function = self.get_loss_function_by_name(loss_name, loss_params)

        return loss_function

    def get_loss_function_by_name(self, loss_func_name, loss_params):
        if loss_func_name == "FocalTversky":
            return FocalTverskyLoss(alpha=loss_params["alpha"], beta=loss_params["beta"], gamma=loss_params["gamma"])
        elif loss_func_name == "Tversky":
            return TverskyLoss(alpha=loss_params["alpha"], beta=loss_params["beta"])
        elif loss_func_name == "Focal":
            return FocalLoss(alpha=loss_params["alpha_focal"], gamma=loss_params["gamma_focal"])
        elif loss_func_name == "IoU":
            return IoULoss()
        elif loss_func_name == "Dice":
            return DiceLoss()
        elif loss_func_name == "BCE":
            return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([loss_params["positive_weight"]])).to("cuda" if torch.cuda.is_available() else "cpu")
        elif loss_func_name == "Topoloss":
            return TopologicalLoss()
        elif loss_func_name == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        elif loss_func_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss function name: {loss_func_name}")

    def get_scheduler(self, scheduler_params, optimizer, warmup_params):
        warmup_scheduler = self.get_warmup_scheduler(warmup_params, optimizer)
        if scheduler_params["scheduler"] in [None, "None"]:
            return None

        elif scheduler_params["scheduler"] == "LinearLR":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                          start_factor=scheduler_params["start_factor"],
                                                          end_factor=scheduler_params["end_factor"],
                                                          total_iters=scheduler_params["iterations"])

        elif scheduler_params["scheduler"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=scheduler_params["t_max"],
                                                                   eta_min=scheduler_params["eta_min"])

        elif scheduler_params["scheduler"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=scheduler_params["factor"],
                                                                   patience=scheduler_params["patience"],
                                                                   min_lr=scheduler_params["min_lr"])

        elif scheduler_params["scheduler"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=scheduler_params["step_size"],
                                                        gamma=scheduler_params["gamma_lr"])

        else:
            raise ValueError('Scheduler type not recognized')

        if warmup_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                              schedulers=[warmup_scheduler, scheduler],
                                                              milestones=[warmup_params["warmup_steps"]])

        return scheduler
    
    def parse_transforms(self, transform_strings_str):  # <-- ADDED METHOD
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
                transform = self.parse_single_transform(transform_str)
                transform_list.append(transform)
            except ValueError as e:
                print(f"Error processing transform '{transform_str}': {e}")
                return None

        return transforms.Compose(transform_list)

    def parse_single_transform(self, transform_str):  # <-- ADDED METHOD
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
