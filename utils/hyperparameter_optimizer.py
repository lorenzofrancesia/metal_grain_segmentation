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
import traceback
import sys
import tqdm
from collections import defaultdict
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
from torch import optim
import torchvision.transforms as transforms

from loss.tversky import TverskyLoss, FocalTverskyLoss  # Correct relative import
from loss.iou import IoULoss
from loss.dice import DiceLoss
from loss.topoloss import TopologicalLoss
from loss.focal import FocalLoss
from data.dataset import SegmentationDataset
from utils.metrics import BinaryMetrics

from optuna.visualization import plot_param_importances, plot_optimization_history, plot_slice, plot_pareto_front, plot_contour, plot_parallel_coordinate


class Trainer():
    
    def __init__(self, 
                 data_dir,
                 model,
                 batch_size=16,
                 normalize=False,
                 negative=False,
                 train_transform= transforms.ToTensor(),
                 optimizer=optim.Adam, 
                 loss_function=nn.BCELoss(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr_scheduler=None,
                 warmup=3,
                 epochs=10,
                 ):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.lr_scheduler = lr_scheduler
        
        # Dataset
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.batch_size = batch_size
        self.normalize = normalize
        self.negative = negative

        
        # Training parameters 
        self.epochs = epochs
        self.warmup_epochs = warmup if isinstance(self.lr_scheduler, optim.lr_scheduler.SequentialLR) else 0
        self.total_epochs = self.warmup_epochs + self.epochs
        self.current_epoch = 0
        
        self.best_loss = float("inf")
        self.best_dice = 0
        self.early_stopping_counter = 0
        
        # Output
        self.train_losses = []
        self.val_losses = []
            
        # Initialize model and optimizer
        self._initialize_training()

    
    def _initialize_training(self):
        self.model.to(self.device) 
    
    def _get_dataloaders(self):
        self.train_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "train/images"),
            mask_dir=os.path.join(self.data_dir, "train/masks"), 
            image_transform=self.train_transform,
            mask_transform=self.train_transform,
            normalize=self.normalize,
            negative=self.negative,
            verbose=False
            )
        
        self.val_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "val/images"),
            mask_dir=os.path.join(self.data_dir, "val/masks"),
            image_transform=self.train_transform,
            mask_transform=self.train_transform,
            normalize=self.normalize,
            negative=self.negative,
            verbose=False,  
            mean=self.train_dataset.mean,
            std=self.train_dataset.std
            )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)  

    def _train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
    
        # Forward pass
        outputs = self.model(inputs)
        
        if isinstance(self.loss_function, list):
            loss_func1, loss_func2, weight1, weight2 = self.loss_function
            loss = weight1 * loss_func1(outputs, targets) + weight2 * loss_func2(outputs, targets)
        else:
            loss = self.loss_function(outputs, targets)  
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
         # --- MEMORY MANAGEMENT ---
        del inputs, targets, outputs  # Explicitly delete tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
        
        return loss.item()
    
    def _validate(self):
        self.model.eval()
        val_loss = 0
        all_outputs = []
        all_targets = []

        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.val_loader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                outputs_probs = torch.sigmoid(outputs)
                
                if isinstance(self.loss_function, list):
                    loss_func1, loss_func2, weight1, weight2 = self.loss_function
                    loss = weight1 * loss_func1(outputs, targets) + weight2 * loss_func2(outputs, targets)
                else:
                    loss = self.loss_function(outputs, targets)  
                val_loss += loss.item()
                
                all_outputs.append(outputs_probs.detach())                
                all_targets.append(targets.detach())
                
                del inputs, targets, outputs, outputs_probs 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Aggregate predicitions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics_results = defaultdict()
        binary_metrics = BinaryMetrics(device=self.device)

        # Calculate metrics at 0.5 threshold
        results_05 = binary_metrics.calculate_metrics(all_outputs, all_targets, threshold=0.5)
        for metric_name, value in results_05.items():
            metrics_results[metric_name] = value

        # Calculate mIoU 
        thresholds = np.arange(0.5, 1.05, 0.05)
        metrics_results["miou"] = 0
        for threshold in thresholds:
            results_thresh = binary_metrics.calculate_metrics(all_outputs, all_targets, threshold=threshold)
            metrics_results["miou"] += results_thresh["IoU"]  # Access IoU from your class
        metrics_results["miou"] /= len(thresholds)

        # Calculate AP
        metrics_results["AP"] = average_precision_score(all_targets.cpu().numpy().flatten(), all_outputs.cpu().numpy().flatten())
            
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        
        del all_outputs, all_targets # Delete aggregated tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return val_loss, { **metrics_results}
    
    def train(self):
        # Initialize dataloaders 
        self._get_dataloaders()
        
        for epoch in range(self.current_epoch, self.total_epochs):
            self.current_epoch = epoch
            train_loss = 0
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Training 
            progress_bar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.total_epochs}', leave=False, file=sys.stdout)
            for batch in progress_bar:
                loss = self._train_step(batch)
                train_loss += loss
                
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)
                    
            if epoch == self.total_epochs-1:
                val_loss, self.metrics = self._validate()
                    
            # Update lr with scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
               
        self.last_dice = self.metrics["Dice"]
        self.last_loss = val_loss

class HyperparameterOptimizer:
    def __init__(self,
                 data_dir,
                 model_class,
                 hyperparameter_space,
                 study_name="segmentation_study",
                 storage=None,
                 output_dir="../optimization_runs",
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 objective_names=("val_loss", "dice"),
                 objective_directions=("minimize", "maximize")):

        self.data_dir = data_dir
        self.model_class = model_class
        self.hyperparameter_space = hyperparameter_space
        self.study_name = study_name
        self.device = device
        self.output_dir = output_dir
        self.objective_names = tuple(objective_names)
        self.objective_directions = tuple(objective_directions)

        if len(objective_names) != len(objective_directions):
            raise ValueError("objective_names and objective_directions must have the same length.")
        
        self.is_multi_objective = len(self.objective_directions) > 1
        
        os.makedirs(self.output_dir, exist_ok=True)
        # Database
        if storage is None:
            db_name = "optimization_studies.db"  # Use a common database file
            self.storage = f"sqlite:///{os.path.join(self.output_dir, db_name)}"
        else:
            self.storage = storage
        print(f"Using Optuna storage: {self.storage}")

        if self.is_multi_objective:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                directions=list(self.objective_directions),
                load_if_exists=True
            )
            print(f"Initialized multi-objective study '{self.study_name}' with objectives {self.objective_names} and directions {self.objective_directions}.")
        else:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=self.objective_directions[0],
                load_if_exists=True
            )
            print(f"Initialized single-objective study '{self.study_name}' with objective '{self.objective_names[0]}' and direction '{self.objective_directions[0]}'.")

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
            batch_size=other_params["batch_size"],  # Pass batch_size
            epochs=other_params["epochs"],  # Pass epochs
            warmup=warmup_params["warmup_steps"] if warmup_params["warmup_scheduler"] != "None" else 0, 
            train_transform=self.parse_transforms(other_params["transform"]),
            normalize=other_params["normalize"],
            negative=other_params["negative"],
        )

        try:
            # Train and evaluate the model
            trainer.train()

            # Get the validation loss
            val_loss = trainer.last_loss
            val_dice = trainer.last_dice
            
            values_to_check = (val_loss, val_dice) if self.is_multi_objective else (val_loss,) if self.objective_names[0] == "val_loss" else (val_dice,)
            
            if not all(np.isfinite(v) for v in values_to_check if v is not None):
                print(f"[WARNING] Trial {trial.number} resulted in non-finite objective values. Loss={val_loss}, Dice={val_dice}. Pruning.")
                raise optuna.TrialPruned("Non-finite objective values")
             
        except optuna.TrialPruned as e:
            print(f"[INFO] Trial {trial.number} pruned: {e}")
            # Clean up before re-raising
            del trainer, model, optimizer, lr_scheduler
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise e # Re-raise prune exception
         
        except Exception as e:
            print(f"[ERROR] Trial {trial.number} failed during training/evaluation: {e}")
            traceback.print_exc()
            # Clean up before pruning
            del trainer, model, optimizer, lr_scheduler
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"Training/Evaluation failed: {e}")
         
        # Store all relevant metrics for this trial
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("optimizer_params", optimizer_params)
        trial.set_user_attr("loss_params", loss_params)
        trial.set_user_attr("warmup_params", warmup_params)
        trial.set_user_attr("scheduler_params", scheduler_params)
        trial.set_user_attr("other_params", other_params)
        trial.set_user_attr("final_val_loss", val_loss)
        trial.set_user_attr("final_dice", val_dice)
    
        # Store training metrics if available
        if hasattr(trainer, "train_losses"):
            trial.set_user_attr("train_losses", trainer.train_losses)
        if hasattr(trainer, "val_losses"):
            trial.set_user_attr("val_losses", trainer.val_losses)
            
        # Log additional metrics if available from trainer
        if hasattr(trainer, "metrics"):
            for metric_name, metric_value in trainer.metrics.items():
                trial.set_user_attr(f"metric_{metric_name}", metric_value)

        del trainer, model, optimizer, lr_scheduler, loss_function
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # self._save_trial_results(trial)
            
        if self.is_multi_objective:
            # Ensure the order matches self.objective_names / self.objective_directions
            # Example: if names = ("val_loss", "dice")
            loss_idx = self.objective_names.index("val_loss") if "val_loss" in self.objective_names else -1
            dice_idx = self.objective_names.index("dice") if "dice" in self.objective_names else -1
            
            returned_values = [None] * len(self.objective_names)
            if loss_idx != -1: returned_values[loss_idx] = val_loss
            if dice_idx != -1: returned_values[dice_idx] = val_dice
            
            # Check if all expected values were filled
            if None in returned_values:
                print(f"[ERROR] Trial {trial.number}: Could not determine all objective values to return based on objective_names {self.objective_names}. Got loss={val_loss}, dice={val_dice}. Pruning.")
                raise optuna.TrialPruned("Objective value mapping failed.")
             
            return tuple(returned_values) 
        else:
            # Single objective
            if self.objective_names[0] == "val_loss":
                print(f"--- Trial {trial.number} Finished. Value (val_loss): {val_loss} ---")
                return val_loss
            elif self.objective_names[0] == "dice":
                print(f"--- Trial {trial.number} Finished. Value (dice): {val_dice} ---")
                return val_dice
            else:
                # Fallback or error if the single objective name is unexpected
                print(f"[ERROR] Trial {trial.number}: Unexpected single objective name '{self.objective_names[0]}'. Returning val_loss={val_loss} by default.")
                return val_loss


    def optimize(self, n_trials=100, timeout=None, gc_after_trial=True):
        """Run the optimization process with enhanced logging."""
        try:
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                timeout=timeout,
                gc_after_trial=gc_after_trial # Helps prevent memory leaks
            )
        except KeyboardInterrupt:
            print("\nOptimization stopped manually (KeyboardInterrupt).")
        except Exception as e:
            print(f"\n--- Optimization interrupted by error ---")
            print(f"Error: {e}")
            traceback.print_exc()
            print("--------------------------------------")
        finally:
            print("\n--- Optimization Finished ---")
            # --- Reporting based on mode ---
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not self.study.trials:
            print("No trials were completed or recorded.")
        elif not completed_trials:
            print("All trials failed or were pruned. Check your setup and logs.")
            self._save_results() # Save summary even if only failures
            return None if self.is_multi_objective else None
        
        
        if self.is_multi_objective:
            pareto_trials = self.study.best_trials
            print(f"Number of Pareto optimal trials found: {len(pareto_trials)}")
            if pareto_trials:
                print("\nPareto Optimal Trials:")
                for i, trial in enumerate(pareto_trials):
                    values_str = ", ".join([f"{name}={val:.6f}" for name, val in zip(self.objective_names, trial.values)])
                    print(f"  Trial {trial.number}: Values: ({values_str}), Params: {trial.params}")
            else:
                    print("No Pareto optimal trials found.")

            # Parameter importance (optional, target one objective)
            if completed_trials:
                try:
                    # Calculate importance based on the *first* objective by default
                    target_idx = 0
                    target_name = self.objective_names[target_idx]
                    importance = optuna.importance.get_param_importances(
                        self.study,
                        target=lambda t: t.values[target_idx] if t.values and len(t.values) > target_idx else float('nan'),
                    )
                    print(f"\nParameter importance (for '{target_name}'):")
                    for param, score in importance.items(): print(f"  {param}: {score:.4f}")
                except Exception as e: print(f"\nCould not calculate parameter importance: {e}")

            self._save_results() # Save multi-objective results
            return pareto_trials # Return list of best trials

        else: # Single Objective
            best_trial = self.study.best_trial
            print(f"\nBest trial found:")
            print(f"  Number: {best_trial.number}")
            print(f"  Value ({self.objective_names[0]}): {best_trial.value:.6f}")
            print(f"  Params: {best_trial.params}")

            # Parameter importance
            if completed_trials:
                try:
                    importance = optuna.importance.get_param_importances(self.study)
                    print("\nParameter importance:")
                    for param, score in importance.items(): print(f"  {param}: {score:.4f}")
                except Exception as e: print(f"\nCould not calculate parameter importance: {e}")

            self._save_results() # Save single-objective results
            return best_trial # Return the single best trial


    def _save_trial_results(self, trial):
        """Save detailed information about each completed trial in a folder named after the study."""
        study_name = self.study.study_name
        study_dir = os.path.join(self.output_dir, study_name)
        os.makedirs(study_dir, exist_ok=True)
        
        trial_dir = os.path.join(study_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        trial_data = {
            "number": trial.number,
            "state": str(trial.state),
            "values": trial.values, # List of objective values (always a list)
            "params": trial.params,
            "user_attrs": {}, # Store user attributes separately
            "datetime_start": str(trial.datetime_start),
            "datetime_complete": str(trial.datetime_complete),
        }
        
        # Add user attributes (metrics, parameter groups, etc.)
        for key, value in trial.user_attrs.items():
            if isinstance(value, np.ndarray):
                 value = value.tolist() # Convert numpy arrays
            elif key in ["train_losses", "val_losses"] and isinstance(value, list) and value:
                try:
                    df = pd.DataFrame({key: value})
                    csv_path = os.path.join(trial_dir, f"{key}.csv")
                    df.to_csv(csv_path, index_label="epoch")
                    trial_data["user_attrs"][key] = f"Saved to {os.path.basename(csv_path)}"
                except Exception as e:
                    print(f"Warning: Could not save {key} to CSV for trial {trial.number}: {e}")
                    trial_data["user_attrs"][key] = value # Store raw if saving failed
            else:
                # Ensure other complex objects are serializable if needed
                try:
                    yaml.dump(value, default_flow_style=False) # Test serializability
                    trial_data["user_attrs"][key] = value
                except TypeError:
                    print(f"Warning: Could not serialize user attribute '{key}' for trial {trial.number}. Storing as string.")
                    trial_data["user_attrs"][key] = str(value)


        # Save as YAML
        try:
            with open(os.path.join(trial_dir, "trial_details.yml"), "w") as f:
                yaml.dump(trial_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        except Exception as e:
             print(f"Error saving trial details YAML for trial {trial.number}: {e}")
                     
            
    def _save_results(self):
        """Enhanced version to save comprehensive results of the optimization."""
        # Get the study name from the study object
        study_name = self.study.study_name
        study_dir = os.path.join(self.output_dir, study_name)
        os.makedirs(study_dir, exist_ok=True)
        
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # --- Save Study Summary ---
        study_info = {
            "study_name": self.study.study_name,
            "objective_names": list(self.objective_names),
            "objective_directions": [d.name for d in self.study.directions], # Get actual directions
            "is_multi_objective": self.is_multi_objective,
            "n_trials_total": len(self.study.trials),
            "n_completed": len(completed_trials),
            "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
        }

        if completed_trials:
            if self.is_multi_objective:
                 try:
                     pareto_trials = self.study.best_trials
                     study_info["num_pareto_optimal_trials"] = len(pareto_trials)
                     study_info["pareto_optimal_trials_summary"] = [
                         {"number": t.number, "values": t.values} for t in pareto_trials
                     ]
                 except Exception as e:
                     print(f"Warning: Could not retrieve Pareto optimal trials for summary: {e}")
                     study_info["num_pareto_optimal_trials"] = "Error"
            else: # Single objective
                 try:
                     best_trial = self.study.best_trial
                     study_info["best_trial_number"] = best_trial.number
                     study_info["best_value"] = best_trial.value
                     # --- Save best hyperparameters separately for single objective ---
                     with open(os.path.join(study_dir, "best_hyperparameters.yml"), "w") as outfile:
                         yaml.dump(best_trial.params, outfile, default_flow_style=False)
                     print(f"Best hyperparameters saved to best_hyperparameters.yml")
                 except Exception as e:
                     print(f"Warning: Could not retrieve best trial for summary: {e}")
                     study_info["best_trial_number"] = "Error"
                     study_info["best_value"] = "Error"


        with open(os.path.join(study_dir, "study_summary.yml"), "w") as outfile:
            yaml.dump(study_info, outfile, default_flow_style=False, sort_keys=False)

        # --- Save Trials Dataframe ---
        try:
            # Include user attrs and make values columns explicit if multi-objective
            trials_df = self.study.trials_dataframe(multi_index=self.is_multi_objective) # Use multi-index for multi-obj values
            # Consider flattening multi-index or user_attrs if needed for simpler CSV
            trials_df.to_csv(os.path.join(study_dir, "all_trials.csv"), index=False)
            print(f"All trials data saved to all_trials.csv")
        except Exception as e:
            print(f"Warning: Could not save trials dataframe: {e}")

        if not completed_trials:
            print("Skipping visualization generation as there are no completed trials.")
            return

        print("Generating visualizations...")
        viz_dir = os.path.join(study_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        try:
            top_params_combined = set()# Will be populated if importance is calculated
            all_importances = {}
            objective_plot_files = {}
            
            plot_kwargs = {
                 "font": dict(family="Arial, sans-serif", size=12),
                 "margin": dict(l=40, r=20, t=60, b=40) # Increased top margin for title
            }
            
            for target_idx, target_name in enumerate(self.objective_names):
                print(f"\nGenerating plots relative to objective: '{target_name}' (Index {target_idx})")
                objective_plot_files[target_name] = {}
                
                target_func = lambda t: t.values[target_idx] if t.values and len(t.values) > target_idx else float('nan')

                # HISTORY
                try:
                    history_fig = plot_optimization_history(self.study, target=target_func, target_name=target_name)
                    history_title = f"Optimization History (Objective: {target_name})"
                    history_fig.update_layout(title_text=history_title, title_x=0.5, **plot_kwargs)
                    fname_base = f"optimization_history_{target_name}"
                    history_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                    objective_plot_files[target_name]['history'] = fname_base
                    print(f"  Saved plot: {fname_base}.png")
                except Exception as e: print(f"  Error generating optimization history for {target_name}: {e}")

                # IMPORTANCCE
                importance_dict = {}
                try:
                    importance_dict = optuna.importance.get_param_importances(self.study, target=target_func)
                    all_importances[target_name] = importance_dict
                    fig_imp = plot_param_importances( self.study, target=target_func)
                    imp_title = f"Parameter Importance (for {target_name})"
                    fig_imp.update_layout(title_text=imp_title, title_x=0.5, **plot_kwargs)
                    fname_base = f"parameter_importance_{target_name}"
                    fig_imp.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                    objective_plot_files[target_name]['importance'] = fname_base
                    print(f"  Saved plot: {fname_base}.png")
                    
                    current_top_params = list(importance_dict.keys())[:min(5, len(importance_dict))]
                    top_params_combined.update(current_top_params)
                except Exception as e: print(f"  Error generating parameter importance for {target_name}: {e}")               
                
            top_params = sorted(list(top_params_combined)) # Sort for consistent ordering
            print(f"\nTop parameters considered for Slice/Contour plots: {top_params}")
            
            #SLICE - CONTOURS     
            if top_params:
                objective_plot_files['slice'] = {} # Store slice filenames per param
                objective_plot_files['contour'] = {} # Store contour filenames per pair
                
                for target_idx, target_name in enumerate(self.objective_names):
                    print(f"\nGenerating Slice/Contour plots relative to '{target_name}' using top params...")
                    target_func = lambda t, idx=target_idx: t.values[idx] if t.values and len(t.values) > idx else float('nan')

                    for param in top_params:
                        # 3. Slice Plot for this objective
                        try:
                            slice_fig = plot_slice(self.study, params=[param], target=target_func, target_name=target_name)
                            slice_title = f"Slice: {param} (vs {target_name})"
                            slice_fig.update_layout(title_text=slice_title, title_x=0.5, **plot_kwargs)
                            # Add objective name to filename
                            fname_base = f"slice_{param}_vs_{target_name}"
                            slice_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                            # Store filename info (param -> {obj: file})
                            if param not in objective_plot_files['slice']: objective_plot_files['slice'][param] = {}
                            objective_plot_files['slice'][param][target_name] = fname_base
                        except Exception as e: print(f"  Error generating slice plot for {param} vs {target_name}: {e}")

                        # 4. Contour Plots for this objective
                        for other_param in top_params:
                            if param >= other_param: continue # Avoid duplicates
                            pair_key = tuple(sorted((param, other_param))) # Consistent key for pair
                            try:
                                contour_fig = plot_contour(self.study, params=[param, other_param], target=target_func, target_name=target_name)
                                contour_title = f"Contour: {param} vs {other_param} (Color: {target_name})"
                                contour_fig.update_layout(title_text=contour_title, title_x=0.5, **plot_kwargs)
                                # Add objective name to filename
                                fname_base = f"contour_{param}_vs_{other_param}_vs_{target_name}"
                                contour_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                                # Store filename info (pair -> {obj: file})
                                if pair_key not in objective_plot_files['contour']: objective_plot_files['contour'][pair_key] = {}
                                objective_plot_files['contour'][pair_key][target_name] = fname_base
                            except ValueError as ve: print(f"  Skipping contour plot for {param} vs {other_param} vs {target_name}: {ve}")
                            except Exception as e: print(f"  Error generating contour plot for {param} vs {other_param} vs {target_name}: {e}")
                    print(f"  Finished Slice/Contour plots for '{target_name}'.")

            if self.is_multi_objective:
                 try:
                     pareto_fig = plot_pareto_front(self.study, target_names=list(self.objective_names))
                     pareto_fig.update_layout(title_text="Pareto Front", title_x=0.5, **plot_kwargs)
                     fname_base = "pareto_front"
                     pareto_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                     objective_plot_files['pareto'] = fname_base # Store filename
                     print("\nSaved Pareto front plot.")
                 except ValueError as ve: print(f"\nCould not generate Pareto front plot: {ve}")
                 except Exception as e: print(f"\nError generating Pareto front plot: {e}")

            # 6. Parallel Coordinate
            if top_params:
                 try:
                     parallel_fig = plot_parallel_coordinate(self.study, params=top_params) # No target needed
                     parallel_title = "Parallel Coordinate Plot (Top Params vs Objectives)"
                     parallel_fig.update_layout(title_text=parallel_title, title_x=0.5, **plot_kwargs)
                     fname_base = "parallel_coordinate"
                     parallel_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                     objective_plot_files['parallel'] = fname_base # Store filename
                     print("Saved parallel coordinate plot.")
                 except Exception as e: print(f"Error generating parallel coordinate plot: {e}")

            # --- Create HTML index ---
            # Pass the collected filenames to the index generator
            self._create_visualization_index(
                viz_dir,
                top_params,
                objective_plot_files # Pass the dict containing filenames
            )
            print(f"\nVisualizations saved to {viz_dir}")

        except Exception as e:
            print(f"General error during visualization generation: {e}")
            traceback.print_exc()
            
    def _create_visualization_index(self, viz_dir, top_params, objective_plot_files):
        """Create an HTML index page for visualizations (adapts to mode)."""
        if not hasattr(self, 'study'): return

        # --- Basic Info (remains the same) ---
        study_name = self.study.study_name
        n_completed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
        try:
            directions_str = ', '.join(d.name.lower() for d in self.study.directions)
        except: directions_str = "Error"
        summary_rows_html = f"""
            <tr><th>Study Name</th><td>{study_name}</td></tr>
            <tr><th>Objective Names</th><td>{', '.join(self.objective_names)}</td></tr>
            <tr><th>Directions</th><td>{directions_str}</td></tr>
            <tr><th>Total Trials</th><td>{len(self.study.trials)}</td></tr>
            <tr><th>Completed</th><td>{n_completed}</td></tr>
            <tr><th>Pruned</th><td>{n_pruned}</td></tr>
            <tr><th>Failed</th><td>{n_failed}</td></tr>
        """
        # --- Mode Specific Summary Info (remains the same) ---
        pareto_info = ""
        if self.is_multi_objective:
             num_pareto_trials = 0
             if n_completed > 0:
                 try: num_pareto_trials = len(self.study.best_trials)
                 except: pass
             pareto_info = f"{num_pareto_trials} trial(s) on the Pareto front."
             summary_rows_html += f"<tr><th>Pareto Optimal Trials</th><td>{pareto_info}</td></tr>"
        else:
             best_trial_info = "N/A"
             if n_completed > 0:
                  try:
                       bt = self.study.best_trial
                       best_trial_info = f"Trial #{bt.number} (Value: {bt.value:.6f})"
                  except: best_trial_info = "Error retrieving"
             summary_rows_html += f"<tr><th>Best Trial</th><td>{best_trial_info}</td></tr>"


        # --- Generate HTML for Key Plots ---
        key_plots_html = ""
        # Pareto Front (if exists)
        if 'pareto' in objective_plot_files:
            fname = objective_plot_files['pareto']
            key_plots_html += f"""
                <div class="plot-item">
                    <h3>Pareto Front</h3>
                    <a href="{fname}.html" target="_blank"><img src="{fname}.png" alt="Pareto Front"></a>
                </div>"""
        # Parallel Coordinate (if exists)
        if 'parallel' in objective_plot_files:
            fname = objective_plot_files['parallel']
            key_plots_html += f"""
                <div class="plot-item">
                    <h3>Parallel Coordinate Plot</h3>
                    <a href="{fname}.html" target="_blank"><img src="{fname}.png" alt="Parallel Coordinate Plot"></a>
                </div>"""
        # Importance Plots (Loop through objectives)
        for obj_name in self.objective_names:
             if obj_name in objective_plot_files and 'importance' in objective_plot_files[obj_name]:
                 fname = objective_plot_files[obj_name]['importance']
                 key_plots_html += f"""
                    <div class="plot-item">
                        <h3>Importance (for {obj_name})</h3>
                        <a href="{fname}.html" target="_blank"><img src="{fname}.png" alt="Importance for {obj_name}"></a>
                    </div>"""
        # History Plots (Loop through objectives)
        for obj_name in self.objective_names:
             if obj_name in objective_plot_files and 'history' in objective_plot_files[obj_name]:
                 fname = objective_plot_files[obj_name]['history']
                 key_plots_html += f"""
                    <div class="plot-item">
                        <h3>History (for {obj_name})</h3>
                        <a href="{fname}.html" target="_blank"><img src="{fname}.png" alt="History for {obj_name}"></a>
                    </div>"""

        # --- Generate HTML for Slice Plots ---
        slice_plots_html = ""
        if 'slice' in objective_plot_files:
             for param in top_params: # Iterate through params first
                 if param in objective_plot_files['slice']:
                     slice_plots_html += f'<div class="plot-item"><h3>Slice Plots: {param}</h3><div class="plot-subcontainer">'
                     # Then iterate through objectives for this param
                     for obj_name in self.objective_names:
                          if obj_name in objective_plot_files['slice'][param]:
                              fname = objective_plot_files['slice'][param][obj_name]
                              slice_plots_html += f'''
                                  <div class="plot-subitem">
                                      <h4>vs {obj_name}</h4>
                                      <a href="{fname}.html" target="_blank"><img src="{fname}.png" alt="Slice: {param} vs {obj_name}"></a>
                                  </div>'''
                     slice_plots_html += '</div></div>' # Close subcontainer and plot-item

        # --- Generate HTML for Contour Plots ---
        contour_plots_html = ""
        if 'contour' in objective_plot_files:
             for pair_key in objective_plot_files['contour']: # Iterate through param pairs
                 param1, param2 = pair_key
                 contour_plots_html += f'<div class="plot-item"><h3>Contour Plots: {param1} vs {param2}</h3><div class="plot-subcontainer">'
                 # Then iterate through objectives for this pair
                 for obj_name in self.objective_names:
                     if obj_name in objective_plot_files['contour'][pair_key]:
                         fname = objective_plot_files['contour'][pair_key][obj_name]
                         contour_plots_html += f'''
                             <div class="plot-subitem">
                                 <h4>Color: {obj_name}</h4>
                                 <a href="{fname}.html" target="_blank"><img src="{fname}.png" alt="Contour: {param1} vs {param2}, Color: {obj_name}"></a>
                             </div>'''
                 contour_plots_html += '</div></div>' # Close subcontainer and plot-item


        # --- Best Params Link (remains the same) ---
        best_params_link_html = ""
        if not self.is_multi_objective and os.path.exists(os.path.join(os.path.dirname(viz_dir), "best_hyperparameters.yml")):
            best_params_link_html = '<p><a href="../best_hyperparameters.yml" target="_blank">Download best hyperparameters (YAML)</a></p>'

        # --- HTML Template (adjusted slightly for subplots) ---
        html_content = f"""<!DOCTYPE html>
    <html><head>
        <meta charset="UTF-8">
        <title>Optuna Optimization Results - {study_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1600px; margin: 10px auto; padding: 20px; background-color:#fff; color: #333; }}
            h1, h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px;}}
            h1 {{ font-size: 1.8em; text-align: center; margin-bottom: 25px;}}
            h2 {{ font-size: 1.4em; margin-top: 35px; }}
            h3 {{ margin: 0 0 15px 0; font-size: 1.15em; color: #444; text-align: center; border-bottom: 1px dashed #ddd; padding-bottom: 8px;}}
            h4 {{ margin: 10px 0 5px 0; font-size: 1.0em; color: #555; text-align: center; }}
            .section {{ margin-bottom: 30px; background-color:#fdfdfd; padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}}
            .plot-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; }}
            .plot-item {{ border: 1px solid #e0e0e0; padding: 15px; background-color:#f9f9f9; border-radius: 4px; transition: box-shadow 0.2s ease;}}
            .plot-item:hover {{ box-shadow: 0 3px 8px rgba(0,0,0,0.1); }}
            .plot-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; display: block; margin-top: 5px; border-radius: 3px;}}
            /* Styles for subplots within an item */
            .plot-subcontainer {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; }}
            .plot-subitem {{ flex: 1 1 45%; min-width: 200px; text-align: center; }}
            a {{ color: #0066cc; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 0.95em;}}
            th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: left; vertical-align: top;}}
            th {{ background-color: #f2f2f2; font-weight: bold; width: 25%;}}
            code {{ background-color: #eee; padding: 2px 5px; border-radius: 3px; font-family: Consolas, monospace; font-size: 0.9em; border: 1px solid #ddd;}}
            p {{ line-height: 1.5; }}
        </style>
    </head><body>
        <h1>Optuna Optimization Results: <code>{study_name}</code></h1>
        <div class="section">
            <h2>Study Summary</h2>
            <table><tbody>{summary_rows_html}</tbody></table>
        </div>
        <div class="section">
            <h2>Key Visualizations</h2>
            <div class="plot-container">{key_plots_html}</div>
        </div>
        <div class="section">
            <h2>Parameter Slice Plots (vs Objectives)</h2>
            <div class="plot-container">{slice_plots_html}</div>
        </div>
        <div class="section">
            <h2>Parameter Contour Plots (vs Objectives)</h2>
            <div class="plot-container">{contour_plots_html}</div>
        </div>
        <div class="section">
            <h2>Trial Data & Downloads</h2>
            <p>Detailed data for each trial (parameters, objectives, attributes, logs) is stored in the <code>trial_<number></code> subdirectories within the main study folder.</p>
            <p><a href="../all_trials.csv" target="_blank">Download all trials data summary (CSV)</a></p>
            {best_params_link_html}
            <p><a href="../study_summary.yml" target="_blank">Download study summary (YAML)</a></p>
        </div>
        <footer style="text-align: center; margin-top: 30px; font-size: 0.9em; color: #777;">
            Generated by HyperparameterOptimizer class
        </footer>
    </body></html>"""

        # Write the HTML file (remains the same)
        try:
            with open(os.path.join(viz_dir, "index.html"), "w", encoding='utf-8') as f:
                f.write(html_content)
        except Exception as e:
             print(f"Error writing visualization index.html: {e}")

                
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
