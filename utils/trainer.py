import os
import sys
import tqdm
import csv
import matplotlib.pyplot as plt
import yaml
from collections import defaultdict
import imageio
from sklearn.metrics import precision_recall_curve, average_precision_score
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
from torch import optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

from data.dataset import SegmentationDataset
from utils.metrics import BinaryMetrics

class Trainer():
    
    def __init__(self, 
                 data_dir,
                 model,
                 batch_size=16,
                 normalize=False,
                 train_transform= transforms.ToTensor(),
                 optimizer=optim.Adam, 
                 loss_function=nn.BCELoss(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr_scheduler=None,
                 warmup=3,
                 epochs=10,
                 output_dir="../runs",
                 early_stopping=50,
                 verbose=True, 
                 config=None,
                 save_output=True
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

        
        # Training parameters 
        self.epochs = epochs
        self.warmup_epochs = warmup if isinstance(self.lr_scheduler, optim.lr_scheduler.SequentialLR) else 0
        self.total_epochs = self.warmup_epochs + self.epochs
        self.current_epoch = 0
        
        # Early stopping
        self.early_stopping = early_stopping
        self.best_loss = float("inf")
        self.early_stopping_counter = 0
        
        # Output
        self.save_output = save_output
        self.writer = None
        if self.save_output:
            self.output_dir = output_dir
            self._initialize_output_folder()
            self.config = config
            self._save_config()
        
            self._initialize_csv()
            self.writer = SummaryWriter(log_dir=self.exp_dir)
            
        self.verbose = verbose
        self.debugging = False
        
        # Initialize model and optimizer
        self._initialize_training()

    
    def _initialize_training(self):
        self.model.to(self.device) 
        if self.debugging:
            print(f"[DEBUG] Model moved to device: {self.device}")
        
    def _initialize_output_folder(self):
        
        exp_dir = os.path.join(self.output_dir, "exp")

        if not os.path.exists(exp_dir):
            self.exp_dir = exp_dir
        else:
            counter = 1
            while True:
                new_dir = f"{exp_dir}{counter}"
                if not os.path.exists(new_dir):
                    self.exp_dir = new_dir
                    break
                counter += 1        
        
        self.models_dir = os.path.join(self.exp_dir, 'models')
        self.results_dir = os.path.join(self.exp_dir, "results")
        
        for sub_dir in [self.exp_dir, self.models_dir, self.results_dir]:
            os.makedirs(sub_dir, exist_ok=True)
            
    def _save_config(self):
        if self.config:
            config_file  = {
                "Model Parameters:" : {
                    "model" : self.config.model,
                    "attention" : self.config.attention,
                    "batchnorm" : self.config.batchnorm
                },
                "Encoder Parameters:" : {
                    "encoder" : self.config.encoder,
                    "weights" : self.config.weights
                },
                "Optimizer Parameters:" : {
                    "optimizer" : self.config.optimizer,
                    "lr" : self.config.lr,
                    "momentum" : self.config.momentum,
                    "weight_decay" : self.config.weight_decay
                },
                "Scheduler Parameters:" : {
                    "scheduler" : self.config.scheduler,
                    "start_factor" : self.config.start_factor,
                    "end_factor" : self.config.end_factor,
                    "iterations" : self.config.iterations,
                    "t_max" : self.config.t_max,
                    "eta_min" : self.config.eta_min,
                    "step_size" : self.config.step_size,
                    "gamma_lr" : self.config.gamma_lr,
                    "warmup_epochs" : self.config.warmup_steps
                },
                "Loss Function Parameters:" : {
                    "loss_function" : self.config.loss_function,
                    "loss_function1" : self.config.loss_function1,
                    "loss_function1_weight" : self.config.loss_function1_weight,
                    "loss_function2" : self.config.loss_function2,
                    "loss_function2_weight" : self.config.loss_function2_weight,
                    "alpha" : self.config.alpha,
                    "beta" : self.config.beta,
                    "gamma" : self.config.gamma
                },
                "Directories:" : {
                    "data_dir" : self.config.data_dir,
                    "output_dir" : self.config.output_dir
                },
                "Training Parameters:" : {
                    "epochs" : self.config.epochs,
                    "batch_size" : self.config.batch_size
                },
                "Dataset Parameters:" : {
                    "normalize" : self.config.normalize,
                    "transform" : self.config.transform
                },
            }
            
            with open(os.path.join(self.exp_dir, "config.yml"), "w+") as outfile:
                yaml.dump(config_file, outfile, sort_keys=False)
    
    def _initialize_csv(self):
        self.results_csv = os.path.join(self.results_dir, "results.csv")

        if not os.path.exists(self.results_csv):
            with open(self.results_csv, mode="w+", newline="") as file:
                writer = csv.writer(file)

                # Get metric names from BinaryMetrics class
                binary_metrics = BinaryMetrics()
                metric_names = list(binary_metrics.metrics.keys())  # Get a list of the metric names

                headers = ["Epochs", "Train Loss", "Val Loss", "Learning Rate"] + metric_names + ["mAP", "mIoU"] # Modified Header
                writer.writerow(headers)
               
    def _log_results_to_csv(self, epoch, train_loss, val_loss):
        if self.save_output:
            if self.results_csv:
                with open(self.results_csv, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    row = [epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr']]
                    row += [self.metrics.get(metric, 0) for metric in self.metrics.keys()]
                    writer.writerow(row)
    
    def _get_dataloaders(self):
        self.train_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "train/images"),
            mask_dir=os.path.join(self.data_dir, "train/masks"), 
            image_transform=self.train_transform,
            mask_transform=self.train_transform,
            normalize=self.normalize,
            verbose=False
            )
        
        self.val_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "val/images"),
            mask_dir=os.path.join(self.data_dir, "val/masks"),
            image_transform=self.train_transform,
            mask_transform=self.train_transform,
            normalize=self.normalize,
            verbose=False,  
            mean=self.train_dataset.mean,
            std=self.train_dataset.std
            )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)  
        
        if self.save_output:
            self.image_evolution_idx = np.random.randint(0, len(self.val_dataset)-1)

    def _save_checkpoint(self, checkpoint_path):
        if self.save_output:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'dataset_mean': self.train_dataset.mean,
                'dataset_std': self.train_dataset.std
            }
            torch.save(checkpoint, checkpoint_path)
            if self.verbose:
                print(f'Checkpoint saved at {checkpoint_path}')
    
    def _train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
    
        # Forward pass
        outputs = self.model(inputs)
        outputs_probs = torch.sigmoid(outputs)     
        
        if isinstance(self.loss_function, list):
            loss_func1, loss_func2, weight1, weight2 = self.loss_function
            loss = weight1 * loss_func1(outputs_probs, targets) + weight2 * loss_func2(outputs_probs, targets)
        else:
            loss = self.loss_function(outputs_probs, targets)  
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
         # --- MEMORY MANAGEMENT ---
        del inputs, targets, outputs, outputs_probs  # Explicitly delete tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
        
        if self.debugging:
            print(f"[DEBUG] Train step computed loss: {loss.item():.4f}")
        
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
                
                targets = targets.long()
                
                if isinstance(self.loss_function, list):
                    loss_func1, loss_func2, weight1, weight2 = self.loss_function
                    loss = weight1 * loss_func1(outputs_probs, targets) + weight2 * loss_func2(outputs_probs, targets)
                else:
                    loss = self.loss_function(outputs_probs, targets)  
                val_loss += loss.item()
                
                # outputs_binary = (outputs_probs > 0.5).long()
                
                # if self.debugging:
                #     print("----------Inputs & Targets-------------")
                #     print(f"\n Batch {batch_idx}")
                #     print("inputs shape:", inputs.shape)
                #     print("Targets shape:", targets.shape)
                #     print("targets unique values:", torch.unique(targets))
                #     print("Targets dtype:" ,targets.dtype)
                #     print("Targets long shape:", targets.shape)
                #     print("targets long unique values:", torch.unique(targets))
                #     print("Targets long dtype:", targets.dtype)
                    
                #     print("----------Outputs (Logits)-------------")               
                #     print("outputs shape:", outputs.shape)
                #     print("outputs unique values:", torch.unique(outputs))
                #     print("outputs dtype:", outputs.dtype)
                    
                #     print("----------Outputs (Probabilities)-------------")
                #     print("probs shape:", outputs_probs.shape)
                #     print("prnbs unique values:", torch.unique(outputs_probs))
                #     print("probs dtype:", outputs_probs.dtype)
                                   
                #     print("----------Outputs (Binary)-------------")
                #     print("binary shape:", outputs_binary.shape)
                #     print("binary unique values:", torch.unique(outputs_binary))
                #     print("binary dtype:", outputs_binary.dtype)

                all_outputs.append(outputs_probs.detach())
                # all_outputs.append(outputs_binary.detach())
                
                all_targets.append(targets.detach())
                
                del inputs, targets, outputs, outputs_probs 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Aggregate predicitions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # if self.debugging:
        #     all_outputs_flat = all_outputs.view(-1)
        #     all_targets_flat = all_targets.view(-1)
            
        #     # Check if all values are between 0 and 1
        #     if not torch.all((all_outputs_flat >= 0) & (all_outputs_flat <= 1)):
        #         raise ValueError("All output values should be between 0 and 1.")
        #     if not torch.all((all_targets_flat >= 0) & (all_targets_flat <= 1)):
        #         raise ValueError("All target values should be between 0 and 1.")
        metrics_results = defaultdict()
        binary_metrics = BinaryMetrics()

        # Calculate metrics at 0.5 threshold
        results_05 = binary_metrics.calculate_metrics(all_outputs, all_targets, threshold=0.5)
        for metric_name, value in results_05.items():
            metrics_results[metric_name] = value

        # Calculate mIoU 
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        metrics_results["miou"] = 0
        for threshold in thresholds:
            results_thresh = binary_metrics.calculate_metrics(all_outputs, all_targets, threshold=threshold)
            metrics_results["miou"] += results_thresh["IoU"]  # Access IoU from your class
        metrics_results["miou"] /= len(thresholds)

        # Calculate AP
        metrics_results["AP"] = average_precision_score(all_targets.cpu().numpy().flatten(), all_outputs.cpu().numpy().flatten())
            
        val_loss /= len(self.val_loader)
        
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
            if self.debugging:
                print(f"[DEBUG] Starting Epoch {epoch+1}/{self.total_epochs} - Current LR: {current_lr:.6f}")
            
            # Training 
            progress_bar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.total_epochs}', leave=False, file=sys.stdout)
            for batch in progress_bar:
                loss = self._train_step(batch)
                train_loss += loss
                
            train_loss /= len(self.train_loader)


            val_loss, self.metrics = self._validate()
            
            if self.save_output:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)

                # Log each metric individually
                for name, result in self.metrics.items():  # Iterate through the metrics dictionary
                    self.writer.add_scalar(f"Metrics/{name}", result, epoch)

            if self.current_epoch > self.warmup_epochs - 1:
                # Print metrics - dynamically access metric names
                print_str = f'\nEpoch {epoch+1}/{self.total_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}'
                for name, result in self.metrics.items():
                    print_str += f' - {name.capitalize()}: {result:.4f}'  # Add each metric to the string
                print(print_str)

                if self.save_output:
                    self._log_results_to_csv(epoch + 1, train_loss, val_loss)  # Pass metrics to _log_results_to_csv
                    self.image_evolution()
            
                # Early stopping
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.early_stopping_counter = 0 
                    if self.save_output:
                        best_path = f"{self.models_dir}/best.pth"
                        self._save_checkpoint(best_path)
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                if self.debugging:
                    print(f"[DEBUG] Epoch {epoch+1} is in warmup phase.")
                    
            # Update lr with scheduler
            if self.lr_scheduler:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.lr_scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if self.debugging:
                    print(f"[DEBUG] LR Scheduler updated LR from {old_lr:.6f} to {new_lr:.6f}")

        if self.save_output:
            if self.best_loss != val_loss: 
                last_path = f"{self.models_dir}/last.pth"
                self._save_checkpoint(last_path)
            
            self.writer.flush() 
            self.writer.close()   
            self.loss_plots()
            self.find_best_threshold()
            self.create_animation()
        
        self.last_loss = val_loss
    
    def loss_plots(self):
        with open(self.results_csv, mode="r", newline="") as file:
            lines = file.readlines()
        
        train_loss = []
        val_loss = []
        for line in lines[1:]:  # Skip the header line
            values = line.strip().split(",")
            train_loss.append(float(values[1]))
            val_loss.append(float(values[2]))
    
        # Train_loss
        n = len(train_loss) 
        # k = 25
        # train_avg = []
        # val_avg = []
        # for i in range(k, n-k):
        #     train_avg.append(sum(train_loss[(i-k):(i+k+1)])/(2*k+1))
        #     val_avg.append(sum(val_loss[(i-k):(i+k+1)])/(2*k+1))
            
        fig_train = plt.figure(figsize=(8,6))
        plt.plot(range(n), train_loss, alpha=0.5, label="Train Loss")
        plt.plot(range(n), val_loss, alpha=0.5, label="Val Loss")
        # plt.plot(range(k,n-k), train_avg, 'maroon')
        plt.legend(loc=1)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.savefig(os.path.join(self.results_dir, "loss.png"))
        
        
    
    def pr_curve(self, outputs, targets):
        precision, recall, _ = precision_recall_curve(targets, outputs, pos_label=1)
        plt.figure(figsize=(8, 6))
        plt.plot(recall[1:], precision[1:], label='PR Curve')
        plt.ylim((0,1))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(self.results_dir, "pr_curve.png"))
        plt.close() # close fig
    
    def image_evolution(self):
        """
        Visualizes the model's output during training, adding a small amount of space around images.

        Args:
            results_dir (str): Path to the directory for saving results.
            current_epoch (int): Current epoch number.
            val_dataset (torch.utils.data.Dataset): Validation dataset.
            device (str): Device to use for computations (e.g., 'cpu' or 'cuda').
        """

        self.image_evolution_dir = os.path.join(self.results_dir, "image_evolution")
        if not os.path.exists(self.image_evolution_dir):
            os.makedirs(self.image_evolution_dir)

        try:
            image, mask = self.val_dataset[self.image_evolution_idx]
            image = image.to(self.device)

            self.model.eval()
            with torch.no_grad():
                out = self.model(image.unsqueeze(0))

                out_np = out.cpu().detach().numpy().squeeze().squeeze()
                mask_np = mask.cpu().detach().numpy().squeeze()


                # --- Plotting with controlled whitespace ---
                num_subplots = 3
                binary_threshold = 0.5

                fig, axes = plt.subplots(1, num_subplots, figsize=(14, 7))  # Create figure and axes objects
                fig.subplots_adjust(wspace=0.1, hspace=0.1)  # Spacing between subplots

                # Prediction Plot
                im1 = axes[0].imshow(out_np, cmap='gray' if out_np.ndim == 2 else None)  # Capture the image object
                axes[0].set_title('Prediction')
                axes[0].axis('off')

                # Binary Prediction Plot
                im2 = axes[1].imshow((out_np > binary_threshold), cmap='gray' if out_np.ndim == 2 else None)  # Capture the image object
                axes[1].set_title('Binary Prediction')
                axes[1].axis('off')

                # Mask Plot
                im3 = axes[2].imshow(mask_np, cmap='gray' if mask_np.ndim == 2 else None)  # Capture the image object
                axes[2].set_title('Mask')
                axes[2].axis('off')

                # Set layout to 'constrained'
                fig.set_layout_engine("constrained")

                filename = f"{self.current_epoch + 1}.png"
                filepath = os.path.join(self.image_evolution_dir, filename)
                plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1, dpi=300)  # Save with bounding box and DPI
                plt.close(fig)  # Close the figure, not just plt
                del out, out_np, mask_np, image

        except Exception as e:
            print(f"Error during image evolution: {e}")
                
    def create_animation(self):        
                
        image_files = sorted([f for f in os.listdir(self.image_evolution_dir) if os.path.isfile(os.path.join(self.image_evolution_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))])
        images = []
        for image_file in image_files:
            image_path = os.path.join(self.image_evolution_dir, image_file)
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                return
        
        try:
            output_path = os.path.join(self.image_evolution_dir, "evolution.gif")
            imageio.mimsave(output_path, images, fps=self.epochs/4, loop=1,)
        except:
            print(f"Error creating animation: {e}")        
                        
    def find_best_threshold(self):
        self.model.eval()
        all_outputs = []
        all_targets = []

        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.val_loader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                outputs_probs = torch.sigmoid(outputs)

                targets = targets.long()

                all_outputs.append(outputs_probs.detach())
                all_targets.append(targets.detach())

        # Aggregate predicitions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        best_iou = 0
        best_threshold = 0

        binary_metrics = BinaryMetrics()  # Initialize your metrics class

        for threshold in np.arange(0, 1.05, 0.05):
            results = binary_metrics.calculate_metrics(all_outputs, all_targets, threshold=threshold)
            temp_iou = results["IoU"] # Access IoU from your class

            if temp_iou > best_iou:
                best_iou = temp_iou
                best_threshold = threshold

        print(f"Best threshold: {best_threshold} with IoU: {best_iou:.4f}")
        
        self.pr_curve(outputs=all_outputs.numpy().flatten(), targets=all_targets.numpy().flatten())
            
        
        
