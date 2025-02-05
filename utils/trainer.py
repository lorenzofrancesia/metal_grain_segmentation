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
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
from torch import optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

from data.dataset import SegmentationDataset

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
                headers = ["Epochs", "Train Loss", "Val Loss", "Learning Rate", "IoU", "Dice", "Precision", "Recall", "Accuracy",  "mAP", "mIoU"]
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
            verbose=False        
            )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)    

    def _save_checkpoint(self, checkpoint_path):
        if self.save_output:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_epoch': self.current_epoch,
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
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
        
        tp, fp, fn, tn = smp.metrics.get_stats(all_outputs, all_targets, mode="binary", threshold=0.5)
        
        metrics_results["iou"]= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro" ).item()
        metrics_results["dice"] = smp.metrics.f1_score(tp, tn, fn ,tn, reduction="micro").item()
        metrics_results["precision"] = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
        metrics_results["recall"]= smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item()
        metrics_results["accuracy"] = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
        metrics_results["mAP"] = average_precision_score(all_targets.numpy().flatten(), all_outputs.numpy().flatten())
        
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        metrics_results["miou"] = 0
        for threshold in thresholds:
            tp, fp, fn, tn = smp.metrics.get_stats(all_outputs, all_targets, mode="binary", threshold=threshold)
            metrics_results["miou"] += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro" ).item()
        metrics_results["miou"] /= len(thresholds)
            
            
        # metrics_results = self.metrics.calculate_metrics(all_outputs, all_targets)
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
                for name, result in self.metrics.items():
                    self.writer.add_scalar(f"Metrics/{name}", result, epoch)

            if self.current_epoch > self.warmup_epochs-1:
                print(f'\nEpoch {epoch+1}/{self.total_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}'
                      f'- mIoU: {self.metrics["miou"]:.4f} - IoU: {self.metrics["iou"]:.4f}'
                      f'- mAP: {self.metrics["mAP"]:.4f} - Acc: {self.metrics["accuracy"]:.4f}'
                      f'- Dice: {self.metrics["dice"]:.4f}')
                if self.save_output:
                    self._log_results_to_csv(epoch+1, train_loss, val_loss)
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
            self.pr_curve()
            self.create_animation()
    
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
        k = 25
        train_avg = []
        for i in range(k, n-k):
            train_avg.append(sum(train_loss[(i-k):(i+k+1)])/(2*k+1))
            
        fig_train = plt.figure(figsize=(8,6))
        plt.plot(range(n), train_loss, alpha=0.5)
        plt.plot(range(k,n-k), train_avg, 'maroon')
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("Train Loss")
        plt.savefig(os.path.join(self.results_dir, "train_loss.png"))
        
        val_avg = []
        for i in range(k, n-k):
            val_avg.append(sum(val_loss[(i-k):(i+k+1)])/(2*k+1))
            
        fig_val = plt.figure(figsize=(8,6))
        plt.plot(range(n), val_loss, alpha=0.5)
        plt.plot(range(k,n-k), val_avg, 'maroon')
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title("Val Loss")
        plt.savefig(os.path.join(self.results_dir, "val_loss.png"))
        plt.close(fig_val) #close fig
    
    def pr_curve(self):
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

                all_outputs.append(outputs_probs.detach())
                all_targets.append(targets.detach())
        
        # Aggregate predicitions and targets
        all_outputs = torch.cat(all_outputs, dim=0).numpy().flatten()
        all_targets = torch.cat(all_targets, dim=0).numpy().flatten()

        
        precision, recall, _ = precision_recall_curve(all_targets, all_outputs, pos_label=1)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(self.results_dir, "pr_curve.png"))
        plt.close() # close fig
        del all_outputs, all_targets
    
    def image_evolution(self):
        """
        This function visualizes the output of the model during training.

        Args:
            results_dir (str): Path to the directory for saving results.
            current_epoch (int): Current epoch number.
            val_dataset (torch.utils.data.Dataset): Validation dataset.
            device (str): Device to use for computations (e.g., 'cpu' or 'cuda').
        """

        # Create directory if it doesn't exist
        self.image_evolution_dir = os.path.join(self.results_dir, "image_evolution")
        if not os.path.exists(self.image_evolution_dir):
            os.makedirs(self.image_evolution_dir)

        try:
            # Get image and mask
            image, mask = self.val_dataset[0]
            image = image.to(self.device)

            # Model prediction
            self.model.eval()
            with torch.no_grad():
                out = self.model(image.unsqueeze(0))[0]

                # Convert to numpy arrays
                out_np = out.cpu().detach().numpy().squeeze().squeeze()
                mask_np = mask.cpu().detach().numpy().squeeze()

                # Print statistics (optional)
                # print(f"MAX: {max(mask_np)}")
                # print(f"MIN: {min(mask_np)}")

                # Create plots
                plt.figure(figsize=(14, 7))

                # Plot options (make these configurable parameters if needed)
                num_subplots = 3
                binary_threshold = 0.5

                # Plot prediction
                plt.subplot(1, num_subplots, 1)
                plt.imshow(out_np, cmap='gray' if out_np.ndim == 2 else None)
                plt.title('Prediction')
                plt.axis('off')

                # Plot binary prediction
                plt.subplot(1, num_subplots, 2)
                plt.imshow((out_np > binary_threshold), cmap='gray' if out_np.ndim == 2 else None)
                plt.title('Binary Prediction')
                plt.axis('off')

                # Plot mask
                plt.subplot(1, num_subplots, 3)
                plt.imshow(mask_np, cmap='gray' if mask_np.ndim == 2 else None)
                plt.title('Mask')
                plt.axis('off')

                filename = f"{self.current_epoch+1}.png"
                filepath = os.path.join(self.image_evolution_dir, filename)
                plt.savefig(filepath)
                plt.close() #close fig
                del out, out_np, mask_np, image #delete vars

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
                        
                    
        
        
