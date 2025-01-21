import os
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
from torch import optim
import torchvision.transforms as transforms

from utils.metrics import BinaryMetrics
from data.dataset import SegmentationDataset

class Trainer():
    
    def __init__(self, 
                 data_dir,
                 model,
                 batch_size=16,
                 normalize=False,
                 train_transform= transforms.ToTensor(),
                 optimizer=optim.Adam, 
                 loss_function=nn.BCEWithLogitsLoss(),
                 metrics=BinaryMetrics(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr_scheduler=None,
                 epochs=10,
                 output_dir=None,
                 resume_from_checkpoint=False,
                 checkpoint_path=None,
                 early_stopping=10,
                 verbose=True
                 ):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.device = device
        self.lr_scheduler = lr_scheduler
        
        # Dataset
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Training parameters 
        self.epochs = epochs
        self.current_epoch = 0
        self.global_step = 0
        
        # Early stopping
        self.early_stopping = early_stopping
        self.best_loss = float("inf")
        self.early_stopping_counter = 0
        
        # Checkpointing
        self.output_dir = output_dir
        self.models_dir = os.path.join(self.output_dir, "models")
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        
        # Initialize model and optimizer
        self._initialize_training()
        self._initialize_csv()
    
    
    def _initialize_training(self):
        
        self.model.to(self.device)

        if self.resume_from_checkpoint:
            self._load_checkpoint()
        
        #if self.verbose:
        #    print(self.model)
        #    print(self.optimizer)
        
    def _initialize_csv(self):
        self.results_csv = os.path.join(self.output_dir, "results", "results.csv")
        
        if not os.path.exists(self.results_csv):
            with open(self.results_csv, mode="w+", newline="") as file:
                writer = csv.writer(file)
                headers = ["Epochs", "Train Loss", "Val Loss", "Learning Rate", "Precision", "Recall", "F1", "Accuracy", "Dice", "IoU"]
                writer.writerow(headers)       
                      
    def _log_results_to_csv(self, epoch, train_loss, metrics):
        if self.results_csv:
            with open(self.results_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                row = [epoch, train_loss, metrics["val_loss"], self.optimizer.param_groups[0]['lr']]
                row += [metrics.get(metric, 0) for metric in self.metrics.metrics.keys()]
                writer.writerow(row)
    
    def _get_dataloaders(self):
        self.train_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "train/images"),
            mask_dir=os.path.join(self.data_dir, "train/masks"), 
            image_transform=self.train_transform,
            mask_transform=self.train_transform,
            normalize=self.normalize,
            verbose=self.verbose
            )
        
        self.val_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "val/images"),
            mask_dir=os.path.join(self.data_dir, "val/masks"),
            image_transform=self.train_transform,
            mask_transform=self.train_transform,
            normalize=self.normalize,
            verbose=self.verbose        
            )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)    

    def _save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'output_dir': self.output_dir
        }
        torch.save(checkpoint, checkpoint_path)
        if self.verbose:
            print(f'Checkpoint saved at {checkpoint_path}')
    
    def _load_checkpoint(self):
        
        if not self.checkpoint_path:
            raise ValueError("Provide path to the checkpoint")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']
        if self.lr_scheduler and checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.output_dir = checkpoint['output_dir']
        if self.verbose:
            print(f'Loaded checkpoint from {self.checkpoint_path}')
            
    def _train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
    
        # Forward pass
        outputs = self.model(inputs) 
        outputs_probs = torch.sigmoid(outputs)       
        loss = self.loss_function(outputs_probs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _validate(self):
        self.model.eval()
        val_loss = 0
        all_outputs = []
        all_targets = []
                
        with torch.inference_mode():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                                
                outputs = self.model(inputs)
                outputs_probs = torch.sigmoid(outputs)
                
                targets = targets.long()
                                
                loss = self.loss_function(outputs_probs, targets)
                val_loss += loss.item()
                
                outputs_binary = (outputs_probs > 0.5).long()
                
                all_outputs.append(outputs_binary.detach())
                all_targets.append(targets.detach())
        
        # Aggregate predicitions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # # Flatten the tensors
        # all_outputs_flat = all_outputs.view(-1)
        # all_targets_flat = all_targets.view(-1)
        
        # # Check if all values are between 0 and 1
        # if not torch.all((all_outputs_flat >= 0) & (all_outputs_flat <= 1)):
        #     raise ValueError("All output values should be between 0 and 1.")
        # if not torch.all((all_targets_flat >= 0) & (all_targets_flat <= 1)):
        #     raise ValueError("All target values should be between 0 and 1.")
               
        metrics_results = self.metrics.calculate_metrics(all_outputs, all_targets)
        val_loss /= len(self.val_loader)
        
        return {'val_loss': val_loss, **metrics_results}
    
    def train(self):
        # Initialize dataloaders 
        self._get_dataloaders()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            train_loss = 0
            
            # Training 
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False)
            for batch in progress_bar:
                loss = self._train_step(batch)
                train_loss += loss
                self.global_step += 1
                
            train_loss /= len(self.train_loader)
            
            # Validation if val_dataloader exists
            val_loss = 0
            if self.val_loader:
                val_results = self._validate()
                val_loss = val_results["val_loss"]
                if self.verbose:
                    print(f'\nEpoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - P: {val_results["Precision"]:.4f} - R: {val_results["Recall"]:.4f} - Acc: {val_results["Accuracy"]:.4f} - F1: {val_results["F1"]:.4f} - IoU: {val_results["IoU"]:.4f} - Dice: {val_results["Dice"]:.4f}')
                self._log_results_to_csv(epoch+1, train_loss, val_results)
            else:
                val_loss = train_loss
                if self.verbose:
                    print(f'\nEpoch {epoch+1}/{self.epochs} - Training Loss: {train_loss:.4f}')
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.early_stopping_counter = 0
                best_path = f"{self.models_dir}/best.pth"
                self._save_checkpoint(best_path)
                
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
            # Update lr with scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        if self.best_loss != val_loss: 
            last_path = f"{self.models_dir}/last.pth"
            self._save_checkpoint(last_path)
            
                
    
    
