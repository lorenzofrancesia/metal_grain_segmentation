import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict
import optuna

from data.dataset import SegmentationDataset
from utils.metrics import BinaryMetrics

class OptunaTrainer:
    
    def __init__(self, 
                 data_dir,
                 model,
                 batch_size=16,
                 normalize=False,
                 negative=True,
                 train_transform=transforms.ToTensor(),
                 optimizer=optim.Adam,
                 optimizer_params=None,
                 loss_function=nn.BCELoss(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr_scheduler=None,
                 epochs=10,
                 early_stopping=20,
                 trial=None,
                 evaluation_metric="Dice"  # Default metric to optimize
                 ):
        
        self.model = model
        self.trial = trial  # Optuna trial object
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.negative = negative
        self.train_transform = train_transform
        self.device = device
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.best_loss = float("inf")
        self.evaluation_metric = evaluation_metric
        self.best_metric_value = 0.0 if evaluation_metric != "loss" else float("inf")
        self.early_stopping_counter = 0
        
        # Set up optimizer with parameters from trial if provided
        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        
        self.loss_function = loss_function
        self.lr_scheduler = lr_scheduler
        
        # Initialize model on device
        self.model.to(self.device)
        
        # Get dataloaders
        self._get_dataloaders()
        
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
        self.optimizer.step()
        
        # Memory management
        del inputs, targets, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                
                if isinstance(self.loss_function, list):
                    loss_func1, loss_func2, weight1, weight2 = self.loss_function
                    loss = weight1 * loss_func1(outputs, targets) + weight2 * loss_func2(outputs, targets)
                else:
                    loss = self.loss_function(outputs, targets)  
                val_loss += loss.item()
                
                all_outputs.append(outputs_probs.detach())
                all_targets.append(targets.detach())
                
                # Memory management
                del inputs, targets, outputs, outputs_probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Aggregate predictions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics_results = defaultdict()
        binary_metrics = BinaryMetrics(device=self.device)

        # Calculate metrics at 0.5 threshold
        results_05 = binary_metrics.calculate_metrics(all_outputs, all_targets, threshold=0.5)
        metrics_results.update(results_05)
        
        val_loss /= len(self.val_loader)
        
        # Memory management
        del all_outputs, all_targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return val_loss, metrics_results
    
    def train(self):
        for epoch in range(self.epochs):
            train_loss = 0
            
            # Training loop
            for batch in self.train_loader:
                loss = self._train_step(batch)
                train_loss += loss
            
            train_loss /= len(self.train_loader)
            
            # Validation
            val_loss, metrics = self._validate()
            
            # Report to Optuna
            if self.trial is not None:
                self.trial.report(val_loss if self.evaluation_metric == "loss" else metrics[self.evaluation_metric], epoch)
                
                # Check for pruning
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            # Early stopping based on evaluation metric
            if self.evaluation_metric == "loss":
                current_metric = val_loss
                is_better = current_metric < self.best_metric_value
            else:
                current_metric = metrics[self.evaluation_metric]
                is_better = current_metric > self.best_metric_value
            
            if is_better:
                self.best_metric_value = current_metric
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Update learning rate with scheduler if provided
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        return self.best_metric_value