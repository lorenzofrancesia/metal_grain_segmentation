import argparse
import os
from tqdm import tqdm
import numpy as np
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
from torch import optim
import torchvision.transforms as transforms

from utils.metrics import BinaryMetrics
from data.dataset import SegmentationDataset, SegmentationTransform
from loss.tversky import TverskyLoss
from utils.output import initialize_output_folder
from utils.input import get_args, get_model
    

class Trainer():
    
    def __init__(self, 
                 model, 
                 data_dir,
                 batch_size=16,
                 normalize=False,
                 train_transform= transforms.ToTensor(),
                 test_transform=transforms.ToTensor(),
                 optimizer=optim.Adam, 
                 loss_function=nn.BCEWithLogitsLoss(),
                 metrics=BinaryMetrics(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr_scheduler=None,
                 epochs=10,
                 output_dir=None,
                 resume_from_checkpoint=False,
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
        self.test_transform = test_transform
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
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.resume_from_checkpoint = resume_from_checkpoint
        self.verbose = verbose
        
        # Initialize model and optimizer
        self._initialize_training()
        self._initialize_csv()
        
        # Initialize dataloaders 
        self._get_dataloaders()
    
    
    def _initialize_training(self):
        self.model.to(self.device)
        
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if self.resume_from_checkpoint:
            self._load_checkpoint()
        
        #if self.verbose:
        #    print(self.model)
        #    print(self.optimizer)
        
    def _initialize_csv(self):
        self.results_csv = os.path.join(self.output_dir, "results", "results.csv")
        
        with open(self.results_csv, mode="w+", newline="") as file:
            writer = csv.writer(file)
            headers = ["Epochs", "Train Loss", "Val Loss", "Learning Rate", "Precision", "Recall", "F1", "Accucary", "Dice", "IoU"]
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
            normalize=self.normalize
            )
        
        self.val_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "val/images"),
            mask_dir=os.path.join(self.data_dir, "val/masks"),
            image_transform=self.train_transform,
            mask_transform=self.train_transform,
            normalize=self.normalize            
            )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)    

    def _save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None
        }
        #torch.save(checkpoint, checkpoint_path)
        if self.verbose:
            print(f'Checkpoint saved at {checkpoint_path}')
    
    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']
        if self.lr_scheduler and checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if self.verbose:
                print(f'Loaded checkpoint from {checkpoint_path}')
            
    def _train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
    
        # Forward pass
        outputs = self.model(inputs)        
        loss = self.loss_function(outputs, targets)
        
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
                output_probs = torch.sigmoid(outputs)
                                
                loss = self.loss_function(outputs, targets)
                val_loss += loss.item()
                
                all_outputs.append(outputs.detach())
                all_targets.append(targets.detach())
        
        # Aggregate predicitions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics_results = self.metrics.calculate_metrics(all_outputs, all_targets)
        val_loss /= len(self.val_loader)
        
        return {'val_loss': val_loss, **metrics_results}
    
    def train(self):
        
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
                
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                        
            # Save checkpoint
            if self.output_dir and self.epochs%5==0:
                checkpoint_path = f"{self.output_dir}/epoch_{epoch+1}.pth"
                self._save_checkpoint(checkpoint_path)
                    
            # Update lr with scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

                
    def test(self):
        
        self.test_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "test/images"),
            mask_dir=os.path.join(self.data_dir, "tst/masks"),
            image_transform=self.test_transform,
            mask_transform=self.test_transform,
            normalize=self.normalize            
            )
        
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        
        self.model.eval()
        test_loss = 0
        all_outputs = []
        all_targets = []    
        
        with torch.inference_mode():
            for batch in tqdm(self.test_loader, desc="Testing", leave=False):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                test_loss += loss.item()        
                
                all_outputs.append(outputs)
                all_outputs.append(targets)
        
        # Aggregate predicitions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics_results = self.metrics.calculate_metrics(all_outputs, all_targets)
        test_loss /= len(self.test_loader)
        
        results = {'loss': test_loss, **metrics_results}
        if self.verbose:
            print(f"Test Results: {results}")
        return results        
    
    
def main():
    
    args = get_args()
    
    output_dir, _, _, _ = initialize_output_folder(args.output_dir)
    model = get_model(args)
    
    loss_function = TverskyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    trainer = Trainer(model=model,
                      data_dir=args.data_dir,
                      batch_size=args.batch_size,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      metrics=BinaryMetrics(),
                      lr_scheduler=scheduler,
                      epochs=args.epochs,
                      output_dir=output_dir,
                      )
    
    trainer.train()
    
if __name__ == "__main__":
    main()