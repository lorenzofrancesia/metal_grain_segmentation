import argparse
import logging 
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
from torch import optim

from utils.metrics import BinaryMetrics
from data.dataset import SegmentationDataset, SegmentationTransform
from loss.tversky import TverskyLoss

#Models
from models.unet import UNet
from models.u2net import U2Net
from models.unetpp import UNetPP

def get_args():
    parser = argparse.ArgumentParser(description='Train a U-Net model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model', type=str, default='unet', help='Model to train')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model checkpoints')
    parser.add_argument('--log_file', type=str, default='train.log', help='File to save training logs')
    
    return parser.parse_args()


def get_model(args):
    if args.model == 'unet':
        model = UNet()
    elif args.model == 'u2net':
        model = U2Net()
    elif args.model == 'unet++':
        model = UNetPP()
    else:
        raise ValueError('Model type not recognized')
        
    return model

def get_dataloaders(data_dir, batch_size):
    
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "train/images"),
        mask_dir=os.path.join(data_dir, "train/masks")
        )
    
    val_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "val/images"),
        mask_dir=os.path.join(data_dir, "val/masks")
        )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader



class Trainer():
    
    def __init__(self, 
                 model, 
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 optimizer=optim.Adam, 
                 loss_function=nn.BCEWithLogitsLoss(),
                 metrics=BinaryMetrics(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr_scheduler=None,
                 epochs=10,
                 checkpoint_dir=None,
                 resume_from_checkpoint=False,
                 early_stopping=10,
                 verbose=True
                 ):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.device = device
        self.lr_scheduler = lr_scheduler
        
        # Training parameters 
        self.epochs = epochs
        self.current_epoch = 0
        self.global_step = 0
        
        # Early stopping
        self.early_stopping = early_stopping
        self.best_loss = float("inf")
        self.early_stopping_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.verbose = verbose
        
        # Initialize model and optimizer
        self._initialize_training()
        
    def _initialize_training(self):
        self.model.to(self.device)
        
        if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        if self.resume_from_checkpoint:
            self._load_checkpoint()
        
        #if self.verbose:
        #    print(self.model)
        #    print(self.optimizer)

    def _save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None
        }
        torch.save(checkpoint, checkpoint_path)
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
                
                #print(f"Inputs shape:{inputs.shape}, targets shape: {targets.shape}")
                
                outputs = self.model(inputs)
                
                #print(f"Outputs shape:{outputs.shape}, targets shape: {targets.shape}")
                
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
                    print(f'\nEpoch {epoch+1}/{self.epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}')
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
            if self.checkpoint_dir and self.epochs%5==0:
                checkpoint_path = f"{self.checkpoint_dir}/epoch_{epoch+1}.pth"
                self._save_checkpoint(checkpoint_path)
                    
            # Update lr with scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

                
    def test(self):
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
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)
    model = UNet()
    loss_function = TverskyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      metrics=BinaryMetrics(),
                      lr_scheduler=scheduler,
                      epochs=10,
                      checkpoint_dir=args.output_dir,
                      )
    
    trainer.train()
    
if __name__ == "__main__":
    main()