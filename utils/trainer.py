import os
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
from torch import optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

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
                 loss_function=nn.BCELoss(),
                 metrics=BinaryMetrics(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr_scheduler=None,
                 epochs=10,
                 output_dir="../runs",
                 resume=False,
                 resume_path=None,
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
        
        # Resume
        self.resume = resume
        self.resume_path = resume_path
        
        # Training parameters 
        self.epochs = epochs
        self.current_epoch = 0
        self.global_step = 0
        
        # Early stopping
        self.early_stopping = early_stopping
        self.best_loss = float("inf")
        self.early_stopping_counter = 0
        
        # Output
        self.output_dir = output_dir
        self._initialize_output_folder()
        
        self._initialize_csv()
        self.writer = SummaryWriter(log_dir=self.exp_dir)
        self.verbose = verbose

        # Initialize model and optimizer
        self._initialize_training()

    
    def _initialize_training(self):
        
        self.model.to(self.device)

        if self.resume:
            self._load_checkpoint()
    

    def _initialize_output_folder(self):
        
        exp_dir = os.path.join(self.output_dir, "exp")
        
        if self.resume:
            if "results" in os.listdir(os.path.join(self.resume_path, "../..")):
                self.exp_dir = os.path.join(self.resume_path, "../../..")
                self.log_dir = os.path.join(self.exp_dir, 'logs')
                self.models_dir = os.path.join(self.exp_dir, 'models')
                self.results_dir = os.path.join(self.exp_dir, "results")
                return
            else:
                self.exp_dir = os.path.join(self.resume_path, "..", "exp")


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
        
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        self.models_dir = os.path.join(self.exp_dir, 'models')
        self.results_dir = os.path.join(self.exp_dir, "results")
        
        for sub_dir in [self.exp_dir, self.log_dir, self.models_dir, self.results_dir]:
            os.makedirs(sub_dir, exist_ok=True)

        
    def _initialize_csv(self):
        self.results_csv = os.path.join(self.results_dir, "results.csv")
        
        if not self.resume:
            if not os.path.exists(self.results_csv):
                with open(self.results_csv, mode="w+", newline="") as file:
                    writer = csv.writer(file)
                    headers = ["Epochs", "Train Loss", "Val Loss", "Learning Rate", "Precision", "Recall", "F1", "Accuracy", "Dice", "IoU"]
                    writer.writerow(headers)       
                      
    def _log_results_to_csv(self, epoch, train_loss, val_loss, metrics):
        if self.results_csv:
            with open(self.results_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                row = [epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr']]
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
        
        if not self.resume_path:
            raise ValueError("Provide path to the checkpoint")
        checkpoint = torch.load(self.resume_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']
        if self.lr_scheduler and checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.output_dir = checkpoint['output_dir']
        if self.verbose:
            print(f'Loaded checkpoint from {self.resume_path}')
    
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
    
    def _train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
    
        # Forward pass
        outputs = self.model(inputs)[0]

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
            for batch_idx, batch in enumerate(self.val_loader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)[0]
                outputs_probs = torch.sigmoid(outputs)
                
                targets = targets.long()
                
                loss = self.loss_function(outputs_probs, targets)
                val_loss += loss.item()
                
                outputs_binary = (outputs_probs > 0.5).long()
                
                
                debugging = False 
                if debugging:
                    print("----------Inputs & Targets-------------")
                    print(f"\n Batch {batch_idx}")
                    print("inputs shape:", inputs.shape)
                    print("Targets shape:", targets.shape)
                    print("targets unique values:", torch.unique(targets))
                    print("Targets dtype:" ,targets.dtype)
                    print("Targets long shape:", targets.shape)
                    print("targets long unique values:", torch.unique(targets))
                    print("Targets long dtype:", targets.dtype)
                    
                    print("----------Outputs (Logits)-------------")               
                    print("outputs shape:", outputs.shape)
                    print("outputs unique values:", torch.unique(outputs))
                    print("outputs dtype:", outputs.dtype)
                    
                    print("----------Outputs (Probabilities)-------------")
                    print("probs shape:", outputs_probs.shape)
                    print("prnbs unique values:", torch.unique(outputs_probs))
                    print("probs dtype:", outputs_probs.dtype)
                                   
                    print("----------Outputs (Binary)-------------")
                    print("binary shape:", outputs_binary.shape)
                    print("binary unique values:", torch.unique(outputs_binary))
                    print("binary dtype:", outputs_binary.dtype)


                all_outputs.append(outputs_binary.detach())
                all_targets.append(targets.detach())
        
        # Aggregate predicitions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        if debugging:
            all_outputs_flat = all_outputs.view(-1)
            all_targets_flat = all_targets.view(-1)
            
            # Check if all values are between 0 and 1
            if not torch.all((all_outputs_flat >= 0) & (all_outputs_flat <= 1)):
                raise ValueError("All output values should be between 0 and 1.")
            if not torch.all((all_targets_flat >= 0) & (all_targets_flat <= 1)):
                raise ValueError("All target values should be between 0 and 1.")
               
        metrics_results = self.metrics.calculate_metrics(all_outputs, all_targets)
        val_loss /= len(self.val_loader)
        
        return val_loss, { **metrics_results}
    
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
            
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            
            # Validation if val_dataloader exists
            val_loss = 0
            if self.val_loader:
                val_loss, metrics = self._validate()
                
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                for name, result in metrics.items():
                    self.writer.add_scalar(f"Metrics/{name}", result, epoch)

                if self.verbose:
                    print(f'\nEpoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - P: {metrics["Precision"]:.4f} - R: {metrics["Recall"]:.4f} - Acc: {metrics["Accuracy"]:.4f} - F1: {metrics["F1"]:.4f} - IoU: {metrics["IoU"]:.4f} - Dice: {metrics["Dice"]:.4f}')
                self._log_results_to_csv(epoch+1, train_loss, val_loss, metrics)
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
                
            self.image_evolution()
        
        if self.best_loss != val_loss: 
            last_path = f"{self.models_dir}/last.pth"
            self._save_checkpoint(last_path)
        
        self.writer.flush() 
        self.writer.close()   
        self.loss_plots()
        
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
        image_evolution_dir = os.path.join(self.results_dir, "image_evolution")
        if not os.path.exists(image_evolution_dir):
            os.makedirs(image_evolution_dir)

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

                filename = f"{self.current_epoch}.png"
                filepath = os.path.join(image_evolution_dir, filename)
                plt.savefig(filepath)
                plt.close()

        except Exception as e:
            print(f"Error during image evolution: {e}")
                
            
                
        
                
                    
        
        
