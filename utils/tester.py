import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
import torchvision.transforms as transforms

from utils.metrics import BinaryMetrics
from data.dataset import SegmentationDataset, masked_image, image_for_plot

class Tester():
    
    def __init__(self, 
                 data_dir,
                 model=None,
                 model_path=None,
                 normalize=False,
                 test_transform=transforms.ToTensor(),
                 loss_function=nn.BCELoss(),
                 metrics=BinaryMetrics(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir=None,
                 batch_size=8
                 ):
        
        self.model = model
        self.model_path = model_path
        self.metrics = metrics
        self.device = device
        self.loss_function = loss_function
        
        # Dataset
        self.data_dir = data_dir
        self.test_transform = test_transform
        self.normalize = normalize
        self.batch_size = batch_size
        
        # Output
        self.output_dir = os.path.join(self.model_path, "../..", "test_results")
        
        # Initialize model and optimizer
        self._initialize()
    
    def _initialize(self):
        self.model.to(self.device)
        self._load_model()
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        
    def _load_model(self):
        if not self.model_path:
            raise ValueError("Provide path to the checkpoint")
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
            
    def _get_dataloader(self):
        self.test_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "test/images"),
            mask_dir=os.path.join(self.data_dir, "test/masks"), 
            image_transform=self.test_transform,
            mask_transform=self.test_transform,
            normalize=self.normalize,
            )
        
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

                
    def test(self):
        
        self._get_dataloader()
        
        self.model.eval()
        test_loss = 0
        self.all_inputs = []
        self.all_outputs = []
        self.all_targets = []    
        
        with torch.inference_mode():
            for batch in tqdm(self.test_loader, desc="Testing", leave=False):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)[0]
                outputs_probs = torch.sigmoid(outputs)
                
                targets = targets.long()
                
                loss = self.loss_function(outputs_probs, targets)
                test_loss += loss.item()        
                
                outputs_binary = (outputs_probs > 0.5).long()
                
                self.all_inputs.append(inputs.clone().detach())
                self.all_outputs.append(outputs_binary.clone().detach())
                self.all_targets.append(targets.clone().detach())
        
        # Aggregate predicitions and targets
        all_outputs_cat = torch.cat(self.all_outputs, dim=0)
        all_targets_cat = torch.cat(self.all_targets, dim=0)
                
        metrics_results = self.metrics.calculate_metrics(all_outputs_cat, all_targets_cat)
        test_loss /= len(self.test_loader)
        
        results = {'loss': test_loss, **metrics_results}
        print(f"Test Results: {results}")
        
        return results
    
    def plot_results(self, n=4):
        
        fig = plt.figure(figsize=(8,6))
        
        for i in range(n):
            image = self.all_inputs[0][i]
            mask = self.all_targets[0][i]
            preds = self.all_outputs[0][i]
            
            ax = fig.add_subplot(n, 3, 3*i+1)
            ax.imshow(image_for_plot(image.cpu()))
            ax.set_title("Image")
            ax.axis("off")
            
            ax = fig.add_subplot(n, 3, 3*i+2)
            ax.imshow(preds.squeeze(0).cpu(), cmap="gray", vmin=0, vmax=1, interpolation="none")
            ax.set_title("Output")
            ax.axis("off")
            
            ax = fig.add_subplot(n, 3, 3*i+3)
            ax.imshow(mask.squeeze(0).cpu(), cmap="gray", vmin=0, vmax=1, interpolation="none")
            ax.set_title("Mask")
            ax.axis("off")
        
        plt.savefig(os.path.join(self.output_dir, "predictions"))

            
            
            
            
            
            
            
            
        