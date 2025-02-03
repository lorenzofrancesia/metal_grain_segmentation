import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from data.dataset import SegmentationDataset, masked_image, image_for_plot

class Tester():
    
    def __init__(self, 
                 data_dir,
                 model,
                 model_path,
                 normalize=False,
                 test_transform=transforms.ToTensor(),
                 loss_function=nn.BCELoss(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size=8
                 ):
        
        self.model = model
        self.model_path = model_path
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
        
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

                
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
                
                outputs = self.model(inputs)
                outputs_probs = torch.sigmoid(outputs)
                
                targets = targets.long()
                
                if isinstance(self.loss_function, list):
                    loss_func1, loss_func2, weight1, weight2 = self.loss_function
                    loss = weight1 * loss_func1(outputs_probs, targets) + weight2 * loss_func2(outputs_probs, targets)
                else:
                    loss = self.loss_function(outputs_probs, targets)  
                test_loss += loss.item()        
                
                # outputs_binary = (outputs_probs > 0.5).long()
                
                self.all_inputs.append(inputs.clone().detach())
                self.all_outputs.append(outputs_probs.clone().detach())
                self.all_targets.append(targets.clone().detach())
        
        # Aggregate predicitions and targets
        all_outputs_cat = torch.cat(self.all_outputs, dim=0)
        all_targets_cat = torch.cat(self.all_targets, dim=0)
        
        metrics_results = defaultdict()
        
        tp, fp, fn, tn = smp.metrics.get_stats(all_outputs_cat, all_targets_cat, mode="binary", threshold=0.5)
        
        metrics_results["iou"]= smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro" ).item()
        metrics_results["dice"] = smp.metrics.f1_score(tp, tn, fn ,tn, reduction="micro").item()
        metrics_results["precision"] = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
        metrics_results["recall"]= smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item()
        metrics_results["accuracy"] = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
        metrics_results["mAP"] = average_precision_score(all_targets_cat.numpy().flatten(), all_outputs_cat.numpy().flatten())
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        metrics_results["miou"] = 0
        for threshold in thresholds:
            tp, fp, fn, tn = smp.metrics.get_stats(all_outputs_cat, all_targets_cat, mode="binary", threshold=threshold)
            metrics_results["miou"] += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro" ).item()
        metrics_results["miou"] /= len(thresholds)
        
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

            
            
            
            
            
            
            
            
        