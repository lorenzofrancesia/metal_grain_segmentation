import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score
from PIL import Image
import numpy as np
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Unless I write my own data loader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from data.dataset import SegmentationDataset, masked_image, image_for_plot
from utils.metrics import BinaryMetrics

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
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            counter = 1
            while True:
                new_dir = f"{self.output_dir}{counter}"
                if not os.path.exists(new_dir):
                    self.output_dir = new_dir
                    os.makedirs(self.output_dir)
                    break
                counter += 1 
        
        self.model.to(self.device)
        self._load_model()  
        
    def _load_model(self):
        if not self.model_path:
            raise ValueError("Provide path to the checkpoint")
        
        # Define the device mapping
        if torch.cuda.is_available():  # Check for CUDA availability
            map_location = None  # Default: loads to current device (usually the first GPU)
        else: 
            map_location = torch.device('cpu') # Map to CPU if no CUDA
            
        print("Loading model...")
        checkpoint = torch.load(self.model_path, map_location=map_location, weights_only=True)
        print("Loading state dictionary...")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        try:
            self.mean = checkpoint['dataset_mean']
            self.std = checkpoint['dataset_std']
        except:
            self.mean = None
            self.std = None
            pass
        
            
    def _get_dataloader(self):
        self.test_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "images"),
            mask_dir=os.path.join(self.data_dir, "masks"), 
            image_transform=self.test_transform,
            mask_transform=self.test_transform,
            normalize=self.normalize,
            mean=self.mean,
            std=self.std
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
            progress_bar = tqdm(self.test_loader, desc=f'Testing', leave=False, file=sys.stdout)
            for batch in progress_bar:
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
        
        # Use BinaryMetrics class
        binary_metrics = BinaryMetrics()

        # Calculate metrics at 0.5 threshold
        results_05 = binary_metrics.calculate_metrics(all_outputs_cat, all_targets_cat, threshold=0.5)
        for metric_name, value in results_05.items():
            metrics_results[metric_name] = value

        # Calculate mIoU
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        metrics_results["miou"] = 0
        for threshold in thresholds:
            results_thresh = binary_metrics.calculate_metrics(all_outputs_cat, all_targets_cat, threshold=threshold)
            metrics_results["miou"] += results_thresh["IoU"]  # Access IoU from BinaryMetrics
        metrics_results["miou"] /= len(thresholds)

        # Calculate mAP
        metrics_results["mAP"] = average_precision_score(all_targets_cat.numpy().flatten(), all_outputs_cat.numpy().flatten())
        
        test_loss /= len(self.test_loader)
        
        results = {'loss': test_loss, **metrics_results}
        # Format results to 4 decimal places
        formatted_results = {key: f"{value:.4f}" if isinstance(value, (int, float, np.float64)) else value for key, value in results.items()}

        # Print results in a readable way
        print("Test Results:")
        for key, value in formatted_results.items():
            print(f"{key.capitalize()}: {value}")
            
        results_file_path = os.path.join(self.output_dir, "test_results.txt")
        with open(results_file_path, "w") as results_file:
            results_file.write("Test Results:\n")
            for key, value in formatted_results.items():
                results_file.write(f"{key.capitalize()}: {value}\n")
                
        return results
    
    def plot_results(self, n=4):
        """
        Plots input images, predicted masks, and target masks, handling batches.
        Creates a *separate* plot for each image, and rescales the *image*
        data for visualization.
        """
        num_batches_to_plot = min(n, len(self.all_inputs))

        for batch_idx in range(num_batches_to_plot):
            image_batch = self.all_inputs[batch_idx].cpu().numpy()
            mask_batch = self.all_targets[batch_idx].cpu().numpy()
            preds_batch = self.all_outputs[batch_idx].cpu().detach().numpy()

            # Handle single images
            if image_batch.ndim == 3:
                image_batch = image_batch[np.newaxis, ...]
                mask_batch = mask_batch[np.newaxis, ...]
                preds_batch = preds_batch[np.newaxis, ...]
            if image_batch.ndim == 2:
                image_batch = image_batch[np.newaxis, np.newaxis, ...]
                mask_batch = mask_batch[np.newaxis, np.newaxis, ...]
                preds_batch = preds_batch[np.newaxis, np.newaxis, ...]

            num_images_in_batch = min(n, image_batch.shape[0])

            for i in range(num_images_in_batch):
                image = image_batch[i]
                mask = mask_batch[i]
                preds = preds_batch[i]

                # Remove singleton channel dim
                if image.ndim == 3 and image.shape[0] == 1:
                    image = image[0]
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                if preds.ndim == 3 and preds.shape[0] == 1:
                    preds = preds[0]

                # --- Rescale Image for Visualization (Option 1) ---
                image_for_plot = image.copy()  # Work on a copy!
                if image_for_plot.ndim == 3 and image_for_plot.shape[0] == 3: #channel first
                    image_for_plot = np.transpose(image_for_plot, (1, 2, 0)) #convert to channel last for a uniform process

                # Rescale to [0, 1]
                image_for_plot = (image_for_plot - image_for_plot.min()) / (image_for_plot.max() - image_for_plot.min())

                # Create a new figure for each image
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                # Plot Image
                axes[0].imshow(image_for_plot)  # Use the rescaled image
                axes[0].set_title("Image")
                axes[0].axis("off")

                # Plot Predictions
                axes[1].imshow((preds>0.5), cmap="gray")
                axes[1].set_title("Output")
                axes[1].axis("off")

                # Plot Mask
                axes[2].imshow(mask, cmap="gray")
                axes[2].set_title("Mask")
                axes[2].axis("off")

                plt.tight_layout()
                filename = f"predictions_{batch_idx}_{i}.png"
                plt.savefig(os.path.join(self.output_dir, filename))
                plt.close(fig)
                
    def save_predictions(self, n=4):
        """
        Plots input images, predicted masks, and target masks, handling batches.
        Creates a *separate* plot for each image, and rescales the *image*
        data for visualization.
        """
        num_batches_to_plot = min(n, len(self.all_inputs))

        for batch_idx in range(num_batches_to_plot):
            image_batch = self.all_inputs[batch_idx].cpu().numpy()
            mask_batch = self.all_targets[batch_idx].cpu().numpy()
            preds_batch = self.all_outputs[batch_idx].cpu().detach().numpy()

            # Handle single images
            if image_batch.ndim == 3:
                image_batch = image_batch[np.newaxis, ...]
                mask_batch = mask_batch[np.newaxis, ...]
                preds_batch = preds_batch[np.newaxis, ...]
            if image_batch.ndim == 2:
                image_batch = image_batch[np.newaxis, np.newaxis, ...]
                mask_batch = mask_batch[np.newaxis, np.newaxis, ...]
                preds_batch = preds_batch[np.newaxis, np.newaxis, ...]

            num_images_in_batch = min(n, image_batch.shape[0])

            for i in range(num_images_in_batch):
                preds = preds_batch[i]

                if preds.ndim == 3 and preds.shape[0] == 1:
                    preds = preds[0]

               

                # Create a new figure for each image
                fig = plt.figure(figsize=(5.12, 5.12), dpi=100)  # Set figure size to 512x512 pixels
                plt.imshow(preds, cmap="gray")
                plt.axis("off")
                plt.tight_layout(pad=0)
                filename = f"{batch_idx}_{i}.png"
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', pad_inches=0)
                plt.close(fig)