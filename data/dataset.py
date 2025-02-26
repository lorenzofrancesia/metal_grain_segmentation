import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    """
    A dataset class for metal grain segmentation.

    Args:
        image_dir (str): Directory containing the images.
        mask_dir (str): Directory containing the masks.
        image_transform (callable, optional): A function/transform to apply to the images. Default is transforms.ToTensor().
        mask_transform (callable, optional): A function/transform to apply to the masks. Default is transforms.ToTensor().
        normalize (bool, optional): Whether to normalize the images. Default is False.
    """
    def __init__(self, image_dir, mask_dir, image_transform=transforms.ToTensor(), mask_transform=transforms.ToTensor(), normalize=False, verbose=False, mean=None, std=None, negative=False):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.negative = negative
        self.verbose = verbose
        
        if self.verbose:
            self._dataset_statistics()
        
            if self.mean is not None and self.std is not None:
                print("Mean and std obtained from different dataset.")
        
        if self.normalize and self.mean is None and self.std is None:
            self.calculate_normalization()
            
        self._validate_dataset()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = self._convert_binary(mask)
            
        if self.normalize and self.mean is not None and self.std is not None:
            image = transforms.Normalize(mean=self.mean, std=self.std)(image)
        
        if self.negative:
            mask = 1 - mask
         
        return image, mask
    
    def _validate_dataset(self):
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("The number of images and masks do not match.")

        if self.verbose:
            for img_name, mask_name in zip(self.image_paths, self.mask_paths):
                img_base = os.path.splitext(img_name)[0]
                mask_base = os.path.splitext(mask_name)[0]
                if img_base != mask_base:
                    print(f"Warning: Image and mask filenames may not match: {img_name}, {mask_name}")
            
    def _is_binary(self, mask):
        """
        Check if a mask is binary.
        """
        unique_values = torch.unique(mask)
        return torch.all((unique_values == 0) | (unique_values == 1))

    def _convert_binary(self, mask):
        """
        Convert tensor to binary
        """
        return (mask > 0.5).float()

              
    def _dataset_statistics(self):
        """
        Calculates and prints dataset statistics, including image sizes and
        optionally, pixel value statistics.  Handles potential errors gracefully.
        """
        image_sizes = []
        widths = []
        heights = []

        for img_path in self.image_paths:
            full_img_path = os.path.join(self.image_dir, img_path)
            try:
                with Image.open(full_img_path) as img:  # Use context manager
                    image_sizes.append(img.size)
                    widths.append(img.size[0])  # Directly append width
                    heights.append(img.size[1]) # Directly append height
            except (IOError, FileNotFoundError) as e:
                print(f"Warning: Could not open image {img_path}: {e}")
                #  Consider adding: continue if you want to skip the problematic image.

        print(f"Number of images: {len(self.image_paths)}")

        if widths and heights: # Check if lists are not empty before min/max
            print(f"Min width: {min(widths)}, Max width: {max(widths)}")
            print(f"Min height: {min(heights)}, Max height: {max(heights)}")
        else:
            print("No valid image sizes found.")
        
    def calculate_normalization(self):
        num_channels = 3
        pixel_sum = np.zeros(num_channels, dtype=np.float64)  # Initialize with zeros
        pixel_squared_sum = np.zeros(num_channels, dtype=np.float64)  # Initialize with zeros
        total_pixels = 0
        
        for img_path in self.image_paths:
            full_img_path = os.path.join(self.image_dir, img_path)
            try:
                img = Image.open(full_img_path).convert("RGB")  # Load directly
                img_np = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

                pixel_sum += img_np.sum(axis=(0, 1))  # Sum across height and width
                pixel_squared_sum += (img_np ** 2).sum(axis=(0, 1))
                total_pixels += img_np.shape[0] * img_np.shape[1]

            except (IOError, FileNotFoundError) as e:
                print(f"Warning: Could not open image {img_path} for normalization: {e}")
                continue  # Skip this image

        self.mean = (pixel_sum / total_pixels).tolist()  # Convert to list
        self.std = (np.sqrt(pixel_squared_sum / total_pixels - (pixel_sum / total_pixels)**2)).tolist()
        
        if self.verbose:
            print(f"Calculated Mean: {self.mean}, Std: {self.std}")
        

    
class SegmentationTransform:
    def __init__(self, resize=(128,128)):
        self.resize = transforms.Resize(resize)
        self.to_tensor = transforms.ToTensor()
        
    def __call__(self, image, mask):
        image = self.resize(image)
        mask = self.resize(mask)
        
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        
        return image, mask
    
