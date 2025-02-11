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
    def __init__(self, image_dir, mask_dir, image_transform=transforms.ToTensor(), mask_transform=transforms.ToTensor(), normalize=False, verbose=False, mean=None, std=None):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mean = mean
        self.std = std
        self.normalize = normalize
        
        if self.mean is not None and self.std is not None:# and verbose:
            print("Mean and std obtained from different dataset.")
        
        self._validate_dataset()
        
        if verbose:
            self._dataset_statistics()
        
        if self.normalize:
            self.calculate_normalization()
    
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
            normalize_transform = transforms.Normalize(mean=self.mean, std=self.std)
            image = normalize_transform(image)
        
        mask = 1 - mask
         
        return image, mask
    
    def _validate_dataset(self):
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("The number of images and masks do not match.")
        
        for idx in range(len(self)):
            _, mask = self[idx]
            if not self._is_binary(mask):
                raise ValueError(f"Mask at index {idx} is not binary.")
            
    def _is_binary(self, mask):
        """
        Check if a mask is binary.

        Args:
            mask (torch.Tensor): The input mask.

        Returns:
            bool: True if the mask is binary, False otherwise.
        """
        unique_values = torch.unique(mask)
        return torch.all((unique_values == 0) | (unique_values == 1))

    def _convert_binary(self, mask):
        
        if isinstance(mask, torch.Tensor):
            binary_mask = (mask > 0.5).float()
        else:
            mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
            binary_mask = (mask_tensor > 0.5).float()
        
        return binary_mask
              
    def _dataset_statistics(self):
        image_sizes = []
        for img_path in self.image_paths:
            img = Image.open(os.path.join(self.image_dir, img_path))
            image_sizes.append(img.size)
        
        widths, heights = zip(*image_sizes)
        print(f"Number of images: {len(self.image_paths)}")
        print(f"Min width: {min(widths)}, Max width: {max(widths)}")
        print(f"Min height: {min(heights)}, Max height: {max(heights)}")
        
    def calculate_normalization(self):
        pixel_sum = None
        pixel_squared_sum = None
        total_pixels = 0
        
        for idx in range(len(self)):
            image, _ = self[idx]
            
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            
            if pixel_sum is None:
                pixel_sum = np.zeros(image.shape[0])
                pixel_squared_sum = np.zeros_like(pixel_sum)
                
            pixel_sum += image.sum(axis=(1,2))
            pixel_squared_sum += (image ** 2).sum(axis=(1,2))
            total_pixels += image.shape[1]*image.shape[2]
            
        self.mean = pixel_sum / total_pixels
        self.std = np.sqrt(pixel_squared_sum / total_pixels - self.mean**2)
        

    
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
    

def visualize_dataset(dataset, num_samples=32):
    """
    Visualize a few samples from the dataset along with their corresponding masks.

    Args:
        dataset (Dataset): The dataset to visualize.
        num_samples (int, optional): The number of samples to visualize. Default is 32.
    """
    num_samples = min(num_samples, len(dataset))
    
    cols = int(np.ceil(np.sqrt(2 * num_samples)))
    if cols % 2 != 0:
        cols += 1
    rows = int(np.ceil((2 * num_samples) / cols))
    
    fig = plt.figure(figsize=(1.5*cols,1.5*rows))
    for i in range(num_samples):
        image, mask = dataset[i]
        
        ax = fig.add_subplot(rows, cols, 2*i+1)
        ax.imshow(image_for_plot(image))
        ax.axis("off")
        ax.set_title(f"Image {i+1}")
        
        ax = fig.add_subplot(rows, cols, 2*i+2)
        ax.imshow(mask.squeeze(0), cmap='gray', vmin=0, vmax=1, interpolation='none')
        ax.axis("off")
        ax.set_title(f"Mask {i+1}")
    
    plt.tight_layout()    
    plt.show()
    
    
def image_for_plot(image):
    """
    Convert the image to a format suitable for plotting.

    Args:
        image (np.ndarray or torch.Tensor): The input image.

    Returns:
        np.ndarray: The image in a format suitable for plotting.
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return image


def class_distribution(dataset, classes_names=['Background', 'Foreground']):
    """
    Calculate and plot the class distribution of the dataset.

    Args:
        dataset (Dataset): The dataset to analyze.
        class_names (list, optional): List of class names. Default is ['Background', 'Foreground'].
    """
    total_pixels = 0
    foreground_pixels = 0

    for _, mask in dataset:
        mask = mask.permute(1, 2, 0).numpy()
        total_pixels += mask.shape[0]*mask.shape[1]
        foreground_pixels += (mask==1).sum()
    
    background_pixels = total_pixels - foreground_pixels
    
    counts = [background_pixels, foreground_pixels]
    percentages = [background_pixels/total_pixels * 100, foreground_pixels/total_pixels * 100]
    
    plt.figure(figsize=(8,6))
    bars = plt.bar(classes_names, counts, color=['blue','orange'])
    plt.ylabel('Pixel Count')
    plt.title('Class Distribution')
    
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{percentage:.2f}%",
                 ha="center", va="bottom", fontsize=10, color="black")
    plt.show()


def visualize_overlay(dataset, idx=None, alpha=0.5):
    """
    Calculate and plot the class distribution of the dataset.

    Args:
        dataset (Dataset): The dataset to analyze.
        class_names (list, optional): List of class names. Default is ['Background', 'Foreground'].
    """
    if idx is None:
        idx = np.random.randint(0, len(dataset))
        
    image, mask = dataset[idx]
    
    image = image_for_plot(image)
    
    plt.figure(figsize=(7,7))
    plt.imshow(image, alpha=1.0)
    plt.imshow(mask.squeeze(0), cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.title(f"Overlay of Image and Mask [Index: {idx}]")
    plt.show()


def image_histogram(image):
    """
    Calculate and plot the histogram of pixel values in an image.

    Args:
        image (np.ndarray or torch.Tensor): The input image.

    Returns:
        None
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Flatten the image to get the pixel values
    pixel_values = image.flatten()
    
    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.hist(pixel_values, bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pixel Values')
    plt.show()
    

def inspect(dataset):
    
    for i in range(5):
        image, mask = dataset[i]
        
        print("Image shape:", image.shape)
        print("mask shape:", mask.shape)
        print("Image Min/Max:", image.min(), image.max())
        print("Mask unique values:", torch.unique(mask))
        
        image = image.permute(1,2,0).numpy() if isinstance(image, torch.Tensor) else image
        mask = mask.squeeze().numpy() if isinstance(mask, torch.Tensor) else mask
    
    
def masked_image(image, mask):
    image = image.permute(1, 2, 0).numpy()
    image = (127.5 * (image+1)).astype(np.uint8) 
    
    mask = mask.squeeze().numpy()
    mask = np.expand_dims(mask, axis=1)
    
    masked_image = (image * mask).astype(np.uint8)
    
    return cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)




if __name__ == '__main__':
    
    path = 'C:/Users/lorenzo.francesia/OneDrive - Swerim/Documents/Project/data/train'
    image_dir = os.path.join(path, "images")
    mask_dir = os.path.join(path, "masks")
    
    # dataset_norm = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, normalize=True)
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)
    dataset_norm = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, normalize=True)
    
    img, _ = dataset[1]
    img_norm, _ = dataset_norm[1]
    
    img = image_for_plot(img)
    img_norm = image_for_plot(img_norm)
    
    # Plot the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(img_norm)
    axes[1].set_title("Normalized Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    visualize_dataset(dataset)
    class_distribution(dataset)
    visualize_overlay(dataset)
    
    randint = np.random.randint(0,len(dataset)-1)
    _, mask = dataset[randint] 
    
    print(randint)
    image_histogram(mask)
    
    # inspect(dataset)