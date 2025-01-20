import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, image_transform=transforms.ToTensor(), mask_transform=transforms.ToTensor(), normalize=False):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mean = None
        self.std = None
        self.normalize = normalize
        
        self._validate_dataset()
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
            
        if self.normalize and self.mean is not None and self.std is not None:
            normalize_transform = transforms.Normalize(mean=self.mean, std=self.std)
            image = normalize_transform(image)
            
        return image, mask
    
    def _validate_dataset(self):
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("The number of images and masks do not match.")

    
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
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return image


def class_distribution(dataset, classes_names=['Background', 'Foreground']):
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








if __name__ == '__main__':
    
    path = 'C:/Users/lorenzo.francesia/OneDrive - Swerim/Documents/Project/data/train'
    image_dir = os.path.join(path, "images")
    mask_dir = os.path.join(path, "masks")
    
    # dataset_norm = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, normalize=True)
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)
    
    # img, _ = dataset[1]
    # img_norm, _ = dataset_norm[1]
    
    # img = image_for_plot(img)
    # img_norm = image_for_plot(img_norm)
    
    # # Plot the images side by side
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(img)
    # axes[0].set_title("Original Image")
    # axes[0].axis("off")
    # axes[1].imshow(img_norm)
    # axes[1].set_title("Normalized Image")
    # axes[1].axis("off")

    # plt.tight_layout()
    # plt.show()

    class_distribution(dataset)
    visualize_overlay(dataset)
    