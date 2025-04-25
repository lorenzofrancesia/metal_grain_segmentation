import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as alb

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_resize_dims_from_transform(transform_compose):
    if not isinstance(transform_compose, transforms.Compose):
        return None 
    for t in transform_compose.transforms:
        if isinstance(t, transforms.Resize):
            size = t.size
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return int(size[0]), int(size[1])
            elif isinstance(size, int):
                return (size, size)
    return None 


class SegmentationDataset(Dataset):
    """
    A dataset class for metal grain segmentation.

    Args:
        image_dir (str): Directory containing the images.
        mask_dir (str): Directory containing the masks.
        image_transform (callable, optional): A function/transform to apply to the images. Default is transforms.ToTensor().
        mask_transform (callable, optional): A function/transform to apply to the masks. Default is transforms.ToTensor().
        normalize (bool, optional): Whether to normalize the images. Default is False.
        augment (bool, optional): Whether to apply data augmentation. Default is False.
        verbose (bool, optional): Whether to print detailed information about the dataset. Default is False.
        mean (list or None, optional): Mean values for normalization. If None, they will be calculated. Default is None.
        std (list or None, optional): Standard deviation values for normalization. If None, they will be calculated. Default is None.
        negative (bool, optional): Whether to invert the mask values (1 -> 0, 0 -> 1). Default is False.
        threshold (float, optional): Threshold for converting masks to binary. Default is 0.5.
    """
    def __init__(self, 
                 image_dir, 
                 mask_dir, 
                 image_transform=transforms.ToTensor(), 
                 mask_transform=transforms.ToTensor(),
                 normalize=False, 
                 augment=False,
                 verbose=False, 
                 mean=None, 
                 std=None, 
                 negative=False,
                 color_space="LAB", 
                 threshold=0.5):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.negative = negative
        self.verbose = verbose
        self.threshold = threshold
        self.augment = augment
        self.color_space = color_space
        
        extracted_dims = get_resize_dims_from_transform(self.image_transform)
        if extracted_dims:
            self.target_height, self.target_width = extracted_dims
        else:
            target_height, target_width = 512, 512 # Or raise an error, or make them args
            if self.verbose:
                print(f"Warning: Could not extract target dimensions from image_transform. Using default: H={target_height}, W={target_width}")
        
        self.albumentation_transform = None
        if self.augment:
            self.albumentation_transform = alb.Compose([
                alb.RandomResizedCrop(p=0.3, size=(self.target_height, self.target_width), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
                alb.HorizontalFlip(p=0.3),
                alb.VerticalFlip(p=0.3),
                alb.Rotate(p=0.3, limit=(-90, 90)),
                
                alb.OneOf([alb.RandomBrightnessContrast(p=1, brightness_limit=0.3, contrast_limit=0.3),
                            alb.HueSaturationValue(p=1, hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30)], p=0.3),
                
                alb.OneOf([alb.ElasticTransform(p=1.0, alpha=1.0, sigma=50, alpha_affine=50),
                            alb.GridDistortion(p=1.0, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR),], p=0.3),
                
                alb.OneOf([alb.GaussianBlur(p=1.0, blur_limit=(3,7)),
                            alb.GaussNoise(p=1.0, var_limit=(10.0, 50.0)),], p=0.3),
                
                alb.CoarseDropout(p=0.1, num_holes_range=(1, 8), hole_height_range=(0.03, 0.1), hole_width_range=(0.03, 0.1))
            ])
        
        self.image_paths = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.mask_paths = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.image_paths.sort()
        self.mask_paths.sort()
        
        if len(self.image_paths) == 0 or len(self.mask_paths) == 0:
            raise ValueError("No images or masks found.")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Mismatch in number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}).")
        
        if self.verbose:
            self._dataset_statistics()
        
        if self.mean is not None and self.std is not None:
            print("Mean and std obtained from different dataset.")
            # print(f"Mean:{self.mean}")
            # print(f"Std:{self.std}")
        
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
        
        if self.color_space not in ['RGB', "None", None]:
            try:
                image = Image.open(img_path).convert(self.color_space)
            except Exception as e:
                print("Color space not available. Proceeding with RGB")
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        if self.augment and self.albumentation_transform:
            try:
                augmented = self.albumentation_transform(image=image_np, mask=mask_np)
                image_np = augmented['image']
                mask_np = augmented['mask']
            except Exception as e:
                 print(f"Error during Albumentations augmentation for index {idx}: {img_path}. Error: {e}")
                 
        image = Image.fromarray(image_np) 
        mask = Image.fromarray(mask_np)
        
        try:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)
        except Exception as e:
            print(f"Error during torchvision transform for index {idx}: {img_path}. Error: {e}")
            # Handle error similar to loading errors (return None, placeholder, or raise)
            raise RuntimeError(f"Torchvision transform failed for index {idx}") from e
        
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

if __name__ == "__main__":
    
    image_dir = r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Documents\Project\datasets\test_dataset\train\images"
    mask_dir = r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Documents\Project\datasets\test_dataset\train\images"
    
    dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_transform=transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()]),
    normalize=False,
    augment=True,
    negative=True
    )


    image, mask = dataset[0]

    plt.imshow(np.array(image.permute(1,2,0)))
    plt.show()