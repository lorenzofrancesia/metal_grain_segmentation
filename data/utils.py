import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import re

import torch
from data.dataset import SegmentationDataset


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
        idx (int, optional): Integer corresponding to image. Defaults to None.
        alpha (float, optional): Transparency of the overlay. Default is 0.5.
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


def convert_jpg_to_png(folder_path):
    """
    Converts all .jpg files in the specified folder to .png format.

    Args:
        folder_path (str): The path to the folder containing .jpg files.

    Returns:
        None
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    
    # Get all .jpg files in the folder
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    if not jpg_files:
        print("No .jpg files found in the folder.")
        return
    
    # Convert each .jpg file to .png
    for jpg_file in jpg_files:
        # Open the .jpg file
        img = Image.open(os.path.join(folder_path, jpg_file))
        
        # Convert the file name to .png
        png_file = os.path.splitext(jpg_file)[0] + '.png'
        
        # Save the image as .png
        img.save(os.path.join(folder_path, png_file))
        
        print(f"Converted {jpg_file} to {png_file}")


def rename_files_remove_regex(folder_path, regex=r"\._$"):
    """
    Renames files in the specified folder by removing trailing '._' from filenames.

    Args:
        folder_path (str): The path to the folder containing files to rename.
        regex (str): The regex pattern to match the part of the filename to remove. Default is r"\._$".

    Returns:
        None
    """
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                new_filename = re.sub(regex, "", filename)  # Regex to remove trailing ._

                if new_filename != filename:
                    new_file_path = os.path.join(folder_path, new_filename)
                    os.rename(file_path, new_file_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")
                else:
                    print(f"No change needed for '{filename}'")

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


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
    
    inspect(dataset)