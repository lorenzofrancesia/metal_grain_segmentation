# from torch import optim

# from utils.metrics import BinaryMetrics
# from loss.tversky import TverskyLoss
# from utils.input import get_args_train, get_model, get_optimizer, get_scheduler, get_loss_function
# from utils.trainer import Trainer            
    
# import torchseg
# from torchsummary import summary
 
# def main():
    
#     args = get_args_train()

#     try:
#         dropout = args.dropout
#         if dropout == 0:
#             dropout = None
#     except AttributeError:
#         dropout = None
        
#     aux_params=dict(
#         pooling='max',            
#         classes=1,
#         dropout=dropout
#         )       

#     model = get_model(args, aux_params=aux_params)
    
#     print(summary(model,(3,512,512)))

#     optimizer = get_optimizer(args, model=model)
    
#     print(optimizer)
    
#     scheduler = get_scheduler(args, optimizer=optimizer)
    
#     print(scheduler)
    
#     loss_function = get_loss_function(args)
    
#     print(loss_function)
    
#     trainer = Trainer(model=model,
#                       data_dir=args.data_dir,
#                       batch_size=args.batch_size,
#                       optimizer=optimizer,
#                       loss_function=loss_function,
#                       metrics=BinaryMetrics(),
#                       lr_scheduler=scheduler,
#                       epochs=args.epochs,
#                       output_dir=args.output_dir,
#                       normalize=args.normalize
#                       )
    
#     trainer.train()
    
    
# if __name__ == "__main__":
#     main()
    
    
    
# python debugging.py --model Unet --dropout 0 --encoder resnet50 --optimizer Adam --lr 0.01 --loss_function IoU --data_dir ../data --output_dir ../runs --epochs 5 --batch_size 6 


import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


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
    def __init__(self, image_dir, mask_dir, image_transform=transforms.ToTensor(), mask_transform=transforms.ToTensor(), normalize=False, verbose=False):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.image_transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])#image_transform
        self.mask_transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])#mask_transform
        self.mean = None
        self.std = None
        self.normalize = normalize
        
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
        
        
val_dataset = SegmentationDataset(image_dir='C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\data\\val\\images',
                                  mask_dir='C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\data\\val\\masks',
                                  )

        
dropout = None
        
aux_params=dict(
    pooling='max',            
    classes=1,
    dropout=dropout
    )       
import torchseg
model = torchseg.Unet(encoder_name='resnet50',
                encoder_weights=False,
                aux_params=aux_params)


weights_path = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\runs\\exp15\\models\\best.pth"
checkpoint = torch.load(weights_path)
model.load_state_dict(checkpoint['model_state_dict'])


def image_evolution(val_dataset, model):
        """
        This function visualizes the output of the model during training.

        Args:
            results_dir (str): Path to the directory for saving results.
            current_epoch (int): Current epoch number.
            val_dataset (torch.utils.data.Dataset): Validation dataset.
            device (str): Device to use for computations (e.g., 'cpu' or 'cuda').
        """

        # Create directory if it doesn't exist
        try:
            # Get image and mask
            image, mask = val_dataset[0]
            image = image.to('cpu')

            # Model prediction
            model.eval()
            with torch.no_grad():
                out = model(image.unsqueeze(0))[0]
                
                

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

                plt.show()

        except Exception as e:
            print(f"Error during image evolution: {e}")
            
# image_evolution(val_dataset, model)

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
            
            
image, mask = val_dataset[0]
image = image.to('cpu')

# Model prediction
model.eval()
with torch.no_grad():
    out = model(image.unsqueeze(0))[0]
    
    out = torch.sigmoid(out)
    
    
out_np = out.cpu().detach().numpy().squeeze().squeeze()
out_binary = (out_np > 0.5)
# image_histogram(out_np)

plt.figure()
plt.imshow(out_np)
plt.title('Prediction')
plt.axis('off')
plt.colorbar()
plt.show()