import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import os
import random
import math
from skimage.morphology import skeletonize, dilation
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- 1. MCANet Module (Improved Channel Attention) ---

class MCANet(nn.Module):
    def __init__(self, in_channels):  
        super(MCANet, self).__init__()
        self.in_channels = in_channels
        
        # Global Context Block
        self.context_conv = nn.Conv2d(in_channels, 1, kernel_size=1)  
        
        # Transform Block
        self.k = int(abs((math.log(in_channels, 2) + 1) / 2)) 
        self.k = max(3, self.k)
        if self.k % 2 == 0:
            self.k += 1 # Make k odd
        
        self.transform = nn.Sequential(
            nn.Linear(in_channels, self.k),
            nn.ReLU(inplace=True),
            nn.Linear(self.k, in_channels),
            nn.Sigmoid()
        )                          

    def forward(self, x):
        b, c, h, w = x.size()

        context = self.context_conv(x) # (b, 1, h, w)
        context = context.view(b, 1, -1) # (b, 1, h*w)
        attention = F.softmax(context, dim=-1) # (b, 1, h*w)
        
        x_reshaped = x.view(b, c, -1) # (b, c, h*w)
        
        weighted = torch.bmm(x_reshaped, attention.transpose(1,2)).squeeze(-1) # (b, c, 1)

        # transform
        channel_weights  = self.transform(weighted) # (b, c)
        channel_weights = channel_weights.view(b, c, 1, 1) # (b, c, 1, 1)
        
        out = channel_weights * x
        
        return out     
    
# --- 2. U-Net Generator (with MCANet) ---

class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(UNetGenerator, self).__init__()

        # --- Encoder (Contraction Path) ---
        self.enc1 = self.conv_block(in_channels, 64, batchnorm=batchnorm)
        self.enc2 = self.conv_block(64, 128, batchnorm=batchnorm)
        self.enc3 = self.conv_block(128, 256, batchnorm=batchnorm)
        self.enc4 = self.conv_block(256, 512, batchnorm=batchnorm)

        self.bottleneck = self.conv_block(512, 1024, batchnorm=batchnorm)

        # --- Decoder (Expansion Path) ---  <-- Corrected blocks here
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(1024, 512, batchnorm=batchnorm)  # Combine 512 (from upconv) + 512 (from skip)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512, 256, batchnorm=batchnorm)   # Combine 256 (from upconv) + 256 (from skip)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128, batchnorm=batchnorm)   # Combine 128 (from upconv) + 128 (from skip)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(128, 64, batchnorm=batchnorm)    # Combine 64 (from upconv) + 64 (from skip)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # --- MCANet Modules (Inserted into Encoder) ---
        self.mcanet1 = MCANet(64)
        self.mcanet2 = MCANet(128)
        self.mcanet3 = MCANet(256)
        self.mcanet4 = MCANet(512)

    def conv_block(self, in_channels, out_channels, batchnorm=True):
        if batchnorm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else: 
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.mcanet1(e1)
        p1 = F.max_pool2d(e1, 2)

        e2 = self.enc2(p1)
        e2 = self.mcanet2(e2)
        p2 = F.max_pool2d(e2, 2)

        e3 = self.enc3(p2)
        e3 = self.mcanet3(e3)
        p3 = F.max_pool2d(e3, 2)

        e4 = self.enc4(p3)
        e4 = self.mcanet4(e4)
        p4 = F.max_pool2d(e4, 2)

        b = self.bottleneck(p4)

        # Decoder
        d1 = self.upconv1(b)
        d1 = torch.cat((e4, d1), dim=1)
        d1 = self.dec1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat((e3, d2), dim=1)
        d2 = self.dec2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.dec3(d3)

        d4 = self.upconv4(d3)
        d4 = torch.cat((e1, d4), dim=1)
        d4 = self.dec4(d4)

        out = self.final_conv(d4)
        return torch.tanh(out)


# --- 5. Dataset and Data Loading ---
class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))] # Add other extensions

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        gt_array = image.numpy().astype(np.float32)
        if np.mean(gt_array) > 0.5:
            binary_img = (gt_array < 0.5)
        else:
            binary_img = (gt_array > 0.5)
        skeleton = skeletonize(binary_img)

        dilated_skeleton = dilation(skeleton)
        input_image = torch.from_numpy(dilated_skeleton.astype(np.float32))
        
        return 1- input_image
    

def tensor_to_image(tensor, rescale=False):
    """Converts a PyTorch tensor to a PIL Image.

    Args:
        tensor: The input tensor (in range [-1, 1]).
        rescale: If True, rescales the tensor to [0, 1] before conversion.
    """
    if rescale:
        tensor = (tensor + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
    return transforms.ToPILImage()(tensor.squeeze(0).cpu())



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Configuration ---
    in_channels = 1
    out_channels = 1
    model_path = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\out\\best_generator.pth"  # Path to your trained generator
    test_image_dir = 'C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\preds'  # Folder with your test images AND ground truths
    output_dir = 'C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\gantestout'  # Folder to save the results

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Trained Generator ---
    generator = UNetGenerator(in_channels, out_channels).to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()  # Set to evaluation mode

    # --- Data Loading for Test Images ---
    transform = transforms.Compose([
      transforms.Resize((256,256)),
      transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)) 
    ])

    # Use the SAME dataset class, but with inference=True
    test_dataset = InferenceDataset(test_image_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # No need to shuffle

    # --- Inference Loop ---
    with torch.no_grad():  # Disable gradient calculations
      for input_img in test_loader: #Get filenames
        input_img = input_img.to(device)
        generated_img = generator(input_img)

        # Convert to PIL Images and save
        input_img = tensor_to_image(input_img)
        gen_image = tensor_to_image(generated_img)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title("Original skeletonized Image")
        axes[0].axis('off')

        axes[1].imshow(gen_image, cmap='gray')
        axes[1].set_title("Generated Image")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    print("Inference complete.")