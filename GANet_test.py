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
    def __init__(self, channel, reduction=5):  # K=5 as per the paper
        super(MCANet, self).__init__()
        self.channel = channel
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1)

        self.k_size =  int(abs((math.log(channel, 2) + 1) / 2)) #Calculate K value, paper references another to determine it.
        if self.k_size % 2 == 0:
            self.k_size += 1
        self.convs = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size -1)//2, bias=False) for _ in range(channel)])#k conv layers for each channel

    def forward(self, x):
        b, c, h, w = x.size()

        # --- Context Modeling ---
        y = self.conv1x1(x)
        y = y.view(b, -1, 1)  # Reshape for Softmax (HW x 1 x 1)
        y = F.softmax(y, dim=1)
        y = y.view(b, c, h, w)  # Reshape back to original

        # --- Transform ---
        # (B, C, H, W) -> (B, C, 1, 1)  Average over H and W
        z = x.mean(dim=[2, 3], keepdim=True)
        
        z = z.view(b, 1, c) # (B, C, 1, 1) -> (B, 1, C)  Prepare for 1D convolutions

        transformed = []
        for i in range(self.channel):
            # Apply 1D convolution to each channel separately
            transformed.append(self.convs[i](z[:, :, i:i+1]))  # (B, 1, 1)

        # Concatenate along the channel dimension: (B, 1, C)
        z = torch.cat(transformed, dim=2)
        z = torch.sigmoid(z)

        # Expand for element-wise multiplication: (B, 1, C) -> (B, C, 1, 1)
        z = z.view(b, c, 1, 1)
        return x * z 
    
# --- 2. U-Net Generator (with MCANet) ---

class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()

        # --- Encoder (Contraction Path) ---
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        # --- Decoder (Expansion Path) ---
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # --- MCANet Modules (Inserted into Encoder) ---
        self.mcanet1 = MCANet(64)
        self.mcanet2 = MCANet(128)
        self.mcanet3 = MCANet(256)
        self.mcanet4 = MCANet(512)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        return out


# --- 5. Dataset and Data Loading ---
class GrainBoundaryDataset(Dataset):
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
        ground_truth = torch.from_numpy(dilated_skeleton.astype(np.float32))#.unsqueeze(0) 
        # print(f"groundtruth has shape {ground_truth.shape}")

        input_skeleton = skeleton.copy()
        input_skeleton = input_skeleton[0,:,:]
        # print(f"input has shape {input_skeleton.shape}")
        
        row_indices, col_indices = np.where(input_skeleton == True)
        coordinates = list(zip(row_indices, col_indices))

        num_erase_points = np.random.randint(9, 16)
        erosion_radius = np.random.randint(5, 10) 
        selected_points = np.random.choice(len(coordinates), min(num_erase_points, len(coordinates)), replace=False)

        for i in selected_points:
            row, col = coordinates[i]
            rr, cc = np.ogrid[:input_skeleton.shape[0], :input_skeleton.shape[1]]
            mask = (rr - row)**2 + (cc - col)**2 <= erosion_radius**2
            input_skeleton[mask] = 0

        dilated_input = dilation(input_skeleton)
        input_image = torch.from_numpy(dilated_input.astype(np.float32)).unsqueeze(0)#.unsqueeze(0)
        
        return input_image, ground_truth


def tensor_to_image(tensor):
    """Converts a PyTorch tensor to a PIL Image."""
    return transforms.ToPILImage()(tensor.squeeze(0).cpu())



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Configuration ---
    in_channels = 1
    out_channels = 1
    model_path = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\out\\final_generator.pth"  # Path to your trained generator
    test_image_dir = 'C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\val'  # Folder with your test images AND ground truths
    output_dir = 'C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\val'  # Folder to save the results

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Trained Generator ---
    generator = UNetGenerator(in_channels, out_channels).to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()  # Set to evaluation mode

    # --- Data Loading for Test Images ---
    transform = transforms.Compose([
      transforms.Resize((256,256)),
      transforms.ToTensor()
    ])

    # Use the SAME dataset class, but with inference=True
    test_dataset = GrainBoundaryDataset(test_image_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # No need to shuffle

    # --- Inference Loop ---
    with torch.no_grad():  # Disable gradient calculations
      for input_img, target_img in test_loader: #Get filenames
        # Only process if the input image is not all zeros (happens during training, not needed here)
        if torch.any(input_img != 0):
            input_img = input_img.to(device)
            target_img = target_img.to(device) # Load target to device.
            generated_img = generator(input_img)


            # --- Save Results ---
            # Convert to PIL Images and save
            input_img = tensor_to_image(input_img)
            gen_image = tensor_to_image(generated_img) # No more batch processing.
            target_image = tensor_to_image(target_img)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(input_img)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(gen_image, cmap='gray')
            axes[1].set_title("Generated Image")
            axes[1].axis('off')

            axes[2].imshow(target_image, cmap='gray')
            axes[2].set_title("target")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()
            

    print("Inference complete.")