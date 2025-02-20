import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
import random
import math
from skimage.morphology import skeletonize, dilation
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- 1. MCANet Module (Improved Channel Attention) ---
class SqueezeExcitation(nn.Module):
    def __init__(self, nb_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.nb_channels=nb_channels
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
                nn.Linear(nb_channels, nb_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(nb_channels // reduction, nb_channels),
                nn.Sigmoid())

        
    def forward(self, x):
        y = self.avg_pool(x).view(-1,self.nb_channels)
        y = self.fc(y).view(-1,self.nb_channels,1,1)
        return x * y
    
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
        
        # self.mcanet1 = SqueezeExcitation(64)
        # self.mcanet2 = SqueezeExcitation(128)
        # self.mcanet3 = SqueezeExcitation(256)
        # self.mcanet4 = SqueezeExcitation(512)

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
       
# --- 3. Discriminator (PatchGAN) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns a layer block for the discriminator."""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),  # x2 because of concatenation
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# --- 4. Loss Function ---
def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    inputs_rescaled = (inputs + 1) / 2.0
    targets_rescaled = (targets + 1) / 2.0
    bce_loss = F.binary_cross_entropy(inputs_rescaled, targets_rescaled, reduction='none')
    pt = torch.exp(-bce_loss)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    f_loss = alpha_factor * (1 - pt)**gamma * bce_loss
    return f_loss.mean()

def gan_loss(output, is_real, smooth_factor=0.1):
    target = torch.ones_like(output) * (1 - smooth_factor) if is_real else torch.zeros_like(output)
    return F.binary_cross_entropy_with_logits(output, target)


# --- 5. Dataset and Data Loading ---
class GrainBoundaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(os.path.join(root_dir, "inputs")) if f.endswith(('.png', '.jpg', '.jpeg'))] # Add other extensions

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "inputs", self.image_files[idx])
        image = Image.open(img_path).convert('L')

        mask_path = os.path.join(self.root_dir, "targets", self.image_files[idx])
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask # white background black boundaries


def calculate_metrics(predictions, targets):
    """Calculates MIoU, Accuracy, and Precision.

    Args:
        predictions: (torch.Tensor) Predicted segmentation masks (logits).
        targets: (torch.Tensor) Ground truth segmentation masks.

    Returns:
        A tuple containing (MIoU, Accuracy, Precision).
    """
    predictions_rescaled = (predictions + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
    
    # Apply sigmoid and threshold to get binary predictions
    predictions = (predictions_rescaled > 0.5).float()
    targets = (targets > 0.5).float()  # Ensure targets are also binary

    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add a small epsilon to avoid division by zero

    total_pixels = targets.numel()  # Total number of pixels
    correct_pixels = (predictions == targets).sum().item()
    accuracy = correct_pixels / total_pixels

    true_positives = (predictions * targets).sum()
    predicted_positives = predictions.sum()
    precision = (true_positives + 1e-6) / (predicted_positives + 1e-6)

    return iou.item(), accuracy, precision  # Convert to Python numbers



def validate(generator, val_loader, device, epoch=0):
    generator.eval()
    total_l1_loss = 0
    total_miou = 0
    total_accuracy = 0
    total_precision = 0
    total_samples = 0

    with torch.no_grad():
        for input_imgs, target_imgs in val_loader:  # Ignore filename
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            generated_imgs = generator(input_imgs)
            l1_loss = F.l1_loss(generated_imgs, target_imgs)
            
            miou, accuracy, precision = calculate_metrics(generated_imgs, target_imgs)

            total_l1_loss += l1_loss.item() * input_imgs.size(0)
            total_miou += miou * input_imgs.size(0)
            total_accuracy += accuracy * input_imgs.size(0)
            total_precision += precision * input_imgs.size(0)
            total_samples += input_imgs.size(0)

    avg_l1_loss = total_l1_loss / total_samples
    avg_miou = total_miou / total_samples
    avg_accuracy = total_accuracy / total_samples
    avg_precision = total_precision / total_samples

    
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    axes[0].imshow(input_imgs.squeeze(0).squeeze(0).cpu(), cmap='gray')
    axes[0].set_title("input Image")
    axes[0].axis('off')

    axes[1].imshow(generated_imgs.squeeze(0).squeeze(0).cpu(), cmap='gray')
    axes[1].set_title("generated Image")
    axes[1].axis('off')
    
    axes[2].imshow((generated_imgs.squeeze(0).squeeze(0).cpu()>0.5), cmap='gray')
    axes[2].set_title("binary generated Image")
    axes[2].axis('off')

    axes[3].imshow(target_imgs.squeeze(0).squeeze(0).cpu(), cmap='gray')
    axes[3].set_title("target")
    axes[3].axis('off')
    
    axes[4].imshow((generated_imgs.squeeze(0).squeeze(0).cpu()>0.5).long()-target_imgs.squeeze(0).squeeze(0).cpu(), cmap='gray')
    axes[4].set_title("difference between binary generated and mask")
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(f"C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\images\\test_output_{epoch+1}.png")
    plt.close()
    
    generator.train()
    return avg_l1_loss, avg_miou, avg_accuracy, avg_precision # Return all metrics

def train(generator, discriminator, train_loader, val_loader, optimizer_G, optimizer_D, device, lambda_L1, lambda_focal, num_epochs, output_dir):

    best_val_loss = float('inf')  # Initialize with a very large value

    for epoch in range(num_epochs):
        for i, (input_imgs, target_imgs) in enumerate(train_loader):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real Images
            real_output = discriminator(input_imgs, target_imgs)
            loss_D_real = gan_loss(real_output, True)

            # Fake Images
            fake_imgs = generator(input_imgs)
            fake_output = discriminator(input_imgs, fake_imgs.detach())
            loss_D_fake = gan_loss(fake_output, False)

            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            fake_output = discriminator(input_imgs, fake_imgs) #No detach
            loss_G_GAN = gan_loss(fake_output, True)

            loss_G_L1 = F.l1_loss(fake_imgs, target_imgs)
            loss_G_Focal = focal_loss(fake_imgs, target_imgs)

            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1 + loss_G_Focal * lambda_focal
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        # --- Validation ---
        val_loss, val_miou, val_accuracy, val_precision = validate(generator, val_loader, device, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, MIoU: {val_miou:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}")

        # --- Save Best Model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), os.path.join(output_dir, 'best_generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, 'best_discriminator.pth'))
            print(f"Best model saved at epoch {epoch+1}")


    # --- Save Final Model ---
    torch.save(generator.state_dict(), os.path.join(output_dir, 'final_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, 'final_discriminator.pth'))
    print("Final model saved.")
    
    
# --- Main Script ---            
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---
    in_channels = 1
    out_channels = 1
    lr = 0.0001
    batch_size = 1
    num_epochs = 20
    lambda_L1 = 100
    lambda_focal = 100
    b = 1
    gamma = 2

    # --- Data Loading ---
    train_dir = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\gan\\train"
    val_dir = 'C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\gan\\val'
    output_dir = 'C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\out'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    train_dataset = GrainBoundaryDataset(train_dir, transform=transform)
    val_dataset = GrainBoundaryDataset(val_dir, transform=transform) # Use the same dataset class

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation data

    # --- Create Models ---
    generator = UNetGenerator(in_channels, out_channels).to(device)
    discriminator = Discriminator(in_channels).to(device)

    # --- Optimizers ---
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # --- Train ---
    train(generator, discriminator, train_loader, val_loader, optimizer_G, optimizer_D, device, lambda_L1, lambda_focal, num_epochs, output_dir)