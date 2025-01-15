import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.initializion import xavier_initializer


# Maybe add pooling between each ConvBlock
class ConvBlock(nn.Module):
    """
    A Convolutional Block that performs a sequence of operations:
    Convolution -> Batch Normalization -> ReLU Activation -> Dropout.

    Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (bool, optional): Flag for dropout. Default is False.
            dropout_prob (float, optional): Probability of an element to be zeroed. Default is 0.5.
    """
        
    def __init__(self, in_channels, out_channels, dropout=False, dropout_prob=0.5):
        super(ConvBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        ]
        
        if dropout:
            layers.append(nn.Dropout2d(p=dropout_prob))       
            
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

class UpConvBlock(nn.Module):
    """
    An UpConvolutional Block that performs a sequence of operations:
    Upsample -> Convolution -> Batch Normalization -> ReLU Activation.

    Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
    """
    
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  
        )

    def forward(self, x):
        return self.block(x)
    
    
class ConvBlockU2Net(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvBlockU2Net, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Initialize weights and biases
        xavier_initializer(self.conv)
        
    def forward(self, x):
        # Apply convolution, batch normalization, and ReLU activation
        return self.relu(self.batchnorm(self.conv(x)))
    

# Residual U-block
class RSU(nn.Module):
    def __init__(self, depth, in_channels, out_channels, hidden_dim):
        super(RSU, self).__init__()
        
        # Initial convolution block
        self.conv = ConvBlockU2Net(in_channels, out_channels)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([ConvBlockU2Net(out_channels, hidden_dim)])
        for _ in range(depth-2):
            self.encoder_layers.append(ConvBlockU2Net(hidden_dim, hidden_dim))
        
        # Middle convolution block with dilation
        self.mid = ConvBlockU2Net(hidden_dim, hidden_dim, dilation=2)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([ConvBlockU2Net(2*hidden_dim, hidden_dim) for i in range(depth-2)])
        self.decoder_layers.append(ConvBlockU2Net(2*hidden_dim, out_channels))
        
        # Downsampling and upsampling layers
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, x):
        # Apply initial convolution block
        x = self.conv(x)
        
        # Encoder forward pass
        out = []
        for i, enc in enumerate(self.encoder_layers):
            if i==0: out.append(enc(x))
            else: out.append(enc(self.downsample(out[i-1])))
        
        # Middle convolution block   
        y = self.mid(out[-1])
        
        # Decoder forward pass
        for i, dec in enumerate(self.decoder_layers):
            if y>0: y = self.upsample(y)
            y = dec(torch.cat((out[len(self.decoder_layers)-i-1], y), dim=1))
        
        # Residual connection
        return x + y
    

# Residual U-block with fixed dilation
class RSU4F(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(RSU4F, self).__init__()
        
        # Initial convolution block
        self.conv = ConvBlockU2Net(in_channels, out_channels)
        
        # Encoder layers with increasing dilation
        self.encoder_layers = nn.ModuleList([
            ConvBlockU2Net(out_channels, hidden_dim),
            ConvBlockU2Net(hidden_dim, hidden_dim, dilation=2),
            ConvBlockU2Net(hidden_dim, hidden_dim, dilation=4)
            ])
        
        # Middle convolution block with larger dilation
        self.mid = ConvBlockU2Net(hidden_dim, hidden_dim, dilation=8)
        
        # Decoder layers with decreasing dilation
        self.decoder_layers = nn.ModuleList([
            ConvBlockU2Net(2*hidden_dim, hidden_dim, dilation=4),
            ConvBlockU2Net(2*hidden_dim, hidden_dim, dilation=2),
            ConvBlockU2Net(2*hidden_dim, out_channels)
            ])
        
    def forward(self, x):
        # Apply initial convolution block
        x = self.conv(x)
        
        # Encoder forward pass
        out = []
        for i, enc in enumerate(self.encoder_layers):
            if i==0: out.append(enc(x))
            else: out.append(enc(out[i-1]))
        
        # Middle convolution block    
        y = self.mid(out[-1])
        
        # Decoder forward pass
        for i, dec in enumerate(self.decoder_layers):
            y = dec(torch.cat((out[len(self.decoder_layers)-i-1], y), dim=1))
        
        # Residual connection
        return x + y