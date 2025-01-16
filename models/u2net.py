import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import sys

parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from initialization import init_weights  # Corrected relative import
else:
    from .initialization import init_weights

  
class ConvBlockU2Net(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvBlockU2Net, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Initialize weights and biases
        init_weights(self.conv, init_type='xavier')
        
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
    
    
class VGG(nn.Module):
    """
    A Convolutional Block that performs a sequence of operations:
    Convolution -> Batch Normalization -> ReLU Activation -> Dropout.

    Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of output channels for the first convolutional block.
            out_channels (int): Number of output channels.
            dropout (bool, optional): Flag for dropout. Default is False.
            dropout_prob (float, optional): Probability of an element to be zeroed. Default is 0.5.
    """
        
    def __init__(self, in_channels, mid_channels, out_channels, dropout=False, dropout_prob=0.5):
        super(VGG, self).__init__()

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        ]
        
        if dropout:
            layers.append(nn.Dropout2d(p=dropout_prob))       
            
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)  
   
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Encoder blocks
        self.enc = nn.ModuleList([
            RSU(7, 3, 64, 32),
            RSU(6, 64, 128, 32),
            RSU(5, 128, 256, 64),
            RSU(4, 256, 512, 128),
            RSU4F(512, 512, 256),
            RSU4F(512, 512, 256)
            ])
        
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Empty list for intermediate results
        enc_out = []
        
        for i, enc in enumerate(self.enc):
            if i==0: enc_out.append(enc(x))
            else: enc_out.append(enc(self.downsample(enc_out[i-1])))
            
        return enc_out 
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Decoder blocks
        self.dec = nn.ModuleList([
            RSU4F(1024, 512, 256),
            RSU(4, 1024, 256, 128),
            RSU(5, 512, 128, 64),
            RSU(6, 256, 64, 32),
            RSU(7, 128, 32, 16)
            ])
        
    def forward(self, enc_out):
        # Decoder forward pass
        dec_out = []
        for i, dec in enumerate(self.dec):
            dec_out.append(dec(torch.cat((self.upsample(dec_out[i], enc_out[4-i]), enc_out[4-i]), dim=1)))
    
    def upsample(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode='bilinear') 
        

class U2Net(nn.Module):
    def __init__(self):
        super(U2Net, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # Side output convolution layers
        self.convs = nn.ModuleList([
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
            nn.Conv2d(512, 1, kernel_size=3, padding=1)
        ])
        
        self.lastconv = nn.Conv2d(6, 1, kernel_size=1)
        
        # Initialize weights and biases
        init_weights(self.lastconv, init_type='xavier')
        for conv in self.convs:
            init_weights(conv, init_type='xavier')
    
    
    def forward(self, x):  
        
        enc_out = self.encoder(x)
        
        dec_out = self.decoder(enc_out)
        
        # Side output convolution layers
        side_out = []
        for i, conv in enumerate(self.convs):
            if i==0: side_out.append(conv(dec_out[5]))
            else: side_out.append(self.upsample(conv(dec_out[5-i], side_out[0])))
        
        # Final convolution layer       
        side_out.append(self.lastconv(torch.cat(side_out, dim=1)))
        
        # Return side output predictions
        return [torch.sigmoid(s.squeeze(1)) for s in side_out]

