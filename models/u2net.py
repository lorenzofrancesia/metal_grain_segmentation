import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import RSU, RSU4F
from utils.initializion import xavier_initializer
    
   
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
        xavier_initializer(self.lastconv)
        for conv in self.convs:
            xavier_initializer(conv)
    
    
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