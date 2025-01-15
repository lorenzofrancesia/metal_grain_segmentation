import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from blocks import ConvBlock, UpConvBlock
    

class Encoder(nn.Module):
    def __init__(self, channels=(3,64,128,256,512), dropout=False, dropout_prob=0.5):
        super(Encoder, self).__init__()

        # Create encoder blocks based on the number of channels
        self.encoder_blocks = nn.ModuleList([ConvBlock(channels[i], channels[i+1], dropout=dropout, dropout_prob=dropout_prob)
                                             for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Empty list for intermediate results
        block_outputs = []
        
        for block in self.encoder_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
            
        return block_outputs
    
class Decoder(nn.Module):
    def __init__(self, channels=(512,256,128,64), dropout=False, dropout_prob=0.5):
        super(Decoder, self).__init__()
        self.channels = channels
        self.up_convs = nn.ModuleList([UpConvBlock(channels[i], channels[i+1])
                                       for i in range(len(channels)-1)])
        self.decoder_blocks = nn.ModuleList([ConvBlock(channels[i], channels[i+1], dropout=dropout, dropout_prob=dropout_prob)
                                             for i in range(len(channels)-1)])
        
    def forward(self, x, encoder_features):
        # Loop through number of channels
        for i in range(len(self.channels)-1):
            x = self.up_convs[i](x)
            
            encoder_features = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_features], dim=1)
            x = self.decoder_blocks[i](x)
            
        return x
    
    def crop(self, encoder_features, x):
        _, _, height, width = x.size()
        encoder_features = transforms.CenterCrop([height, width])(encoder_features)
        
        return encoder_features
    

class UNet(nn.Module):
    def __init__(self, encoder_channels=(3,64,128,256,512), decoder_channels=(512,256,128,64), class_number=1, retain_dimension=True, output_size=(256,256), dropout=False, dropout_prob=0.5):
        super(UNet, self).__init__()
        # Initialize encoder and decoder
        self.encoder = Encoder(encoder_channels, dropout=dropout, dropout_prob=dropout_prob)
        self.decoder = Decoder(decoder_channels, dropout=dropout, dropout_prob=dropout_prob)
        
        # Initialize regression head and store the class variables
        self.head = nn.Conv2d(decoder_channels[-1], class_number, kernel_size=1)
        self.retain_dimension = retain_dimension
        self.output_size = output_size
        
    def forward(self, x):
        # Features from encoder
        encoder_features = self.encoder(x)
        
        # Pass features to decoder
        decoder_features = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        
        # Pass features through head
        map = self.head(decoder_features)
        
        # Check if resizing is needed+
        if self.retain_dimension:
            map = F.interpolate(map, size=self.output_size)
            
        return map