import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchinfo import summary
       
class VGG(nn.Module):
    """
    A Convolutional Block that performs a sequence of operations:
    Convolution -> Batch Normalization -> ReLU Activation -> Dropout.
nb 
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
    def __init__(self, channels=(3,32,64,128,256,512), dropout=False, dropout_prob=0.5):
        super(Encoder, self).__init__()

        # Create encoder blocks based on the number of channels
        self.encoder_blocks = nn.ModuleList([VGG(channels[i], channels[i+1], channels[i+1], dropout=dropout, dropout_prob=dropout_prob)
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
    def __init__(self, channels=(3,32,64,128,256,512), dropout=False, dropout_prob=0.5):
        super(Decoder, self).__init__()
        
        # Create encoder blocks based on the number of channels skipping channels[0] which is the input
        self.nested_level_one = nn.ModuleList([VGG(channels[i+1]+channels[i+2], channels[i+1], channels[i+1], dropout=dropout, dropout_prob=dropout_prob)
                                             for i in range(len(channels)-2)])
        self.nested_level_two = nn.ModuleList([VGG(2*channels[i+1]+channels[i+2], channels[i+1], channels[i+1], dropout=dropout, dropout_prob=dropout_prob)
                                             for i in range(len(channels)-3)])
        self.nested_level_three = nn.ModuleList([VGG(3*channels[i+1]+channels[i+2], channels[i+1], channels[i+1], dropout=dropout, dropout_prob=dropout_prob)
                                                 for i in range(len(channels)-4)])
        self.nested_level_four = VGG(4*channels[1]+channels[2], channels[1], channels[1], dropout=dropout, dropout_prob=dropout_prob)   
        

        
    def upsample(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode='bilinear')
    
    def forward(self, encoder_features):
        # Empty list for intermediate results
        nested_one_out = []
        # for i, block in enumerate(self.nested_level_one):
        #     print(block)
        #     x = block(torch.cat([encoder_features[i], self.upsample(encoder_features[i+1], encoder_features[i])], dim=1))
        #     print(x.shape)
        #     nested_one_out.append(x)
            
        for i, block in enumerate(self.nested_level_one):
            x = block(torch.cat([encoder_features[i], self.upsample(encoder_features[i+1], encoder_features[i])], dim=1))
            nested_one_out.append(x)
            
        # Empty list for intermediate results
        nested_two_out = []
        for i, block in enumerate(self.nested_level_two):
            x = block(torch.cat([encoder_features[i], nested_one_out[i], self.upsample(nested_one_out[i+1], nested_one_out[i])], dim=1))
            nested_two_out.append(x)
        
        # Empty list for intermediate results
        nested_three_out = []
        for i, block in enumerate(self.nested_level_three):
            x = block(torch.cat([encoder_features[i], nested_one_out[i], nested_two_out[i], self.upsample(nested_two_out[i+1], nested_two_out[i])], dim=1))
            nested_three_out.append(x)

        nested_four_out = self.nested_level_four(torch.cat([encoder_features[0], nested_one_out[0], nested_two_out[0], nested_three_out[0], self.upsample(nested_three_out[1], nested_three_out[0])], dim=1))
        
        return nested_four_out


class UPPNet(nn.Module):
    def __init__(self, channels=(3,32,64,128,256,512), class_number=1, dropout=False, dropout_prob=0.5):
        super(UPPNet, self).__init__()
        # Initialize encoder and decoder
        self.encoder = Encoder(channels, dropout=dropout, dropout_prob=dropout_prob)
        self.decoder = Decoder(channels, dropout=dropout, dropout_prob=dropout_prob)
        
        # Initialize regression head and store the class variables
        self.head = nn.Conv2d(channels[1], class_number, kernel_size=1)
        
    def forward(self, x):
        # Features from encoder
        encoder_features = self.encoder(x)
        
        # Pass features to decoder
        decoder_features = self.decoder(encoder_features)
        
        # Pass features through head
        map = self.head(decoder_features[-1])
            
        return map
    
    
net = UPPNet()
summary(net, input_size=(1, 3, 256, 256))