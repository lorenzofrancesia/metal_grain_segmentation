import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        
        # Comment out if sigmoid already in model
        inputs = F.sigmoid(inputs)
        targets = targets.long()

        inputs = inputs.flatten()
        targets = targets.flatten()
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice
    

class LCDiceLoss(nn.Module):
    
    def __init__(self):
        super(LCDiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        
        # Comment out if sigmoid already in model
        inputs = F.sigmoid(inputs)

        inputs = inputs.flatten()
        targets = targets.flatten()
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return np.log(np.cosh(1 - dice))
