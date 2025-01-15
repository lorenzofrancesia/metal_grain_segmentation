import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        
        # Comment out if sigmoid already in model
        inputs = F.sigmoid(inputs)
        
        inputs = inputs.flatten()
        targets = targets.flatten()
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        IoU = (intersection + smooth)/(union + smooth)
        
        return 1- IoU