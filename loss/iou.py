import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    """
    IoULoss (Intersection over Union Loss) is a loss function used 
    for evaluating the overlap between predicted and target binary 
    masks. It is commonly used in image segmentation tasks. The loss 
    is computed as 1 minus the IoU score, which ranges from 0 (no 
    overlap) to 1 (perfect overlap).

    Args:
        smooth (float): A smoothing factor to avoid division by zero. Default is 1.

    Returns:
        torch.Tensor: The computed IoU loss value.
    """
    
    def __init__(self):
        super(IoULoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        
        # Comment out if sigmoid already in model
        inputs = F.sigmoid(inputs)
        targets = targets.long()
        
        inputs = inputs.flatten()
        targets = targets.flatten()
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        IoU = (intersection + smooth)/(union + smooth)
        
        return 1- IoU