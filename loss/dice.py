import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    DiceLoss is a loss function used for evaluating the similarity 
    between predicted and target binary masks. It is commonly used 
    in image segmentation tasks. The loss is computed as 1 minus 
    the Dice coefficient, which ranges from 0 (no overlap) to 1 
    (perfect overlap).

    Args:
        smooth (float): A smoothing factor to avoid division by zero. Default is 1.

    Returns:
        torch.Tensor: The computed Dice loss value.
    """
    
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
    """
    LCDiceLoss (Log-Cosh Dice Loss) is a variant of the Dice loss 
    function that applies the log-cosh transformation to the Dice 
    coefficient. This transformation can make the loss function 
    more robust to outliers while maintaining the properties of 
    the Dice loss. It is suitable for image segmentation tasks.

    Args:
        smooth (float): A smoothing factor to avoid division by zero. Default is 1.

    Returns:
        torch.Tensor: The computed Log-Cosh Dice loss value.
    """
    
    def __init__(self):
        super(LCDiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)

        inputs = inputs.flatten()
        targets = targets.flatten()
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return np.log(np.cosh(1 - dice))
