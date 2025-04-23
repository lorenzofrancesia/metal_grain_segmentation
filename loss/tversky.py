import torch.nn as nn
import torch


class TverskyLoss(nn.Module):
    """
    TverskyLoss is a loss function used for image segmentation tasks, 
    particularly in cases of class imbalance. It generalizes the Dice 
    loss by introducing parameters `alpha` and `beta` to control the 
    penalty for false positives and false negatives, respectively.

    Args:
        alpha (float): Weight for false positives. Default is 0.7.
        beta (float): Weight for false negatives. Default is 0.3.
        smooth (float): A smoothing factor to avoid division by zero. Default is 1.

    Returns:
        torch.Tensor: The computed Tversky loss value.
    """
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)   
        targets = targets.long()    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        
        return 1 - Tversky
    
    
class FocalTverskyLoss(nn.Module):
    """
    FocalTverskyLoss is a variant of the Tversky loss that applies 
    an exponential focusing parameter `gamma` to the Tversky index. 
    It is particularly useful for handling severe class imbalance 
    in image segmentation tasks.

    Args:
        alpha (float): Weight for false positives. Default is 0.7.
        beta (float): Weight for false negatives. Default is 0.3.
        gamma (float): Focusing parameter to adjust the contribution 
                       of easy and hard examples. Default is 4/3.
        smooth (float): A smoothing factor to avoid division by zero. Default is 1.

    Returns:
        torch.Tensor: The computed Focal Tversky loss value.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3):
        super(FocalTverskyLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)      
        targets = targets.long() 
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        
        return (1 - Tversky)**(1/self.gamma)



