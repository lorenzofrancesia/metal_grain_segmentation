import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    FocalLoss is a loss function designed to address class imbalance 
    in binary or multi-class classification tasks. It applies a modulating 
    factor to the standard binary cross-entropy loss, focusing more on 
    hard-to-classify examples.

    Args:
        alpha (float): Weighting factor for the class imbalance. Default is 0.8.
        gamma (float): Focusing parameter to reduce the relative loss for 
                       well-classified examples. Default is 2.

    Returns:
        torch.Tensor: The computed focal loss value.
    """
    def __init__(self,  alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)     
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss