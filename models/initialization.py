import torch

def xavier_initializer(layer, bias_value=0):
    torch.nn.init.xavier_uniform_(layer.weight)
    
    if layer.bias is not None:
        layer.bias.data.fill_(bias_value)
        
    return layer

