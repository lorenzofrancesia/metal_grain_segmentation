from torch import optim
import os

from utils.metrics import BinaryMetrics
from loss.tversky import TverskyLoss
from utils.input import get_args_test, get_model
from utils.tester import Tester   
import torchseg      
    
    
def main():
    
    args = get_args_test()
    
    
    aux_params=dict(
    pooling='max',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    classes=4,                 # define number of output labels
    )
    
    model = torchseg.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  
        aux_params=aux_params           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        )
    
    

    loss_function = TverskyLoss()
    tester = Tester(
        model=model,
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        loss_function=loss_function,
        normalize=args.normalize
    )
    
    tester.test()
    tester.plot_results()
    
if __name__ == "__main__":
    main()
    
    
#python test.py --model unet --data_dir ../data --model_path ../output_1/models/best.pth 