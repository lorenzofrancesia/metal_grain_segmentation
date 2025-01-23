from torch import optim

from utils.metrics import BinaryMetrics
from loss.tversky import TverskyLoss
from utils.input import get_args_train, get_model
from utils.trainer import Trainer            
    
import torchseg
 
def main():
    
    args = get_args_train()
    
    aux_params=dict(
    pooling='max',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    classes=1,                 # define number of output labels
    )
    
    model = torchseg.UnetPlusPlus(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  
        aux_params=aux_params           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        )
    
    loss_function = TverskyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    trainer = Trainer(model=model,
                      data_dir=args.data_dir,
                      batch_size=args.batch_size,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      metrics=BinaryMetrics(),
                      lr_scheduler=scheduler,
                      epochs=args.epochs,
                      output_dir=args.output_dir,
                      resume=args.resume,
                      resume_path=args.checkpoint_path,
                      normalize=args.normalize
                      )
    
    trainer.train()
    
    
if __name__ == "__main__":
    main()
    
    
#python train.py --data_dir ../data --batch_size 6 --epochs 10 --lr 0.0001 --model unet --output_dir ../output

#python train.py --resume --model unet --data_dir ../data  --epochs 3 --checkpoint_path ../output/models/best.pth
