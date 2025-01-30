from torch import optim
from torchvision.transforms import transforms

from utils.metrics import BinaryMetrics
from loss.tversky import TverskyLoss
from utils.input import get_args_train, get_model, get_optimizer, get_scheduler, get_loss_function, parse_transforms
from utils.trainer import Trainer            
    
import torchseg


 
def main():
    
    args = get_args_train()

    try:
        dropout = args.dropout
        if dropout == 0:
            dropout = None
    except AttributeError:
        dropout = None
        
    aux_params=dict(
        pooling=args.pooling,            
        classes=1,
        dropout=dropout
        )       

    model = get_model(args, aux_params=aux_params)

    optimizer = get_optimizer(args, model=model)
    scheduler = get_scheduler(args, optimizer=optimizer)
    loss_function = get_loss_function(args)
    
    transform = parse_transforms(args.transform)
    
    trainer = Trainer(model=model,
                      data_dir=args.data_dir,
                      train_transform=transform,
                      batch_size=args.batch_size,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      lr_scheduler=scheduler,
                      epochs=args.epochs,
                      output_dir=args.output_dir,
                      normalize=args.normalize,
                      config=args
                      )
    
    trainer.train()
    
    
if __name__ == "__main__":
    main()
    
    
#python train.py --data_dir ../data --batch_size 6 --epochs 10 --lr 0.0001 --model unet --output_dir ../output

#python train.py --resume --model unet --data_dir ../data  --epochs 3 --checkpoint_path ../output/models/best.pth
