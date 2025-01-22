from torch import optim

from utils.metrics import BinaryMetrics
from loss.tversky import TverskyLoss
from utils.output import initialize_output_folder
from utils.input import get_args_train, get_model
from utils.trainer import Trainer            
    
    
def main():
    
    args = get_args_train()
    
    output_dir = initialize_output_folder(args.output_dir, args.resume)
    model = get_model(args)
    
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
                      output_dir=output_dir,
                      resume=args.resume,
                      resume_path=args.checkpoint_path,
                      normalize=args.normalize
                      )
    
    trainer.train()
    
    
if __name__ == "__main__":
    main()
    
    
#python train.py --data_dir ../data --batch_size 6 --epochs 10 --lr 0.0001 --model unet --output_dir ../output

#python train.py --resume --model unet --data_dir ../data  --epochs 3 --checkpoint_path ../output/models/best.pth
