from utils.input import get_args_train, get_model, get_optimizer, get_scheduler, get_loss_function, parse_transforms, get_warmup_scheduler
from utils.trainer import Trainer            

 
def main():
    
    args = get_args_train()

    model = get_model(args)
    
    freeze = False
    if freeze:
        for param in model.encoder.parameterws():
            param.requires_grad = False

    optimizer = get_optimizer(args, model=model)
    
    warmup = get_warmup_scheduler(args, optimizer)
    scheduler = get_scheduler(args, optimizer=optimizer, warmup=warmup)

    loss_function = get_loss_function(args)
    
    transform = parse_transforms(args.transform)
    
    trainer = Trainer(model=model,
                      data_dir=args.data_dir,
                      train_transform=transform,
                      batch_size=args.batch_size,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      lr_scheduler=scheduler,
                      warmup=args.warmup_steps,
                      epochs=args.epochs,
                      output_dir=args.output_dir,
                      normalize=args.normalize,
                      negative=args.negative,
                      augment=args.augment,
                      config=args
                      )
    
    trainer.train()
    
    
if __name__ == "__main__":
    main()
    
    
#python train.py --model Unet --encoder seresnext101_32x8d --pretrained_weights --data_dir c:\Users\lorenzo.francesia\Documents\github\data_plus --batch_size 64 --epochs 30 --lr 0.0021916468169702035 --momentum 0.9554716028979953 --weight_decay 9.7083784789214e-05 --model unet --loss_function BCE --positive_weight 3.9815239295230382 --output_dir c:\Users\lorenzo.francesia\Documents\github\runs --negative --normalize --augment --transform "['transforms.Resize((512, 512))','transforms.ToTensor()']"

#python train.py --resume --model unet --data_dir ../data  --epochs 3 --checkpoint_path ../output/models/best.pth
