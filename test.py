from utils.input import get_args_test, get_model, get_loss_function, parse_transforms
from utils.tester import Tester   

    
    
def main():
    
    args = get_args_test()
    
    model = get_model(args)
    
    loss_function = get_loss_function(args)
    
    transform = parse_transforms(args.transform)

    tester = Tester(
        model=model,
        model_path=args.test_model_path,
        data_dir=args.test_data_dir,
        batch_size=args.test_batch_size,
        loss_function=loss_function,
        normalize=args.test_normalize, 
        test_transform = transform
    )
    
    tester.test()
    tester.plot_results()
    # tester.save_predictions()
    
if __name__ == "__main__":
    main()
    
    
#python test.py --model unet --data_dir ../data --model_path ../output_1/models/best.pth 