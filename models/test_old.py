from torch import optim
import os

from utils.metrics import BinaryMetrics
from loss.tversky import TverskyLoss
from utils.input import get_args_test, get_model
from utils.tester import Tester            
    
    
def main():
    
    args = get_args_test()
    model = get_model(args)
    
    

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
    
    
#python test.py --model unet --data_dir ../data --model_path ../output_1/models/best.pth --output_dir ../output