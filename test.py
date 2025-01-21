from torch import optim
import os

from utils.metrics import BinaryMetrics
from loss.tversky import TverskyLoss
from utils.output import initialize_test_output
from utils.input import get_args_test
from utils.trainer import Trainer            
    
    
def main():
    
    args = get_args_test()
    
    if args.output_dir:
        output_dir, _ = initialize_test_output(args.output_dir)
    else:
        output_dir, _ = initialize_test_output(os.path.split(args.model_path)[0])

    
    trainer = Trainer(
        data_dir=args.data_dir,
        output_dir=output_dir,
        resume_from_checkpoint=True,
        checkpoint_path=args.model_path
    )
    
    trainer.test()
    
if __name__ == "__main__":
    main()
    
    
#python test.py --data_dir ../data --model_path ../output_1/models/best.pth --output_dir ../output