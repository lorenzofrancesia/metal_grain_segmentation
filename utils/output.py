import os

def make_unique_dir(base_dir):
    
    if not os.path.exists(base_dir):
        return base_dir
    
    counter = 1
    while True:
        new_dir = f"{base_dir}_{counter}"
        if not os.path.exists(new_dir):
            return new_dir
        counter += 1
        
def initialize_output_folder(output_dir):
    
    output_dir = make_unique_dir(output_dir)
    os.makedirs(output_dir)
    
    log_dir = os.path.join(output_dir, 'logs')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    results_dir = os.path.join(output_dir, "results")
    
    for sub_dir in [log_dir, checkpoint_dir, results_dir]:
        os.makedirs(sub_dir, exist_ok=True)
    
    return output_dir, log_dir, checkpoint_dir, results_dir
    