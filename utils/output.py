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
        
def initialize_output_folder(output_dir, resume):
    if resume:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir
    
    output_dir = make_unique_dir(output_dir)
    os.makedirs(output_dir)
    
    log_dir = os.path.join(output_dir, 'logs')
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, "results")
    
    for sub_dir in [log_dir, models_dir, results_dir]:
        os.makedirs(sub_dir, exist_ok=True)
    
    return output_dir

def initialize_test_output(output_dir):
    
    output_dir = make_unique_dir(output_dir)
    os.makedirs(output_dir)

    results_dir = os.path.join(output_dir, "results")
    
    for sub_dir in [results_dir]:
        os.makedirs(sub_dir, exist_ok=True)
    
    return output_dir, results_dir
    