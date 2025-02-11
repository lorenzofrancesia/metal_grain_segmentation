from sklearn.model_selection import train_test_split
import os
import shutil

def split_dataset(path, mask_path, val_percent, test_percent, output_path):
    # Get all files in the dataset directory
    all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    all_files = [f for f in all_files if os.path.isfile(os.path.join(path, f))]
    
    # Calculate the number of samples for each split
    total_samples = len(all_files)
    test_size = int(total_samples * test_percent)/total_samples
    val_size = int(total_samples * val_percent)/total_samples
    
    # First split: train + val and test
    train_val_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)
    
    # Second split: train and val
    train_files, val_files = train_test_split(train_val_files, test_size=val_size, random_state=42)
    
    # Create output directories if they don't exist
    train_output_path = os.path.join(output_path, 'train/images')
    val_output_path = os.path.join(output_path, 'val/images')
    test_output_path = os.path.join(output_path, 'test/images')
    
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)
    
    # Create corresponding mask directories
    train_mask_output_path = os.path.join(output_path, 'train/masks')
    val_mask_output_path = os.path.join(output_path, 'val/masks')
    test_mask_output_path = os.path.join(output_path, 'test/masks')
    
    os.makedirs(train_mask_output_path, exist_ok=True)
    os.makedirs(val_mask_output_path, exist_ok=True)
    os.makedirs(test_mask_output_path, exist_ok=True)
    
    # Move files and corresponding masks to the respective directories
    for f in train_files:
        shutil.copy(os.path.join(path, f), os.path.join(train_output_path, f))
        shutil.copy(os.path.join(mask_path, f), os.path.join(train_mask_output_path, f))
    
    for f in val_files:
        shutil.copy(os.path.join(path, f), os.path.join(val_output_path, f))
        shutil.copy(os.path.join(mask_path, f), os.path.join(val_mask_output_path, f))
    
    for f in test_files:
        shutil.copy(os.path.join(path, f), os.path.join(test_output_path, f))
        shutil.copy(os.path.join(mask_path, f), os.path.join(test_mask_output_path, f))
    
    print(f"Dataset split completed. Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

import os
import re

def rename_files(folder_path):
    """
    Renames files in a folder, replacing "RGMask" with "RG" in their names.

    Args:
        folder_path: The path to the folder containing the files to rename.
    """

    try:
        for filename in os.listdir(folder_path):
            if "RGMask" in filename:  # Only process files containing "RGMask"
                new_filename = filename.replace("RGMask", "RG")  # Simple replace
                # OR, for more robust matching (e.g., case-insensitive, whole word):
                # new_filename = re.sub(r"\bRGMask\b", "RG", filename, flags=re.IGNORECASE)  # \b for word boundaries

                old_filepath = os.path.join(folder_path, filename)
                new_filepath = os.path.join(folder_path, new_filename)

                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"Renamed '{filename}' to '{new_filename}'")
                except OSError as e:
                    print(f"Error renaming '{filename}': {e}")
            #else:  # Optional: Print if no change was needed.
            #     print(f"No change needed for: {filename}")


    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# Example usage:
# rename_files("C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\data\\archive\\GRAIN DATA SET\\RGMask")

split_dataset(path="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\data\\archive\\GRAIN DATA SET\\RG",
              mask_path="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\data\\archive\\GRAIN DATA SET\\RGMask",
              test_percent=0.1,
              val_percent=0.1,
              output_path="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\data\\data")