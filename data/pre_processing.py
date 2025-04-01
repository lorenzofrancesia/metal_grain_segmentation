import os
import cv2
import random
import shutil
from sklearn.model_selection import train_test_split

def process_image_pairs(directory, output_dir,train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Processes image pairs in a directory, splits them into slices, and saves the slices.

    Args:
        directory: The path to the directory containing the image files.
    """

    if not os.path.isdir(directory):
        raise ValueError("The provided path is not a valid directory.")
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

    image_pairs = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            base_name, ext = os.path.splitext(filename)
            gb_filename = base_name + "_gb" + ext
            image_path = os.path.join(directory, filename)
            gb_image_path = os.path.join(directory, gb_filename)
            if os.path.exists(gb_image_path):
                image_pairs.append((image_path, gb_image_path))
    
    train_pairs, temp_pairs = train_test_split(image_pairs, train_size=train_ratio)
    remaining_ratio = 1.0 - train_ratio
    val_size = val_ratio / remaining_ratio
    val_pairs, test_pairs = train_test_split(temp_pairs, train_size=val_size)

    
    def process_and_save(pairs, split_name):
        slice_counter = 0

        for image_path, gb_image_path in pairs:
            try:
                image = cv2.imread(image_path)
                gb_image = cv2.imread(gb_image_path)

                if image is None or gb_image is None:
                    print(f"Warning: Could not read one or both images: {filename}, {gb_filename}. Skipping.")
                    continue

                height, width = image.shape[:2]
                mid_x = width // 2
                mid_y = height // 2
                
                # Create slices
                image_slices = [
                    image[0:mid_y, 0:mid_x],
                    image[0:mid_y, mid_x:width],
                    image[mid_y:height, 0:mid_x],
                    image[mid_y:height, mid_x:width],
                ]

                gb_image_slices = [
                    gb_image[0:mid_y, 0:mid_x],
                    gb_image[0:mid_y, mid_x:width],
                    gb_image[mid_y:height, 0:mid_x],
                    gb_image[mid_y:height, mid_x:width],
                ]
                
                # Save slices
                for i in range(4):
                    image_output_path = os.path.join(output_dir, split_name, "images", f"{slice_counter}.png")
                    mask_output_path = os.path.join(output_dir, split_name, "masks", f"{slice_counter}.png")
                    cv2.imwrite(image_output_path, image_slices[i])
                    cv2.imwrite(mask_output_path, gb_image_slices[i])
                    slice_counter += 1

            except Exception as e:
                print(f"Error processing pair ({image_path}, {gb_image_path}): {e}")
                
    process_and_save(train_pairs, "train")
    process_and_save(val_pairs, "val")
    process_and_save(test_pairs, "test")

    print(f"Image processing and splitting complete. Slices saved in train/val/test subdirectories within {output_dir}.")
    


def add_training_data(image_dir, output_dir):
    """
    Processes image pairs in a directory, splits them into slices, and saves the slices.

    Args:
        directory: The path to the directory containing the image files.
    """

    if not os.path.isdir(image_dir):
        raise ValueError("The provided path is not a valid directory.")


    images_output_dir = os.path.join(output_dir, "images")
    masks_output_dir = os.path.join(output_dir, "masks")

    indexes = []
    for filename in os.listdir(images_output_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            base_name, ext = os.path.splitext(filename)
            indexes.append(int(base_name))
    index = max(indexes)

    image_pairs = []
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            base_name, ext = os.path.splitext(filename)
            gb_filename = base_name + "_gb" + ext
            image_path = os.path.join(image_dir, filename)
            gb_image_path = os.path.join(image_dir, gb_filename)
            if os.path.exists(gb_image_path):
                image_pairs.append((image_path, gb_image_path))
    
    def process_and_save(pairs, index):
        slice_counter = index + 1
        for image_path, gb_image_path in pairs:
            try:
                image = cv2.imread(image_path)
                gb_image = cv2.imread(gb_image_path)

                if image is None or gb_image is None:
                    print(f"Warning: Could not read one or both images: {filename}, {gb_filename}. Skipping.")
                    continue

                height, width = image.shape[:2]
                mid_x = width // 2
                mid_y = height // 2
                
                # Create slices
                image_slices = [
                    image[0:mid_y, 0:mid_x],
                    image[0:mid_y, mid_x:width],
                    image[mid_y:height, 0:mid_x],
                    image[mid_y:height, mid_x:width],
                ]

                gb_image_slices = [
                    gb_image[0:mid_y, 0:mid_x],
                    gb_image[0:mid_y, mid_x:width],
                    gb_image[mid_y:height, 0:mid_x],
                    gb_image[mid_y:height, mid_x:width],
                ]
                
                # Save slices
                for i in range(4):
                    image_output_path = os.path.join(images_output_dir, f"{slice_counter}.png")
                    mask_output_path = os.path.join(masks_output_dir, f"{slice_counter}.png")
                    cv2.imwrite(image_output_path, image_slices[i])
                    cv2.imwrite(mask_output_path, gb_image_slices[i])
                    slice_counter += 1

            except Exception as e:
                print(f"Error processing pair ({image_path}, {gb_image_path}): {e}")
                
    process_and_save(image_pairs, index)
                

if __name__ == '__main__':

    # input_dir = r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Desktop\align_test\results"
    # output_dir = r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Desktop\electrical_steel_dataset"
    
    # process_image_pairs(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    
    add_training_data(image_dir="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\alignment\\align_data\\new2",
                      output_dir=r'C:\Users\lorenzo.francesia\OneDrive - Swerim\Documents\Project\datasets\electrical_steel_dataset_plus\train')
    
    

