import os 
import cv2
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, dilation

def natural_sort_key(s):
    """
    Generates a key for natural sorting.  Splits a string into
    chunks of digits and non-digits, converting digits to integers.
    This allows for sorting like ['file1', 'file2', 'file10'] instead of
    ['file1', 'file10', 'file2'].
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]
    
def process_filenames(filenames, prefix=None):
    """
    Removes the prefix and sorts filenames naturally.

    Args:
        filenames: A list of filenames.

    Returns:
        A list of processed and sorted filenames.
    """
    # prefix = "Sura_different HBA_TDND C950-1-3 min "
    processed_filenames = []
    for filename in filenames:
        # Remove the prefix
        if prefix is not None:
            processed_filename = filename.replace(prefix, "")
            processed_filenames.append(processed_filename)
        else: 
            processed_filenames.append(filename)

    # Sort naturally
    processed_filenames.sort(key=natural_sort_key)
    return processed_filenames

data_dir = "D:\\EBSD\\C950\\output"
filenames = os.listdir(data_dir)

sorted_filenames = process_filenames(filenames)
    
img_path = os.path.join(data_dir, sorted_filenames[2])
image = Image.open(img_path).convert('L')
image = np.array(image)
image = (image < 255).astype(np.uint8)
image = dilation(skeletonize(image))

min_x = np.min(np.where(image == 1)[1])+1
max_x = np.max(np.where(image == 1)[1])-1
min_y = np.min(np.where(image == 1)[0])+1
max_y = np.max(np.where(image == 1)[0])-1

image = image[min_y:max_y, min_x:max_x]

print(image.shape)



fig = plt.figure(figsize=(15, 5))
plt.imshow(image, cmap='gray')
plt.title("input Image")
plt.axis('off')
plt.show()