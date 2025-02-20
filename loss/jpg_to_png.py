import os
from PIL import Image
import re

def convert_jpg_to_png(folder_path):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    
    # Get all .jpg files in the folder
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    if not jpg_files:
        print("No .jpg files found in the folder.")
        return
    
    # Convert each .jpg file to .png
    for jpg_file in jpg_files:
        # Open the .jpg file
        img = Image.open(os.path.join(folder_path, jpg_file))
        
        # Convert the file name to .png
        png_file = os.path.splitext(jpg_file)[0] + '.png'
        
        # Save the image as .png
        img.save(os.path.join(folder_path, png_file))
        
        print(f"Converted {jpg_file} to {png_file}")


def rename_files_remove_regex(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                new_filename = re.sub(r"\._$", "", filename)  # Regex to remove trailing ._

                if new_filename != filename:
                    new_file_path = os.path.join(folder_path, new_filename)
                    os.rename(file_path, new_file_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")
                else:
                    print(f"No change needed for '{filename}'")

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example Usage:
folder_path = "D:\\EBSD\\C950\\output" 
rename_files_remove_regex(folder_path)

# Example usage
# convert_jpg_to_png("C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\gts")