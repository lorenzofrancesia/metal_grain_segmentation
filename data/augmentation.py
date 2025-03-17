import albumentations as alb
import os
from PIL import Image
import numpy as np


def rotate(image, mask, angle, exact_angle):
    """
    Rotate the image and mask by a specified angle.

    Args:
        image (np.ndarray): The input image to be rotated.
        mask (np.ndarray): The input mask to be rotated.
        angle (int): The angle by which to rotate the image and mask.
        exact_angle (bool): Whether to use the exact angle or a random angle within the range.

    Returns:
        tuple: A tuple containing the rotated image and mask.
    """
    if exact_angle:
        angle = (angle, angle)
        
    transform = alb.Rotate(limit=angle, p=1, crop_border=True)
    
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    
    return transformed_image, transformed_mask


def flip_horizontally(image, mask):
    """
    Flip the image and mask horizontally.

    Args:
        image (np.ndarray): The input image to be flipped.
        mask (np.ndarray): The input mask to be flipped.

    Returns:
        tuple: A tuple containing the horizontally flipped image and mask.
    """
    transform = alb.HorizontalFlip(p=1)
    
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    
    return transformed_image, transformed_mask


def flip_vertically(image, mask):
    """
    Flip the image and mask vertically.

    Args:
        image (np.ndarray): The input image to be flipped.
        mask (np.ndarray): The input mask to be flipped.

    Returns:
        tuple: A tuple containing the vertically flipped image and mask.
    """
    transform = alb.VerticalFlip(p=1)
    
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    
    return transformed_image, transformed_mask


def flip_vert_hor(image, mask):
    """
    Flip the image and mask both horizontally and vertically.

    Args:
        image (np.ndarray): The input image to be flipped.
        mask (np.ndarray): The input mask to be flipped.

    Returns:
        tuple: A tuple containing the vertically flipped image and mask.
    """
    transform = alb.Compose([
        alb.VerticalFlip(p=1),
        alb.HorizontalFlip(p=1)  
    ])
    
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    
    return transformed_image, transformed_mask

def crop(image, mask):
    """
    Crop the image and mask randomly and resizes themto the original size.

    Args:
        image (np.ndarray): The input image to be cropped.
        mask (np.ndarray): The input mask to be cropped.

    Returns:
        tuple: A tuple containing the cropped and resized image and mask.
    """
    h, w = image.shape[:2]
    
    transform = alb.RandomResizedCrop(p=1, size=(h,w), scale=(0.5,1))
    
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    
    return transformed_image, transformed_mask



def offline_augmentation(image_dir, mask_dir, out_image_dir=None, out_mask_dir=None, angles=None, exact_angle=True, flip_h=False, flip_v=False, flip_hv=False, rand_crop=False, num_crops=1):
    
    if out_image_dir is None:
        out_image_dir = image_dir
    if out_mask_dir is None:
        out_mask_dir = mask_dir
    
    image_paths = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    mask_paths = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if len(image_paths) == 0:
        print(f"No valid files in {image_dir}")
    if len(mask_paths) == 0:
        print(f"No valid files in {mask_dir}")
        
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Mismatch in number of images ({len(image_paths)}) and masks ({len(mask_paths)}).")
    
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)
    
    if angles:
        if not isinstance(angles, list):
            if isinstance(angles, int):
                angles = [angles]
            else:
                raise TypeError("rotate_angles must be object of type list or int.")
    
    
    for img_path in image_paths:
        name, ext = os.path.splitext(os.path.basename(img_path))
        
        mask_path = next((mp for mp in mask_paths if mp == img_path), None)

        if mask_path is None:
            raise ValueError(f"No matching mask for {img_path}")      
          
        image = np.array(Image.open(os.path.join(image_dir,img_path)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(mask_dir,mask_path)).convert("L"))
        
        if angles:
            for angle in angles:
                new_img, new_mask = rotate(image, mask, angle, exact_angle)
                new_img, new_mask = Image.fromarray(new_img), Image.fromarray(new_mask)
                new_name = f"{name}_{angle}{ext}"
                
                new_img.save(os.path.join(out_image_dir, new_name))
                new_mask.save(os.path.join(out_mask_dir, new_name))    
                
        if flip_h:        
            new_img, new_mask = flip_horizontally(image, mask)
            new_img, new_mask = Image.fromarray(new_img), Image.fromarray(new_mask)
            new_name = f"{name}_h{ext}"
            
            new_img.save(os.path.join(out_image_dir, new_name))
            new_mask.save(os.path.join(out_mask_dir, new_name))  
        
        if flip_v:        
            new_img, new_mask = flip_vertically(image, mask)
            new_img, new_mask = Image.fromarray(new_img), Image.fromarray(new_mask)
            new_name = f"{name}_v{ext}"
            
            new_img.save(os.path.join(out_image_dir, new_name))
            new_mask.save(os.path.join(out_mask_dir, new_name))
        
        if flip_hv:
            new_img, new_mask = flip_vert_hor(image, mask)
            new_img, new_mask = Image.fromarray(new_img), Image.fromarray(new_mask)
            new_name = f"{name}_hv{ext}"
            
            new_img.save(os.path.join(out_image_dir, new_name))
            new_mask.save(os.path.join(out_mask_dir, new_name))
            
        if rand_crop:
            for i in range(num_crops):
                new_img, new_mask = crop(image, mask)
                new_img, new_mask = Image.fromarray(new_img), Image.fromarray(new_mask)
                new_name = f"{name}_c{i}{ext}"
                
                new_img.save(os.path.join(out_image_dir, new_name))
                new_mask.save(os.path.join(out_mask_dir, new_name))
        
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        image.save(os.path.join(out_image_dir, f"{name}{ext}"))
        mask.save(os.path.join(out_mask_dir, f"{name}{ext}"))     
                
                
offline_augmentation(image_dir="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\electrical_steel_dataset\\train\\images",
                     mask_dir="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\electrical_steel_dataset\\train\\masks",
                     angles=90,
                     flip_h=True,
                     flip_v=True,
                     flip_hv=False,
                     rand_crop=True,
                     num_crops=1
)