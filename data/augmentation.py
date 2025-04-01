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

def apply_color_jitter(image, mask, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Apply color jitter (brightness, contrast, saturation, hue) to the image.
    The mask remains unchanged.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The input mask (will not be modified by this transform).
        brightness (float): Brightness jitter factor.
        contrast (float): Contrast jitter factor.
        saturation (float): Saturation jitter factor.
        hue (float): Hue jitter factor.

    Returns:
        tuple: A tuple containing the augmented image and the original mask.
    """
    transform = alb.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=1)
    transformed = transform(image=image, mask=mask)
    # Mask is returned by albumentations but should be unchanged by ColorJitter
    return transformed['image'], transformed['mask']

def apply_random_brightness_contrast(image, mask, brightness_limit=0.2, contrast_limit=0.2):
    """
    Apply random brightness and contrast adjustments to the image.
    The mask remains unchanged.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The input mask (will not be modified by this transform).
        brightness_limit (float): Factor range for changing brightness.
        contrast_limit (float): Factor range for changing contrast.

    Returns:
        tuple: A tuple containing the augmented image and the original mask.
    """
    transform = alb.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1)
    transformed = transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']

def apply_rgb_shift(image, mask, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20):
    """
    Apply random shifts to R, G, B channels of the image.
    The mask remains unchanged.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The input mask (will not be modified by this transform).
        r_shift_limit (int): Range for shifting red channel.
        g_shift_limit (int): Range for shifting green channel.
        b_shift_limit (int): Range for shifting blue channel.

    Returns:
        tuple: A tuple containing the augmented image and the original mask.
    """
    transform = alb.RGBShift(r_shift_limit=r_shift_limit, g_shift_limit=g_shift_limit, b_shift_limit=b_shift_limit, p=1)
    transformed = transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']

def apply_gauss_noise(image, mask, var_limit=(10.0, 50.0)):
    """
    Apply Gaussian noise to the image.
    The mask remains unchanged.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The input mask (will not be modified by this transform).
        var_limit (tuple): Range for variance of Gaussian noise.

    Returns:
        tuple: A tuple containing the augmented image and the original mask.
    """
    transform = alb.GaussNoise(var_limit=var_limit, p=1)
    transformed = transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']



def offline_augmentation(image_dir, 
                         mask_dir,
                         out_image_dir=None, 
                         out_mask_dir=None, 
                         angles=None, exact_angle=True, 
                         flip_h=False, flip_v=False, flip_hv=False,
                         rand_crop=False, num_crops=1,
                         color_jitter=False, num_jitter=1, # New flag and count
                         brightness_contrast=False, num_bc=1, # New flag and count
                         rgb_shift=False, num_rgb=1, # New flag and count
                         gauss_noise=False, num_noise=1 # New flag and count
                         ):
    
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
    
    processed_count = 0
    for img_path in image_paths:
        name, ext = os.path.splitext(os.path.basename(img_path))
        
        mask_path = next((mp for mp in mask_paths if mp == img_path), None)

        if mask_path is None:
            raise ValueError(f"No matching mask for {img_path}")      
          
        image = np.array(Image.open(os.path.join(image_dir,img_path)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(mask_dir,mask_path)).convert("L"))
        
        def save_augmented(img_arr, mask_arr, suffix):
            try:
                new_name = f"{name}_{suffix}{ext}"
                img_save_path = os.path.join(out_image_dir, new_name)
                mask_save_path = os.path.join(out_mask_dir, new_name)

                Image.fromarray(img_arr).save(img_save_path)
                Image.fromarray(mask_arr).save(mask_save_path)
            except Exception as e:
                print(f"  Error saving {new_name}: {e}")
        
        if angles:
            for angle in angles:
                rot_suffix = f"rot{angle}" if exact_angle else f"rot_rand{angle}"
                new_img, new_mask = rotate(image, mask, angle, exact_angle)
                save_augmented(new_img, new_mask, rot_suffix)   
                
        if flip_h:        
            new_img, new_mask = flip_horizontally(image, mask)
            save_augmented(new_img, new_mask, "hflip") 
        
        if flip_v:        
            new_img, new_mask = flip_vertically(image, mask)
            save_augmented(new_img, new_mask, "vflip")
        
        if flip_hv:
            new_img, new_mask = flip_vert_hor(image, mask)
            save_augmented(new_img, new_mask, "hvflip")
            
        if rand_crop:
            for i in range(num_crops):
                new_img, new_mask = crop(image, mask)
                save_augmented(new_img, new_mask, f"crop{i}")
                
        if color_jitter:
            for i in range(num_jitter):
                new_img, new_mask = apply_color_jitter(image, mask)
                save_augmented(new_img, new_mask, f"jitter{i}")
                
        if brightness_contrast:
            for i in range(num_bc):
                new_img, new_mask = apply_random_brightness_contrast(image, mask) 
                save_augmented(new_img, new_mask, f"bc{i}")
                
        if rgb_shift:
            for i in range(num_rgb):
                new_img, new_mask = apply_rgb_shift(image, mask)
                save_augmented(new_img, new_mask, f"rgb{i}")
                
        if gauss_noise:
            for i in range(num_noise):
                new_img, new_mask = apply_gauss_noise(image, mask)
                save_augmented(new_img, new_mask, f"noise{i}")
        
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        image.save(os.path.join(out_image_dir, f"{name}{ext}"))
        mask.save(os.path.join(out_mask_dir, f"{name}{ext}"))     
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count}/{len(image_paths)} images...")

    print(f"Offline augmentation finished for {processed_count} images.")
                
                
offline_augmentation(image_dir=r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Documents\Project\datasets\electrical_steel_dataset_plus_aug3\train\images",
                     mask_dir=r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Documents\Project\datasets\electrical_steel_dataset_plus_aug3\train\masks",
                     angles=90,
                     flip_h=True,
                     flip_v=True,
                     flip_hv=True,
                     rand_crop=True,
                     num_crops=1
)