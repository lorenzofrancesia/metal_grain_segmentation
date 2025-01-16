import albumentations as alb

def rotate(image, mask, angle):
    """
    Rotate the image and mask by a specified angle.

    Args:
        image (np.ndarray): The input image to be rotated.
        mask (np.ndarray): The input mask to be rotated.
        angle (int): The angle by which to rotate the image and mask.

    Returns:
        tuple: A tuple containing the rotated image and mask.
    """
    transform = alb.Rotate(limit=angle, p=1)
    
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
        alb.vertical_flip(p=1),
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
    
    transform = alb.RandomResizedCrop(p=1, size=(h,w))
    
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    
    return transformed_image, transformed_mask