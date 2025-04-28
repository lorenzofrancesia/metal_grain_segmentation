import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import average_precision_score
from PIL import Image
import numpy as np
import scipy.stats
import warnings
import glob
import torchseg

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


#--------------------------------------------------------------------------------------------------
import torch
import torchmetrics
import numpy as np
from skimage import measure
from scipy.stats import wasserstein_distance,ks_2samp
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For distinct colors

# Helper function for distinct colors - can be improved
def get_distinct_colors(n):
    """Generates n visually distinct colors."""
    # Use a standard qualitative colormap and sample from it
    # Using tab20, which has 20 distinct colors. Repeat if n > 20.
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    if n <= 20:
        return colors[:n]
    else:
        # Repeat the colormap - not ideal but simple
        num_repeats = (n + 19) // 20
        colors = np.tile(colors, (num_repeats, 1))
        return colors[:n]


class BinaryMetrics():
    """
    A class to calculate various binary classification metrics using TorchMetrics.

    Args:
        eps (float, optional): A small value to avoid division by zero.  TorchMetrics
            handles this internally, but we keep it for consistency with the original
            class's interface. Default is 1e-5.
        num_classes (int): Number of classes. Should be 1 for binary classification.

    Attributes:
        eps (float): A small value to avoid division by zero.
        metrics (dict): A dictionary to store the calculated metrics.  The keys are
            the names of the metrics, and the values are the *TorchMetrics metric objects*
            (not the functions themselves, as in the original).
    """
    
    def __init__(self, eps=1e-5, num_classes=1, device=None):
        self.eps = eps
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create metrics on the correct device
        self.metrics = {
            "Precision": torchmetrics.classification.BinaryPrecision().to(self.device),
            "Recall": torchmetrics.classification.BinaryRecall().to(self.device),
            "F1": torchmetrics.classification.BinaryF1Score().to(self.device),
            "Accuracy": torchmetrics.classification.BinaryAccuracy().to(self.device),
            "Dice": torchmetrics.classification.BinaryF1Score().to(self.device),
            "IoU": torchmetrics.classification.BinaryJaccardIndex().to(self.device),
            "F2": torchmetrics.classification.BinaryFBetaScore(beta=2.0).to(self.device)
        }

    def _check_and_invert(self, outputs, targets):
        targets = targets.float()
        if torch.mean(targets) > 0.5:
            return 1 - outputs, 1 - targets
        return outputs, targets

    def calculate_metrics(self, outputs, targets, threshold=0.5):
        """
        Calculates all the defined metrics.

        Args:
            outputs (torch.Tensor): Model predictions (probabilities or logits).
            targets (torch.Tensor): Ground truth labels (0 or 1).
            threshold (float, optional):  Threshold to convert probabilities to
                binary predictions. Default: 0.5.

        Returns:
            dict: A dictionary where keys are metric names and values are the
                calculated metric values (as floats).
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)

        # outputs, targets = self._check_and_invert(outputs, targets)

        # Apply threshold to outputs
        preds = (outputs >= threshold).int()

        
        results = {}
        for metric_name, metric_object in self.metrics.items():
            results[metric_name] = metric_object(preds, targets).item()
        
        return results

    
    def reset(self):
        """
        Resets the internal state of all metrics.  This is useful when you want
        to calculate metrics over multiple batches/epochs and need to clear the
        previous calculations.
        """
        for metric_object in self.metrics.values():
            metric_object.reset()


    def to(self, device):
        """Move all metrics to the specified device"""
        self.device = device
        for metric_name in self.metrics:
            self.metrics[metric_name] = self.metrics[metric_name].to(device)
        return self


# --------------------------------------------------------------------------------------------------
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as alb

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_resize_dims_from_transform(transform_compose):
    if not isinstance(transform_compose, transforms.Compose):
        return None 
    for t in transform_compose.transforms:
        if isinstance(t, transforms.Resize):
            size = t.size
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return int(size[0]), int(size[1])
            elif isinstance(size, int):
                return (size, size)
    return None 


class SegmentationDataset(Dataset):
    """
    A dataset class for metal grain segmentation.

    Args:
        image_dir (str): Directory containing the images.
        mask_dir (str): Directory containing the masks.
        image_transform (callable, optional): A function/transform to apply to the images. Default is transforms.ToTensor().
        mask_transform (callable, optional): A function/transform to apply to the masks. Default is transforms.ToTensor().
        normalize (bool, optional): Whether to normalize the images. Default is False.
        augment (bool, optional): Whether to apply data augmentation. Default is False.
        verbose (bool, optional): Whether to print detailed information about the dataset. Default is False.
        mean (list or None, optional): Mean values for normalization. If None, they will be calculated. Default is None.
        std (list or None, optional): Standard deviation values for normalization. If None, they will be calculated. Default is None.
        negative (bool, optional): Whether to invert the mask values (1 -> 0, 0 -> 1). Default is False.
        threshold (float, optional): Threshold for converting masks to binary. Default is 0.5.
    """
    def __init__(self, 
                 image_dir, 
                 mask_dir, 
                 image_transform=transforms.ToTensor(), 
                 mask_transform=transforms.ToTensor(),
                 normalize=False, 
                 augment=False,
                 verbose=False, 
                 mean=None, 
                 std=None, 
                 negative=False,
                 color_mode="RGB", 
                 threshold=0.5):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.negative = negative
        self.verbose = verbose
        self.threshold = threshold
        self.augment = augment
        self.color_space = color_mode
        
        extracted_dims = get_resize_dims_from_transform(self.image_transform)
        if extracted_dims:
            self.target_height, self.target_width = extracted_dims
        else:
            target_height, target_width = 512, 512 # Or raise an error, or make them args
            if self.verbose:
                print(f"Warning: Could not extract target dimensions from image_transform. Using default: H={target_height}, W={target_width}")
        
        self.albumentation_transform = None
        if self.augment:
            self.albumentation_transform = alb.Compose([
                alb.RandomResizedCrop(p=0.3, size=(self.target_height, self.target_width), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
                alb.HorizontalFlip(p=0.3),
                alb.VerticalFlip(p=0.3),
                alb.Rotate(p=0.3, limit=(-90, 90)),
                
                alb.OneOf([alb.RandomBrightnessContrast(p=1, brightness_limit=0.3, contrast_limit=0.3),
                            alb.HueSaturationValue(p=1, hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30)], p=0.3),
                
                alb.OneOf([alb.ElasticTransform(p=1.0, alpha=1.0, sigma=50, alpha_affine=50),
                            alb.GridDistortion(p=1.0, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR),], p=0.3),
                
                alb.OneOf([alb.GaussianBlur(p=1.0, blur_limit=(3,7)),
                            alb.GaussNoise(p=1.0, var_limit=(10.0, 50.0)),], p=0.3),
                
                alb.CoarseDropout(p=0.1, num_holes_range=(1, 8), hole_height_range=(0.03, 0.1), hole_width_range=(0.03, 0.1))
            ])
        
        self.image_paths = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.mask_paths = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.image_paths.sort()
        self.mask_paths.sort()
        
        if len(self.image_paths) == 0 or len(self.mask_paths) == 0:
            raise ValueError("No images or masks found.")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Mismatch in number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}).")
        
        if self.verbose:
            self._dataset_statistics()
        
        if self.mean is not None and self.std is not None:
            print("Mean and std obtained from different dataset.")
            # print(f"Mean:{self.mean}")
            # print(f"Std:{self.std}")
        
        if self.normalize and self.mean is None and self.std is None:
            self.calculate_normalization()
            
        self._validate_dataset()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.color_space not in ['RGB', "None", None]:
            try:
                image = Image.open(img_path).convert(self.color_space)
            except Exception as e:
                print("Color space not available. Proceeding with RGB")
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        if self.augment and self.albumentation_transform:
            try:
                augmented = self.albumentation_transform(image=image_np, mask=mask_np)
                image_np = augmented['image']
                mask_np = augmented['mask']
            except Exception as e:
                 print(f"Error during Albumentations augmentation for index {idx}: {img_path}. Error: {e}")
                 
        image = Image.fromarray(image_np) 
        mask = Image.fromarray(mask_np)
        
        try:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)
        except Exception as e:
            print(f"Error during torchvision transform for index {idx}: {img_path}. Error: {e}")
            # Handle error similar to loading errors (return None, placeholder, or raise)
            raise RuntimeError(f"Torchvision transform failed for index {idx}") from e
        
        mask = self._convert_binary(mask)
            
        if self.normalize and self.mean is not None and self.std is not None:
            image = transforms.Normalize(mean=self.mean, std=self.std)(image)
        
        if self.negative:
            mask = 1 - mask
         
        return image, mask
    
    def _validate_dataset(self):
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("The number of images and masks do not match.")

        if self.verbose:
            for img_name, mask_name in zip(self.image_paths, self.mask_paths):
                img_base = os.path.splitext(img_name)[0]
                mask_base = os.path.splitext(mask_name)[0]
                if img_base != mask_base:
                    print(f"Warning: Image and mask filenames may not match: {img_name}, {mask_name}")
            
    def _is_binary(self, mask):
        """
        Check if a mask is binary.
        """
        unique_values = torch.unique(mask)
        return torch.all((unique_values == 0) | (unique_values == 1))

    def _convert_binary(self, mask):
        """
        Convert tensor to binary
        """
        return (mask > 0.5).float()

              
    def _dataset_statistics(self):
        """
        Calculates and prints dataset statistics, including image sizes and
        optionally, pixel value statistics.  Handles potential errors gracefully.
        """
        image_sizes = []
        widths = []
        heights = []

        for img_path in self.image_paths:
            full_img_path = os.path.join(self.image_dir, img_path)
            try:
                with Image.open(full_img_path) as img:  # Use context manager
                    image_sizes.append(img.size)
                    widths.append(img.size[0])  # Directly append width
                    heights.append(img.size[1]) # Directly append height
            except (IOError, FileNotFoundError) as e:
                print(f"Warning: Could not open image {img_path}: {e}")
                #  Consider adding: continue if you want to skip the problematic image.

        print(f"Number of images: {len(self.image_paths)}")

        if widths and heights: # Check if lists are not empty before min/max
            print(f"Min width: {min(widths)}, Max width: {max(widths)}")
            print(f"Min height: {min(heights)}, Max height: {max(heights)}")
        else:
            print("No valid image sizes found.")
        
    def calculate_normalization(self):
        num_channels = 3
        pixel_sum = np.zeros(num_channels, dtype=np.float64)  # Initialize with zeros
        pixel_squared_sum = np.zeros(num_channels, dtype=np.float64)  # Initialize with zeros
        total_pixels = 0
        
        for img_path in self.image_paths:
            full_img_path = os.path.join(self.image_dir, img_path)
            try:
                img = Image.open(full_img_path).convert("RGB")  # Load directly
                if self.color_space not in ['RGB', "None", None]:
                    try:
                        img = Image.open(full_img_path).convert(self.color_space)
                    except Exception as e:
                        pass
                img_np = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

                pixel_sum += img_np.sum(axis=(0, 1))  # Sum across height and width
                pixel_squared_sum += (img_np ** 2).sum(axis=(0, 1))
                total_pixels += img_np.shape[0] * img_np.shape[1]

            except (IOError, FileNotFoundError) as e:
                print(f"Warning: Could not open image {img_path} for normalization: {e}")
                continue  # Skip this image

        self.mean = (pixel_sum / total_pixels).tolist()  # Convert to list
        self.std = (np.sqrt(pixel_squared_sum / total_pixels - (pixel_sum / total_pixels)**2)).tolist()
        
        if self.verbose:
            print(f"Calculated Mean: {self.mean}, Std: {self.std}")

#----------------------------------------------------------------------------------------------------------

def calculate_iou_simple(pred_bin, target_bin):
    intersection = torch.logical_and(pred_bin, target_bin).sum().float(); union = torch.logical_or(pred_bin, target_bin).sum().float()
    return (intersection / union if union > 0 else torch.tensor(1.0 if intersection == 0 else 0.0)).item()
def calculate_dice_simple(pred_bin, target_bin):
    intersection = torch.logical_and(pred_bin, target_bin).sum().float(); denominator = pred_bin.sum().float() + target_bin.sum().float()
    return ((2. * intersection) / denominator if denominator > 0 else torch.tensor(1.0 if intersection == 0 else 0.0)).item()


# --- Tester ---
class EnsembleTester():
    """
    Tests an ensemble using separate DataLoaders for each model, based on
    lists defining normalization, negative transform, and color mode.
    """

    def __init__(self,
                 data_dir,
                 model_archs,
                 model_paths,
                 normalize_list, 
                 negative_list, 
                 color_mode_list,
                 test_transform=transforms.ToTensor(),
                 output_dir_base="./ensemble_output", 
                 loss_function=nn.BCELoss(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size=1,
                 comparison_metric='iou',
                 binarization_threshold=0.5
                 ):

        # --- Validate Inputs ---
        if not isinstance(model_archs, (list, tuple)) or \
           not isinstance(model_paths, (list, tuple)) or \
           not isinstance(normalize_list, (list, tuple)) or \
           not isinstance(negative_list, (list, tuple)) or \
           not isinstance(color_mode_list, (list, tuple)):
            raise ValueError("`model_archs`, `paths`, `normalize_list`, `negative_list`, `color_mode_list` must be lists/tuples.")
        num_models = len(model_archs)
        if not (num_models == len(model_paths) == len(normalize_list) == len(negative_list) == len(color_mode_list)):
            raise ValueError("Lengths of model args (`archs`, `paths`, `normalize_list`, `negative_list`, `color_mode_list`) must match.")
        if num_models == 0: raise ValueError("At least one model must be provided.")
        
        self.num_models = num_models
        self.model_paths = model_paths
        self.normalize_list = normalize_list
        self.negative_list = negative_list
        self.color_mode_list = color_mode_list
        self.test_transform = test_transform 
        self.device = device
        self.loss_function = loss_function
        self.batch_size = batch_size

        # Dataset & Output
        self.data_dir = data_dir
        self.output_dir_base = output_dir_base
        mode_suffix = "_multi_loader" # Indicate separate loaders used
        self.output_dir = os.path.join(self.output_dir_base, f"ensemble_test_results{mode_suffix}")

        # Models
        self.models = [arch().to(self.device) for arch in model_archs]

        # Normalization params (loaded from checkpoints) & Comparison
        self.model_means = [None] * self.num_models # Will hold loaded means
        self.model_stds = [None] * self.num_models  # Will hold loaded stds
        self.comparison_metric = comparison_metric.lower()
        self.binarization_threshold = binarization_threshold
        if self.comparison_metric not in ['iou', 'dice']: raise ValueError("`comparison_metric` must be 'iou' or 'dice'")

        self._initialize()

    def _initialize(self):
        """Prepares output directory, loads models, creates DataLoaders."""
        # ... (Output directory creation logic - adapt naming) ...
        if not os.path.exists(self.output_dir_base): os.makedirs(self.output_dir_base)
        output_dir_final = self.output_dir
        if os.path.exists(output_dir_final):
             counter = 1; base_name = self.output_dir.replace("_multi_loader","")
             while True: 
                new_dir = f"{base_name}_multi_loader{counter}"
                if not os.path.exists(new_dir): 
                    output_dir_final = new_dir
                    break
                counter += 1
        self.output_dir = output_dir_final; os.makedirs(self.output_dir)

        self.visualization_dir = os.path.join(self.output_dir, "viz"); os.makedirs(self.visualization_dir, exist_ok=True)

        self._load_models_and_norm_params() # Load models AND norm params needed for Datasets
        self._get_dataloaders()           # Create the list of DataLoaders using loaded params

    def _load_models_and_norm_params(self):
        """Loads models state dicts AND normalization params from checkpoints."""
        print(f"Loading {self.num_models} models and their normalization parameters...")
        map_location = torch.device('cpu') if not torch.cuda.is_available() else None
        for i, model_path in enumerate(self.model_paths):
            model_mode = self.color_mode_list[i]
            expected_channels = 3 

            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"Provide valid path for model {i+1}: {model_path}")

            print(f"Loading model {i+1} ({model_mode} input) from {model_path}...")
            # Load full checkpoint to get norm params
            checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)

            # Load state dict
            try:
                 if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
                 else: state_dict = checkpoint; print(f"  Checkpoint for model {i+1} seems to be a raw state_dict.")
                 self.models[i].load_state_dict(state_dict)
                 print(f"  Loaded state dictionary for model {i+1}.")
            except Exception as e: raise RuntimeError(f"Could not load state dict for model {i+1} from {model_path}. Error: {e}")

            # Load and store normalization parameters if normalize flag is True for this model
            if self.normalize_list[i]:
                if isinstance(checkpoint, dict):
                    mean_chk, std_chk = checkpoint.get('dataset_mean'), checkpoint.get('dataset_std')
                    if mean_chk is not None and std_chk is not None:
                        try:
                            # Convert to list/tuple of floats for Dataset init
                            mean_list = list(np.array(mean_chk).flatten())
                            std_list = list(np.array(std_chk).flatten())

                            # Validate number of elements vs expected channels
                            if len(mean_list) != expected_channels: raise ValueError(f"Mean size ({len(mean_list)}) != expected ({expected_channels})")
                            if len(std_list) != expected_channels: raise ValueError(f"Std size ({len(std_list)}) != expected ({expected_channels})")

                            self.model_means[i] = mean_list # Store list/tuple
                            self.model_stds[i] = std_list
                            print(f"  Loaded norm params for model {i+1}: mean={mean_list}, std={std_list}")
                        except Exception as e:
                            print(f"  ERROR: Failed processing norm params for model {i+1}: {e}. Check checkpoint format.")
                            raise # Re-raise as this is critical if normalize=True
                    else:
                        raise ValueError(f"Normalization is True for model {i+1}, but 'dataset_mean' or 'dataset_std' not found in checkpoint: {model_path}")
                else:
                    raise ValueError(f"Normalization is True for model {i+1}, but checkpoint is not a dictionary: {model_path}")
            else:
                 print(f"  Normalization disabled for model {i+1}, skipping norm param loading.")


    def _get_dataloaders(self):
        """Creates a list of DataLoaders using model-specific parameters."""
        print("Preparing DataLoaders for each model...")
        self.dataloaders = []
        dataset_length = -1

        for i in range(self.num_models):
            color_mode_i = self.color_mode_list[i]
            normalize_i = self.normalize_list[i]
            negative_i = self.negative_list[i]
            mean_i = self.model_means[i] # Already loaded list/tuple or None
            std_i = self.model_stds[i]   # Already loaded list/tuple or None

            print(f"  Creating DataLoader for Model {i+1} (Mode: {color_mode_i}, Norm: {normalize_i}, Neg: {negative_i})")

            try:
                 dataset_i = SegmentationDataset(
                     image_dir=os.path.join(self.data_dir, "images"),
                     mask_dir=os.path.join(self.data_dir, "masks"),
                     image_transform=self.test_transform,
                     mask_transform=self.test_transform,
                     color_mode=color_mode_i,
                     normalize=normalize_i,
                     negative=negative_i,
                     mean=mean_i, 
                     std=std_i
                 )
            except Exception as e:
                 print(f"\nError creating dataset for Model {i+1}: {e}")
                 print("Check SegmentationDataset implementation and arguments.")
                 raise

            if i == 0:
                dataset_length = len(dataset_i)
                if dataset_length == 0: raise ValueError("Test dataset is empty!")
            elif len(dataset_i) != dataset_length:
                raise ValueError(f"Dataset length mismatch! Model 0: {dataset_length}, Model {i}: {len(dataset_i)}.")

            loader_i = DataLoader(dataset_i, batch_size=self.batch_size, shuffle=False, num_workers=0)
            self.dataloaders.append(loader_i)

        print(f"Created {len(self.dataloaders)} DataLoaders, each with {dataset_length} samples.")


    def test(self):
        """Evaluates models using separate dataloaders."""

        # DataLoaders are created in _initialize -> _get_dataloaders
        if not hasattr(self, 'dataloaders'):
             raise RuntimeError("Dataloaders not initialized. Call _initialize() first.")

        for model in self.models: model.eval()

        test_loss_ensemble = 0
        self.all_inputs_plot = [] # Store input from first loader for plotting
        self.all_targets = []     # Store targets (checked for consistency)
        self.all_ensemble_outputs = []
        self.all_individual_outputs = [[] for _ in range(self.num_models)]

        data_iters = [iter(dl) for dl in self.dataloaders]
        num_samples = len(self.dataloaders[0]) # Length of DataLoader (number of batches)

        progress_bar = tqdm(range(num_samples), desc=f'Testing (Multi-Loader)', leave=False, file=sys.stdout)
        for batch_idx in progress_bar: # Iterate batches
            batch_outputs_probs_individual = []
            target_ref = None

            try:
                # Fetch the next batch from EACH loader
                batch_inputs = []
                batch_targets = []
                for i, model_iter in enumerate(data_iters):
                    input_i, target_i = next(model_iter) # Get preprocessed batch
                    input_i, target_i = input_i.to(self.device), target_i.to(self.device)
                    batch_inputs.append(input_i)
                    batch_targets.append(target_i)

                    # --- Target Consistency Check (for the first sample in the batch) ---
                    if i == 0:
                        target_ref = target_i
                        # Store representative input (first batch of first loader) for plotting
                        if batch_idx == 0: self.all_inputs_plot = input_i.detach().cpu() # Store entire first batch
                    elif not torch.equal(target_i, target_ref):
                        warnings.warn(f"Target mismatch for batch index {batch_idx} between loader 0 and loader {i}! Using target from loader 0.", RuntimeWarning)

                # --- Store targets once per batch (use reference target) ---
                self.all_targets.append(target_ref.detach().cpu())

                # --- Inference for each model using its specific input ---
                for i in range(self.num_models):
                    input_i = batch_inputs[i] # Get the already processed input for model i
                    try:
                        outputs = self.models[i](input_i)
                        outputs_probs = torch.sigmoid(outputs) # Output [B, 1, H, W]
                        # Store outputs for this batch (will be cat later)
                        self.all_individual_outputs[i].append(outputs_probs.detach().cpu()) # Store CPU tensors
                        batch_outputs_probs_individual.append(outputs_probs) # Keep on device for averaging
                    except Exception as e:
                        print(f"\nError during inference for Model {i+1} on batch index {batch_idx}: {e}")
                        print(f"  Input shape: {input_i.shape}, Color Mode: {self.color_mode_list[i]}, Norm: {self.normalize_list[i]}, Neg: {self.negative_list[i]}")
                        # How to handle? If we continue, ensemble might use fewer models for this batch.
                        # Let's allow continuing but maybe track failures. For now, just print.
                        # We need a placeholder of matching type/device if needed later.
                        # Let's just not append to batch_outputs_probs_individual
                        continue # Skip this model for this batch's ensemble

                # --- Ensemble Prediction ---
                if batch_outputs_probs_individual:
                    # Stack along dim 0 -> [N_ok, B, 1, H, W], mean -> [B, 1, H, W]
                    ensemble_probs = torch.mean(torch.stack(batch_outputs_probs_individual), dim=0)
                else: # All models failed on this batch
                    print(f"Warning: All models failed inference for batch index {batch_idx}.")
                    ensemble_probs = torch.zeros_like(target_ref, dtype=torch.float) # Use ref target shape

                # --- Loss Calculation ---
                try:
                    loss = self.loss_function(ensemble_probs, target_ref) # Target shape [B, 1, H, W]
                    test_loss_ensemble += loss.item()
                except Exception as e:
                     print(f"Warning: Could not calculate ensemble loss for batch index {batch_idx}: {e}.")
                     if test_loss_ensemble == 0 and not np.isnan(test_loss_ensemble): test_loss_ensemble = float('nan')

                # --- Store Ensemble Result ---
                self.all_ensemble_outputs.append(ensemble_probs.detach().cpu())

            except StopIteration:
                warnings.warn("DataLoader iterator exhausted unexpectedly.", RuntimeWarning); break
            except Exception as e:
                raise RuntimeError(f"Failed processing batch index {batch_idx}") from e

        if not self.all_targets: print("Error: No batches were successfully processed."); return None

        # --- Aggregate Results (Cat tensors from batches) ---
        all_targets_cat = torch.cat(self.all_targets, dim=0)
        valid_individual_outputs = [torch.cat(outputs, dim=0) for outputs in self.all_individual_outputs if outputs] # Filter empty lists & Cat
        if not valid_individual_outputs and not self.all_ensemble_outputs: print("Error: No models produced any valid outputs."); return None
        all_ensemble_outputs_cat = torch.cat(self.all_ensemble_outputs, dim=0)
        all_individual_outputs_cat = valid_individual_outputs # Already concatenated above

        # --- Calculate Metrics (Ensemble & Individual) ---
        # ... (No changes needed here, uses aggregated tensors) ...
        print("\nCalculating Overall Ensemble Metrics...")
        ensemble_metrics_results = self._calculate_overall_metrics(all_ensemble_outputs_cat, all_targets_cat, prefix="ensemble")
        avg_loss = test_loss_ensemble / num_samples if num_samples > 0 else float('nan') # Loss averaged over batches
        ensemble_metrics_results['ensemble_loss'] = avg_loss

        print("\nCalculating Overall Metrics for Individual Models...")
        individual_models_metrics = defaultdict(dict)
        model_names = [f"Model_{i+1}" for i, outputs in enumerate(self.all_individual_outputs) if outputs] # Names of successful models
        for i, preds_cat in enumerate(all_individual_outputs_cat):
            model_name = model_names[i]; print(f"Calculating for {model_name}...")
            individual_models_metrics[model_name] = self._calculate_overall_metrics(preds_cat, all_targets_cat, prefix=model_name)

        # --- Calculate Per-Image Metrics ---
        # ... (No changes needed here) ...
        print(f"\nCalculating Per-Image Metrics ({self.comparison_metric.upper()}) for Comparison...")
        per_image_iou_ensemble, per_image_dice_ensemble = self._calculate_per_image_metrics(all_ensemble_outputs_cat, all_targets_cat, threshold=self.binarization_threshold)
        ensemble_scores_for_comparison = per_image_iou_ensemble if self.comparison_metric == 'iou' else per_image_dice_ensemble
        ensemble_metrics_results[f'ensemble_avg_per_image_{self.comparison_metric.upper()}'] = np.nanmean(ensemble_scores_for_comparison)
        individual_scores_for_comparison_list = []
        for i, preds_cat in enumerate(all_individual_outputs_cat): # Iterate successful models
            per_image_iou_ind, per_image_dice_ind = self._calculate_per_image_metrics(preds_cat, all_targets_cat, threshold=self.binarization_threshold)
            individual_scores = per_image_iou_ind if self.comparison_metric == 'iou' else per_image_dice_ind
            individual_scores_for_comparison_list.append(individual_scores)
            individual_models_metrics[model_names[i]][f'avg_per_image_{self.comparison_metric.upper()}'] = np.nanmean(individual_scores)


        # --- Perform Statistical Comparison ---
        # ... (No changes needed here) ...
        statistical_results = self._perform_statistical_comparison(
            ensemble_scores_for_comparison, individual_scores_for_comparison_list, model_names
        )

        # --- Format and Save Results ---
        # (Log the flags used for each model)
        self._format_and_save_results(ensemble_metrics_results, individual_models_metrics, statistical_results)

        final_results = { # ... (remains same) ... }
            "ensemble_metrics": dict(ensemble_metrics_results), "individual_metrics": dict(individual_models_metrics), "statistical_comparison": statistical_results
        }
        return final_results


    # --- Helper Methods (_calculate_overall_metrics, etc.) ---
    # ... (No changes needed in the logic of these helpers) ...
    def _calculate_overall_metrics(self, preds_cat, targets_cat, prefix=""):
        # ... (previous metric calculations remain the same) ...
        metrics_results = defaultdict(lambda: float('nan')); binary_metrics = BinaryMetrics(device='cpu')
        try:
            results_05 = binary_metrics.calculate_metrics(preds_cat, targets_cat, threshold=self.binarization_threshold)
            # Ensure values are floats before storing
            for metric_name, value in results_05.items():
                metrics_results[f"{prefix}_{metric_name}_{self.binarization_threshold}"] = value.item() if isinstance(value, torch.Tensor) else float(value)
        except Exception as e: print(f"Warning: Failed calculating {prefix} binary metrics @{self.binarization_threshold}: {e}")

        # --- Fix for mIoU Calculation ---
        try:
            thresholds_miou = np.arange(0.5, 1.0, 0.05)
            miou_sum, count = 0.0, 0 # Use float for sum
            for threshold in thresholds_miou:
                results_thresh = binary_metrics.calculate_metrics(preds_cat, targets_cat, threshold=float(threshold))
                iou_val = results_thresh.get("IoU", float('nan')) # Default to float NaN

                # Check for NaN appropriately based on type
                is_nan = False
                if isinstance(iou_val, torch.Tensor):
                    is_nan = torch.isnan(iou_val).item() # Get boolean value from tensor
                elif isinstance(iou_val, (float, np.floating)):
                     is_nan = np.isnan(iou_val)
                # Handle potential None case as well
                elif iou_val is None:
                     is_nan = True


                if not is_nan:
                     # Ensure value added is float
                     value_to_add = iou_val.item() if isinstance(iou_val, torch.Tensor) else float(iou_val)
                     miou_sum += value_to_add
                     count += 1

            metrics_results[f"{prefix}_mIoU"] = miou_sum / count if count > 0 else float('nan')
        except Exception as e: print(f"Warning: Failed calculating {prefix} mIoU: {e}")
        # --- End Fix ---

        try:
            metrics_results[f"{prefix}_mAP"] = average_precision_score(targets_cat.numpy().flatten(), preds_cat.numpy().flatten())
        except Exception as e: print(f"Warning: Failed calculating {prefix} mAP: {e}")
        return metrics_results

    def _calculate_per_image_metrics(self, all_preds_cat, all_targets_cat, threshold):
        # ... (Implementation identical to previous version) ...
        num_images = all_preds_cat.shape[0]; per_image_iou, per_image_dice = [], []; all_targets_cat_bool = all_targets_cat.bool(); all_preds_cat_bin = (all_preds_cat > threshold).bool()
        for i in range(num_images): pred_bin_i = all_preds_cat_bin[i].squeeze(); target_bin_i = all_targets_cat_bool[i].squeeze(); iou = calculate_iou_simple(pred_bin_i, target_bin_i); dice = calculate_dice_simple(pred_bin_i, target_bin_i); per_image_iou.append(iou); per_image_dice.append(dice)
        return per_image_iou, per_image_dice

    def _perform_statistical_comparison(self, ensemble_scores, individual_scores_list, model_names):
        # ... (Implementation identical to previous version) ...
        results = {}; print("\n--- Statistical Comparison (Wilcoxon Signed-Rank Test) ---"); print(f"Comparing Ensemble vs. Individual Models based on per-image {self.comparison_metric.upper()}")
        ensemble_scores_np = np.array(ensemble_scores)
        for i, individual_scores in enumerate(individual_scores_list):
            model_name = model_names[i]; individual_scores_np = np.array(individual_scores); valid_indices = ~np.isnan(ensemble_scores_np) & ~np.isnan(individual_scores_np); scores1, scores2 = ensemble_scores_np[valid_indices], individual_scores_np[valid_indices]
            if len(scores1) < 10: stat, p_value = np.nan, np.nan; print(f"Skipping test for Ensemble vs {model_name}: Insufficient valid data points ({len(scores1)}).")
            elif np.all(np.isclose(scores1, scores2)): stat, p_value = np.nan, 1.0; print(f"Skipping test for Ensemble vs {model_name}: All paired scores are identical.")
            else:
                 try: stat, p_value = scipy.stats.wilcoxon(scores1, scores2, alternative='two-sided', zero_method='pratt')
                 except ValueError as e: stat, p_value = np.nan, np.nan; print(f"Warning: Wilcoxon test failed for Ensemble vs {model_name}. Error: {e}")
            results[model_name] = {'statistic': stat, 'p_value': p_value}; significance = "Statistically significant difference (p < 0.05)" if p_value < 0.05 else "No significant difference (p >= 0.05)" if not np.isnan(p_value) else "Test not applicable or failed"; print(f"Ensemble vs. {model_name}: Statistic={stat:.4f}, p-value={p_value:.4f} ({significance})")
        return results


    def _format_and_save_results(self, ensemble_metrics, individual_metrics, statistical_results):
        """Formats and saves results, logging model config flags."""
        print("\n--- Overall Test Results ---"); # ... (console printing same) ...
        formatted_ensemble_results = {k: f"{v:.4f}" if isinstance(v,(int,float,np.number)) and not np.isnan(v) else str(v) for k,v in ensemble_metrics.items()}; print("Ensemble Performance:"); [print(f"  {k.replace('_',' ').capitalize()}: {v}") for k,v in formatted_ensemble_results.items()]
        formatted_individual_results = {}; print("\nIndividual Model Performance:")
        for model_name, metrics in individual_metrics.items(): formatted_individual_results[model_name] = {k: f"{v:.4f}" if isinstance(v,(int,float,np.number)) and not np.isnan(v) else str(v) for k,v in metrics.items()}; print(f"  {model_name}:"); [print(f"    {k.replace('_',' ').capitalize()}: {v}") for k,v in formatted_individual_results[model_name].items()]

        results_file_path = os.path.join(self.output_dir, "ensemble_comparison_results.txt")
        with open(results_file_path, "w") as f:
            f.write("--- Ensemble Test Results (Using Separate DataLoaders) ---\n"); f.write(f"Comparison Metric for Stats: {self.comparison_metric.upper()} @ Threshold: {self.binarization_threshold}\n"); f.write(f"Base Image Transform: {repr(self.test_transform)}\n"); f.write(f"Mask Transform: {repr(self.test_transform)}\n"); f.write("Ensemble Performance:\n"); [f.write(f"  {k.replace('_',' ').capitalize()}: {v}\n") for k,v in formatted_ensemble_results.items()]
            f.write("\n--- Individual Model Performance ---\n")
            processed_model_indices = [int(name.split('_')[-1]) - 1 for name in individual_metrics.keys()]
            for idx, model_name in enumerate(individual_metrics.keys()):
                 model_idx_original = processed_model_indices[idx]
                 f.write(f"  {model_name}:\n");
                 # Log the specific flags used for this model's dataset
                 f.write(f"    Input Color Mode: {self.color_mode_list[model_idx_original]}\n")
                 f.write(f"    Apply Normalize: {self.normalize_list[model_idx_original]}\n")
                 f.write(f"    Apply Negative: {self.negative_list[model_idx_original]}\n")
                 mean_str = str(self.model_means[model_idx_original] if self.model_means[model_idx_original] is not None else "N/A"); std_str = str(self.model_stds[model_idx_original] if self.model_stds[model_idx_original] is not None else "N/A"); f.write(f"    Normalization Params Used: Mean={mean_str}, Std={std_str}\n")
                 [f.write(f"    {k.replace('_',' ').capitalize()}: {v}\n") for k,v in formatted_individual_results[model_name].items()]
            f.write("\n--- Statistical Comparison (Wilcoxon Signed-Rank Test) ---\n"); # ... (stats saving same) ...
            f.write(f"Comparing Ensemble vs. Individual Models based on per-image {self.comparison_metric.upper()}\n"); [ (lambda p, s, t: f.write(f"  Ensemble vs. {m}: Statistic={s:.4f}, p-value={p:.4f} ({t})\n"))(stats['p_value'], stats['statistic'], "Statistically significant difference (p < 0.05)" if stats['p_value'] < 0.05 else "No significant difference (p >= 0.05)" if not np.isnan(stats['p_value']) else "Test not applicable or failed") for m, stats in statistical_results.items() ]
            f.write("\n--- Model Paths Used ---\n"); [f.write(f"  Model {i+1} (Mode: {self.color_mode_list[i]}, Norm: {self.normalize_list[i]}, Neg: {self.negative_list[i]}): {p}\n") for i, p in enumerate(self.model_paths)]

        print(f"\nComprehensive results saved to {results_file_path}"); print(f"Visualizations saved in {self.visualization_dir}")

    # --- Plotting and Saving Predictions ---
    # Plotting uses the stored input from the *first* loader
    def plot_results(self, n=10):
        """Plots input from 1st loader, ensemble pred, and target."""
        # --- Fix for Plotting Check ---
        plot_data_available = hasattr(self, 'all_inputs_plot') and \
                              self.all_inputs_plot is not None and \
                              isinstance(self.all_inputs_plot, torch.Tensor) and \
                              self.all_inputs_plot.numel() > 0

        if not plot_data_available:
             print("No representative input data available to plot (self.all_inputs_plot is missing or empty). Run test() first.")
             return
        # --- End Fix ---


        plot_dir = os.path.join(self.output_dir, "result_plots"); os.makedirs(plot_dir, exist_ok=True)
        input_mode_plot = self.color_mode_list[0] # Mode of first loader
        input_neg_plot = self.negative_list[0]
        plot_title_prefix = f"Input ({input_mode_plot}{', Neg' if input_neg_plot else ''}, Loader 0)"
        print(f"Plotting results using {plot_title_prefix} and ENSEMBLE predictions.")

        # all_inputs_plot holds the *first batch* from loader 0
        num_images_in_first_batch = self.all_inputs_plot.shape[0]
        # Ensure n does not exceed available images or target/prediction length
        num_targets_preds = len(self.all_targets[0]) if self.all_targets else 0 # No. images in first target batch
        num_available = min(num_images_in_first_batch, num_targets_preds)
        num_to_plot = min(n, num_available)

        if num_to_plot == 0:
            print("Warning: No corresponding targets/predictions found for the first input batch. Cannot plot.")
            return

        for idx in range(num_to_plot):
            # Data is already on CPU
            image_tensor = self.all_inputs_plot[idx].numpy() # [C, H, W]
            # Check if target and prediction lists have data for the first batch
            if not self.all_targets or not self.all_ensemble_outputs:
                 print(f"Warning: Missing targets or predictions for batch 0. Cannot plot image {idx}.")
                 continue
            target_tensor = self.all_targets[0][idx].numpy()   # [C_tgt, H, W]
            preds_tensor = self.all_ensemble_outputs[0][idx].numpy() # [C_out, H, W]


            # --- Prepare Image for Plotting ---
            image_for_plot = image_tensor.copy()
            if self.normalize_list[0]:
                 min_val, max_val = image_for_plot.min(), image_for_plot.max()
                 if max_val > min_val: image_for_plot = (image_for_plot - min_val) / (max_val - min_val)
            image_for_plot = np.clip(image_for_plot, 0, 1)

            if image_for_plot.shape[0] in [1, 3]: image_for_plot = np.transpose(image_for_plot, (1, 2, 0))
            if image_for_plot.ndim == 3 and image_for_plot.shape[-1] == 1: image_for_plot = image_for_plot.squeeze(-1)

            # --- Plotting ---
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            cmap_img = 'gray' if image_for_plot.ndim == 2 else None
            axes[0].imshow(image_for_plot, cmap=cmap_img); axes[0].set_title(plot_title_prefix); axes[0].axis("off")
            axes[1].imshow((preds_tensor.squeeze() > self.binarization_threshold), cmap="gray"); axes[1].set_title(f"Ensemble Output (Thr={self.binarization_threshold})"); axes[1].axis("off")
            axes[2].imshow(target_tensor.squeeze(), cmap="gray"); axes[2].set_title("Ground Truth Mask"); axes[2].axis("off")
            plt.tight_layout(); filename = f"ensemble_pred_{idx}.png"; plt.savefig(os.path.join(plot_dir, filename)); plt.close(fig)

        print(f"Saved {num_to_plot} plot(s) (from first batch) to {plot_dir}")

    def save_predictions(self, n=None):
        """Saves ENSEMBLE predicted masks as grayscale images."""
        if not hasattr(self, 'all_ensemble_outputs') or not self.all_ensemble_outputs: print("No test results available to save. Run test() first."); return
        save_dir = os.path.join(self.output_dir, "saved_predictions_ensemble"); os.makedirs(save_dir, exist_ok=True); print(f"Saving ENSEMBLE predicted masks.")

        # all_ensemble_outputs is a list of batch tensors
        img_counter = 0
        num_to_save = n if n is not None else float('inf')

        for batch_tensor in self.all_ensemble_outputs:
            batch_preds = batch_tensor.numpy() # [B, C_out, H, W]
            for i in range(batch_preds.shape[0]): # Iterate through batch
                 if img_counter >= num_to_save: break
                 preds = batch_preds[i].squeeze() # Get [H, W] numpy array
                 pred_img_data = (np.clip(preds, 0, 1) * 255).astype(np.uint8); pred_img = Image.fromarray(pred_img_data, mode='L'); filename = f"ensemble_pred_{img_counter}.png"; pred_img.save(os.path.join(save_dir, filename))
                 img_counter += 1
            if img_counter >= num_to_save: break

        print(f"Saved {img_counter} predicted mask(s) to {save_dir}")


# --- Example Usage ---
if __name__ == '__main__':


    # Example: Model 1 (RGB, Norm, No Neg), Model 2 (L, No Norm, Neg), Model 3 (RGB, Norm, Neg)
    model_architectures = [lambda: torchseg.Unet(encoder_name="resnet34d",
                                                encoder_weights=None,
                                                decoder_attention_type=None,
                                                decoder_use_batchnorm=True, 
                                                ),
                           lambda: torchseg.Unet(encoder_name="resnet34d",
                                                encoder_weights=None,
                                                decoder_attention_type=None,
                                                decoder_use_batchnorm=True, 
                                                ),
                           lambda: torchseg.Unet(encoder_name="resnet34d",
                                                encoder_weights=None,
                                                decoder_attention_type=None,
                                                decoder_use_batchnorm=True, 
                                                ),]
    model_paths = [r"C:\Users\lorenzo.francesia\Documents\github\runs\models\color_spaces\RGB\models\best_similarity.pth",
                   r"C:\Users\lorenzo.francesia\Documents\github\runs\models\color_spaces\HSV\models\best_similarity.pth",
                   r"C:\Users\lorenzo.francesia\Documents\github\runs\models\color_spaces\LAB\models\best_similarity.pth"]

    # 2. Define preprocessing flags FOR EACH model
    normalize_flags = [True, True, True]
    negative_flags = [True, True, True]
    color_modes = ['RGB', 'HSV', 'LAB']

    # 3. Define common base transform (applied before neg/norm in Dataset)
    base_transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor() 
    ])

    # 5. Define Data and Output Paths
    data_directory = r"P:\Lab_Gemensam\Lorenzo\datasets\data_plusplus/test"
    output_base = r"c:\Users\lorenzo.francesia\Documents\github\runs"

    # 6. Instantiate the EnsembleTester
    ensemble_tester = EnsembleTester(
        data_dir=data_directory,
        model_archs=model_architectures,
        model_paths=model_paths,
        normalize_list=normalize_flags, 
        negative_list=negative_flags,
        color_mode_list=color_modes,
        test_transform=base_transform, 
        output_dir_base=output_base,
        loss_function=nn.BCELoss(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=1, 
        comparison_metric='iou',
        binarization_threshold=0.5
    )

    # 7. Run test
    test_results = ensemble_tester.test()

    # 8. Plot/Save (Plotting shows first batch from first loader)
    if test_results:
        ensemble_tester.plot_results(n=1) # Plot up to 2 images from the first batch
        ensemble_tester.save_predictions(n=10) # Save up to 10 predictions total

    print("\n--- Finished ---")