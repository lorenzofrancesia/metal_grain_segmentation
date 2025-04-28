import torch
import torchmetrics
import numpy as np
from skimage import measure
from scipy.stats import wasserstein_distance,ks_2samp, lognorm, kstest
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
    

class GrainMetrics():
    
    """
    Calculates and compares grain statistics between predicted and true masks.

    Focuses on comparing the distributions of grain properties like Area,
    Aspect Ratio, and Circularity using metrics like Wasserstein distance,
    Kolmogorov-Smirnov test, and histogram intersection. Also provides
    overall grain counts and an aggregated similarity score. Includes options
    for visualizing the distributions and labeled masks.
    """
    
    _DISTRIBUTION_PREFIXES = ['area', "diameter", 'aspect_ratio'] #, 'circularity']
    _DISTRIBUTION_METRICS_SUFFIXES = ['wasserstein_similarity', 'ks_similarity', 'histogram_similarity']
    _COUNT_METRICS = ['grain_count_true_total', 'grain_count_pred_total']
    _OVERALL_METRIC = 'grain_overall_similarity'
    
    @staticmethod
    def get_metric_names():
        """
        Returns the list of metric names calculated by calculate_grain_similarity,
        in the order they are returned.
        """
        names = []
        names.extend(GrainMetrics._COUNT_METRICS)
        for prefix in GrainMetrics._DISTRIBUTION_PREFIXES:
            for suffix in GrainMetrics._DISTRIBUTION_METRICS_SUFFIXES:
                names.append(f"{prefix}_{suffix}")
        names.append(GrainMetrics._OVERALL_METRIC)
        return names
    
    def __init__(self, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 visualization_dir=None, 
                 counter=0, 
                 eps=1e-8,
                 min_area_threshold=100):
        
        """
        Initializes the metrics calculation class.

        Args:
            device (str): Computation device ('cuda' or 'cpu').
            visualization_dir (str, optional): Directory to save visualization plots. If None, plots are not saved.
            counter (int): Initial counter for unique visualization filenames (e.g., based on epoch or batch count).
            eps (float): Small epsilon value to prevent division by zero in calculations.
            min_area_threshold (int): Minimum area (in pixels) for a detected region to be considered a valid grain.
        """
        
        self.device = device
        self.visualization_dir = visualization_dir
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        self.vis_counter = counter
        self.eps = eps
        self.min_area_threshold = min_area_threshold

    def _extract_grains(self, mask):    
        """
        Labels connected components in a binary mask and filters them by area.

        Handles binarization (threshold > 0.5) and potential inversion if the
        mask predominantly represents the background (mean < 0.5 - user specific).

        Args:
            mask (torch.Tensor or np.ndarray): Input mask representing grains.

        Returns:
            tuple:
                - np.ndarray: Labeled mask where each valid grain has a unique integer ID (starting from 1). Background is 0.
                - list: List of skimage.measure.RegionProperties for grains meeting the area threshold.
        """
        mask = (mask > 0.5).float()
        if torch.mean(mask) < 0.5:
            mask = 1 - mask
        
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().astype(bool)
        
        labeled_mask = measure.label(mask, connectivity=2)

        regions = measure.regionprops(labeled_mask)
        
        filtered_regions = []
        filtered_labeled_mask = np.zeros_like(labeled_mask)
        current_label = 1
        for region in regions:
            if region.area > self.min_area_threshold:
                filtered_regions.append(region)
                filtered_labeled_mask[labeled_mask == region.label] = current_label
                current_label += 1
        
        return filtered_labeled_mask, filtered_regions

    def _visualize_distributions(self, data_dict, vis_identifier):
        """
        Visualize distributions (Histogram + CDF) for metrics, saving each metric to a separate file.

        Args:
            data_dict (dict): Contains numpy arrays of true/pred properties (e.g., 'true_areas', 'pred_areas').
            vis_identifier (str): Unique string used in filenames (e.g., 'batch_X').
        """
        if not self.visualization_dir: return

        metrics_to_plot = [
            ('Area', 'Grain Area (pixels)', data_dict.get('true_areas', None), data_dict.get('pred_areas', None)),
            ('Diameter', 'Equivalent Diameter (pixels)', data_dict.get('true_diameters', None), data_dict.get('pred_diameters', None)),
            ('AspectRatio', 'Aspect Ratio (Minor/Major)', data_dict.get('true_aspect_ratios', None), data_dict.get('pred_aspect_ratios', None)),
            #('Circularity', 'Circularity (4*pi*A/P^2)', data_dict.get('true_circularities', None), data_dict.get('pred_circularities', None)),
        ]

        for name, xlabel, true_data, pred_data in metrics_to_plot:
            true_data = np.asarray(true_data)
            pred_data = np.asarray(pred_data)
        
        # Skip plotting if either true or predicted data is empty for this metric
            if true_data.size == 0 or pred_data.size == 0:
                print(f"Skipping distribution plot for {name} ({vis_identifier}): No comparable grain data. True_N={true_data.size}, Pred_N={pred_data.size}")
                continue

            fig = None
            try:
                # Create a NEW figure for EACH metric
                fig, axes = plt.subplots(1, 2, figsize=(17, 6)) # Hist, CDF
                ax1, ax2 = axes[0], axes[1]

                # Determine bins and scale (log for Area if wide range, linear otherwise)
                combined_data = np.concatenate((true_data, pred_data))
                min_val = max(self.eps, np.min(combined_data)) # Avoid log(0)
                max_val = np.max(combined_data)

                if min_val >= max_val: # Handle single-value case or near-zero range
                     delta = max(self.eps, abs(max_val * 0.1))
                     min_val = max(self.eps, max_val - delta) # Ensure min_val > 0 for log scale
                     max_val = max_val + delta

                use_log = (name == 'Area' and max_val / (min_val + self.eps) > 100)
                num_bins = 30

                if use_log:
                    bins = np.logspace(np.log10(min_val), np.log10(max_val + self.eps), num_bins + 1)
                    ax1.set_xscale('log')
                    ax2.set_xscale('log')
                elif max_val > min_val:
                    bins = np.linspace(min_val, max_val, num_bins + 1)
                else: # Handle single value edge case after adjustments
                    bins = np.array([min_val, max_val])

                # Histogram Plot
                ax1.hist(true_data, bins=bins, alpha=0.7, label=f'Truth (N={len(true_data)})', density=True, color='blue')
                ax1.hist(pred_data, bins=bins, alpha=0.7, label=f'Pred (N={len(pred_data)})', density=True, color='orange')
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel('Density')
                ax1.set_title(f'{name} Distribution (Histogram)')
                ax1.legend()
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

                # CDF Plot (using step plot)
                true_sorted = np.sort(true_data)
                pred_sorted = np.sort(pred_data)
                true_cdf = np.arange(1, len(true_sorted)+1) / len(true_sorted)
                pred_cdf = np.arange(1, len(pred_sorted)+1) / len(pred_sorted)

                ax2.step(true_sorted, true_cdf, label='Truth CDF', where='post', color='blue')
                ax2.step(pred_sorted, pred_cdf, label='Pred CDF', where='post', color='orange')
                ax2.set_xlabel(xlabel)
                ax2.set_ylabel('Cumulative Probability')
                ax2.set_title(f'{name} Cumulative Distribution')
                ax2.legend()
                ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax2.set_ylim(0, 1.05)

                # Save and Close Individual Figure
                save_path = os.path.join(self.visualization_dir, f'grain_distribution_{name}_{vis_identifier}.png')
                plt.suptitle(f'{name} Distribution Comparison - ID: {vis_identifier}', fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(save_path)
                # print(f"Saved distribution plot: {save_path}") # Optional: uncomment for verbose output
                plt.close(fig)

            except Exception as e:
                print(f"\n--- Error during distribution visualization for {name} (ID {vis_identifier}) ---")
                print(f"Error message: {e}")
                # Potentially log traceback here if needed for debugging
                if fig is not None and plt.fignum_exists(fig.number):
                     plt.close(fig)


    def _visualize_labeled_masks(self, true_labeled_mask, pred_labeled_mask, vis_identifier):
        """ Visualize the detected grains (labeled masks) for ground truth and prediction. """
        if not self.visualization_dir: return
        fig = None
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Use a suitable colormap and explicitly set background (0) color
            cmap_labels = plt.cm.turbo
            cmap_labels.set_bad(color='black') # Color for NaN if any
            cmap_labels.set_under(color='black') # Color for values below vmin (i.e., 0)

            n_true = true_labeled_mask.max()
            n_pred = pred_labeled_mask.max()

            # Plotting with vmin slightly above 0 makes background distinct
            axes[0].imshow(true_labeled_mask, cmap=cmap_labels, vmin=0.1, vmax=max(1, n_true))
            axes[0].set_title(f'Ground Truth Grains (N={n_true})')
            axes[0].axis('off')

            axes[1].imshow(pred_labeled_mask, cmap=cmap_labels, vmin=0.1, vmax=max(1, n_pred))
            axes[1].set_title(f'Predicted Grains (N={n_pred})')
            axes[1].axis('off')

            save_path = os.path.join(self.visualization_dir, f'labeled_masks_{vis_identifier}.png')
            plt.suptitle(f'Detected Grain Comparison - ID: {vis_identifier}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(save_path)
            # print(f"Saved labeled mask visualization: {save_path}") # Optional: uncomment for verbose output
            plt.close(fig)
        except Exception as e:
            print(f"Error during labeled mask visualization for ID {vis_identifier}: {e}")
            if fig is not None and plt.fignum_exists(fig.number):
                 plt.close(fig)

    
    def _calculate_shape_properties(self, regions):
        """Helper calculates area, aspect ratio, circularity for a list of skimage regions."""
        areas, diameters, aspect_ratios, circularities = [], [], [], []
        for region in regions:
            area = region.area
            areas.append(area)
            
            diameter = 2 * np.sqrt(area / np.pi) if area > self.eps else 0.0
            diameters.append(diameter)

            # Aspect Ratio (minor/major), handles zero major axis
            minor_ax = region.minor_axis_length
            major_ax = region.major_axis_length
            aspect_ratio = (minor_ax / major_ax) if major_ax > self.eps else 0.0
            aspect_ratios.append(aspect_ratio)

            # Circularity (4*pi*Area / Perimeter^2), handles zero/invalid perimeter
            perimeter = region.perimeter
            if perimeter is not None and perimeter > self.eps:
                 # Theoretical max is 1, clamp pixel-based results
                 circularity = (4 * np.pi * area) / (perimeter**2)
                 circularities.append(np.clip(circularity, 0.0, 1.0))
            else:
                 circularities.append(0.0) # Assign 0 for invalid perimeter

        return np.array(areas), np.array(diameters), np.array(aspect_ratios), np.array(circularities)
    
    
    def _get_distribution_similarity(self, true_dist, pred_dist, metric_prefix):
        """
        Calculates distribution similarity metrics (Wasserstein, KS, Histogram Intersection).

        Returns 0 similarity if either distribution is empty.
        """
        sim_results = {}
        similarity_value = 0.0 # Default similarity if comparison cannot be made

        # Cannot compare if one or both distributions lack data
        if true_dist.size == 0 or pred_dist.size == 0:
            # print(f"[{metric_prefix}] Distribution Similarity: Cannot compare. True_N={true_dist.size}, Pred_N={pred_dist.size}. Setting similarity to {similarity_value}.")
            sim_results[f'{metric_prefix}_wasserstein_similarity'] = similarity_value
            sim_results[f'{metric_prefix}_ks_similarity'] = similarity_value
            sim_results[f'{metric_prefix}_histogram_similarity'] = similarity_value
            return sim_results

        try:
            # Wasserstein: Cost to transform one distribution into the other. Normalized by mean true value.
            emd_raw = wasserstein_distance(true_dist, pred_dist)
            scale_factor = np.mean(true_dist)
            scaled_emd = (emd_raw / (scale_factor + self.eps)) # Use normalized distance
            # Convert distance to similarity (1 = identical, -> 0 for large distances)
            sim_results[f'{metric_prefix}_wasserstein_similarity'] = 1.0 / (1.0 + scaled_emd)

            # KS Test: Max difference between CDFs. Similarity = 1 - statistic.
            ks_stat, _ = ks_2samp(true_dist, pred_dist)
            sim_results[f'{metric_prefix}_ks_similarity'] = 1.0 - ks_stat

            # Histogram Intersection: Overlap between normalized histograms.
            combined = np.concatenate((true_dist, pred_dist))
            min_val_hist = max(self.eps, np.min(combined))
            max_val_hist = np.max(combined)
            num_bins = 30

            # Robust bin definition
            if min_val_hist >= max_val_hist:
                delta = max(self.eps, abs(max_val_hist * 0.1))
                bins = np.array([max(self.eps, max_val_hist - delta), max_val_hist + delta])
            elif metric_prefix == 'area' and max_val_hist / (min_val_hist + self.eps) > 100: # Log scale for area
                 bins = np.logspace(np.log10(min_val_hist), np.log10(max_val_hist + self.eps), num_bins + 1)
            else: # Linear scale
                 bins = np.linspace(min_val_hist, max_val_hist, num_bins + 1)

            true_hist, _ = np.histogram(true_dist, bins=bins, density=True)
            pred_hist, _ = np.histogram(pred_dist, bins=bins, density=True)
            bin_widths = np.diff(bins)
            intersection = np.sum(np.minimum(true_hist, pred_hist) * bin_widths)
            sim_results[f'{metric_prefix}_histogram_similarity'] = np.clip(intersection, 0.0, 1.0)

        except Exception as e:
            print(f"\n--- Error calculating distribution similarity for {metric_prefix} ---")
            print(f"Error message: {e}")
            # Assign default 0 similarity on calculation error
            sim_results[f'{metric_prefix}_wasserstein_similarity'] = 0.0
            sim_results[f'{metric_prefix}_ks_similarity'] = 0.0
            sim_results[f'{metric_prefix}_histogram_similarity'] = 0.0

        return sim_results

    def calculate_grain_similarity(self, pred_masks, true_masks, visualize=True, visualize_example_image=True):
        """
        Calculates grain similarity metrics for a batch of masks.

        Compares distributions of area, aspect ratio, and circularity between
        all grains found in the batch's predicted masks vs. true masks.

        Args:
            pred_masks (torch.Tensor or np.ndarray): Batch masks (B, H, W) or (B, 1, H, W). Assumes values > 0.5 are grains.
            true_masks (torch.Tensor or np.ndarray): Batch masks (B, H, W) or (B, 1, H, W). Assumes values > 0.5 are grains.
            visualize (bool): Generate and save distribution plots if visualization_dir is set.
            visualize_example_image (bool): Save labeled masks for the first image in the batch if visualize=True.

        Returns:
            dict: Dictionary containing calculated similarity metrics (counts, distribution similarities, overall score).
        """
        batch_size = pred_masks.shape[0]
        results = defaultdict(float)

        # Accumulate properties across all images in the batch
        batch_true_areas, batch_pred_areas = [], []
        batch_true_diameters, batch_pred_diameters = [], []
        batch_true_aspect_ratios, batch_pred_aspect_ratios = [], []
        batch_true_circularities, batch_pred_circularities = [], []
        

        first_img_labeled_true, first_img_labeled_pred = None, None
        first_img_processed = False

        for i in range(batch_size):
            pred_mask_single = torch.as_tensor(pred_masks[i], device=self.device).squeeze()
            true_mask_single = torch.as_tensor(true_masks[i], device=self.device).squeeze()

            labeled_mask_true, true_regions = self._extract_grains(true_mask_single)
            labeled_mask_pred, pred_regions = self._extract_grains(pred_mask_single)

            # Store first image's labeled masks for visualization
            if visualize and visualize_example_image and not first_img_processed:
                first_img_labeled_true = labeled_mask_true
                first_img_labeled_pred = labeled_mask_pred
                first_img_processed = True

            # Calculate properties for this image's grains
            true_areas_img, true_diam_img, true_ar_img, true_circ_img = self._calculate_shape_properties(true_regions)
            pred_areas_img, pred_diam_img, pred_ar_img, pred_circ_img = self._calculate_shape_properties(pred_regions)

            # Add this image's properties to the batch lists
            batch_true_areas.extend(true_areas_img)
            batch_pred_areas.extend(pred_areas_img)
            batch_true_diameters.extend(true_diam_img) 
            batch_pred_diameters.extend(pred_diam_img) 
            batch_true_aspect_ratios.extend(true_ar_img)
            batch_pred_aspect_ratios.extend(pred_ar_img)
            batch_true_circularities.extend(true_circ_img)
            batch_pred_circularities.extend(pred_circ_img)

        # Convert accumulated lists to numpy arrays for calculations
        all_true_areas = np.array(batch_true_areas)
        all_pred_areas = np.array(batch_pred_areas)
        all_true_diameters = np.array(batch_true_diameters) 
        all_pred_diameters = np.array(batch_pred_diameters) 
        all_true_ar = np.array(batch_true_aspect_ratios)
        all_pred_ar = np.array(batch_pred_aspect_ratios)
        all_true_circ = np.array(batch_true_circularities)
        all_pred_circ = np.array(batch_pred_circularities)

        # Record total grain counts found across the batch
        results['grain_count_true_total'] = float(all_true_areas.size)
        results['grain_count_pred_total'] = float(all_pred_areas.size)

        # Calculate distribution similarities for the aggregated batch data
        vis_id = f"batch_{self.vis_counter}" # Unique ID for this call's visualizations

        results.update(self._get_distribution_similarity(all_true_areas, all_pred_areas, "area"))
        results.update(self._get_distribution_similarity(all_true_diameters, all_pred_diameters, "diameter"))
        results.update(self._get_distribution_similarity(all_true_ar, all_pred_ar, "aspect_ratio"))
        # results.update(self._get_distribution_similarity(all_true_circ, all_pred_circ, "circularity"))

        # Calculate Overall Similarity Score (weighted average of Wasserstein similarities)
        weights = {'area': 0.75, 'diameter': 0.2, 'aspect_ratio': 0.05} #, 'circularity': 0.05}
        shape_similarity_weighted = 0.0
        total_weight = 0.0
        
        for prefix, weight in weights.items():
             metric_key = f'{prefix}_{self._DISTRIBUTION_METRICS_SUFFIXES[0]}' # Using Wasserstein as the basis for overall score
             if metric_key in results:
                 # Similarity is 0 if data was missing/incomparable, correctly contributing 0 here.
                 shape_similarity_weighted += results[metric_key] * weight
                 total_weight += weight
        
        wasserstein_similarity = (shape_similarity_weighted / total_weight) if total_weight > self.eps else 0.0
        n_true = results[self._COUNT_METRICS[0]]
        n_pred = results[self._COUNT_METRICS[1]]
        
        if n_true == 0 and n_pred == 0:
             count_similarity = 1.0
             wasserstein_similarity = 0.0 
        else:
             min_count = min(n_true, n_pred)
             max_count = max(n_true, n_pred)
             count_similarity = min_count / (max_count + self.eps)
             
        results['grain_overall_similarity'] = wasserstein_similarity * count_similarity

        # Perform Visualization if requested and directory is set
        if visualize and self.visualization_dir:
            vis_data = {
                'true_areas': all_true_areas, 'pred_areas': all_pred_areas,
                'true_diameters': all_true_diameters, 'pred_diameters': all_pred_diameters,
                'true_aspect_ratios': all_true_ar, 'pred_aspect_ratios': all_pred_ar,
                #'true_circularities': all_true_circ, 'pred_circularities': all_pred_circ
            }
            self._visualize_distributions(vis_data, vis_id)

            if visualize_example_image and first_img_labeled_true is not None and first_img_labeled_pred is not None:
                self._visualize_labeled_masks(first_img_labeled_true, first_img_labeled_pred, f"{vis_id}_example_img0")
            elif visualize_example_image:
                 print(f"[{vis_id}] Labeled mask visualization requested, but no valid masks generated for the first image.")

        self.vis_counter += 1 # Increment counter for the next function call
        return dict(results)