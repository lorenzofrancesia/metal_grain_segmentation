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
    
    
class GrainMetrics():
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', visualization_dir=r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Desktop\viz", counter=0):
        self.device = device
        
        self.visualization_dir = visualization_dir
        # Create visualization directory if specified
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        
        self.vis_counter = counter
        
    def _extract_grains(self, mask):
        
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        inverted_mask = ~mask.astype(bool)
        
        labeled_mask = measure.label(inverted_mask, connectivity=2)
        
        regions = measure.regionprops(labeled_mask)
        
        return labeled_mask, regions
    
    def _visualize_distributions(self, true_areas, pred_areas, vis_identifier):
        """
        Visualize grain size distributions (histogram and CDF) for analysis.

        Args:
            true_areas (np.ndarray): Array of grain areas from the ground truth.
            pred_areas (np.ndarray): Array of grain areas from the prediction.
            vis_identifier (str): Unique identifier for the saved plot file.
        """
        if not self.visualization_dir: return
        if len(true_areas) == 0 or len(pred_areas) == 0:
            print(f"Skipping distribution visualization for {vis_identifier}: No grains found in true or pred.")
            return

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # --- Histogram comparison ---
            min_area = max(1, min(np.min(true_areas), np.min(pred_areas))) # Avoid log(0)
            max_area = max(np.max(true_areas), np.max(pred_areas))
            if max_area / min_area > 100 and min_area > 0 : # Heuristic for large range
                 bins = np.logspace(np.log10(min_area), np.log10(max_area + 1), 30) # +1 in case max_area=min_area
                 ax1.set_xscale('log')
            else:
                 bins = np.linspace(min_area, max_area, 30)

            ax1.hist(true_areas, bins=bins, alpha=0.7, label=f'Ground Truth (N={len(true_areas)})', density=True)
            ax1.hist(pred_areas, bins=bins, alpha=0.7, label=f'Prediction (N={len(pred_areas)})', density=True)
            ax1.set_xlabel('Grain Area (pixels)')
            ax1.set_ylabel('Density')
            ax1.set_title('Grain Size Distribution (Density Histogram)')
            ax1.legend()
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

            # --- CDF comparison ---
            true_sorted = np.sort(true_areas)
            pred_sorted = np.sort(pred_areas)
            true_cdf = np.arange(1, len(true_sorted)+1) / len(true_sorted)
            pred_cdf = np.arange(1, len(pred_sorted)+1) / len(pred_sorted)

            # Match x-axis scale with histogram if log
            if ax1.get_xscale() == 'log':
                ax2.set_xscale('log')

            ax2.step(true_sorted, true_cdf, label='Ground Truth CDF')
            ax2.step(pred_sorted, pred_cdf, label='Prediction CDF')
            ax2.set_xlabel('Grain Area (pixels)')
            ax2.set_ylabel('Cumulative Probability (CDF)')
            ax2.set_title('Cumulative Distribution Functions')
            ax2.legend()
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Save figure
            save_path = os.path.join(self.visualization_dir, f'grain_distribution_{vis_identifier}.png')
            plt.suptitle(f'Grain Area Distribution Comparison - ID: {vis_identifier}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_path)
            plt.close(fig)

        except Exception as e:
            print(f"Error during distribution visualization for ID {vis_identifier}: {e}")


    def _visualize_labeled_masks(self, true_labeled_mask, pred_labeled_mask, vis_identifier):
        """
        Visualize the detected grains (labeled masks) for ground truth and prediction.

        Args:
            true_labeled_mask (np.ndarray): Labeled mask for ground truth grains.
            pred_labeled_mask (np.ndarray): Labeled mask for predicted grains.
            vis_identifier (str): Unique identifier for the saved plot file.
        """
        if not self.visualization_dir: return

        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Prepare colormaps: Use a map that shows background (0) distinctly (e.g., black)
            # Using 'nipy_spectral' often works well. Add black for background.
            cmap_labels = plt.cm.nipy_spectral
            cmap_labels.set_bad(color='black') # Map background (NaN or masked) to black if needed
            cmap_labels.set_under(color='black') # Map values < vmin to black

            # Count grains for titles
            n_true = true_labeled_mask.max()
            n_pred = pred_labeled_mask.max()

            # Plot Ground Truth Labeled Mask
            ax = axes[0]
            # Using vmin=0.1 ensures label 0 (background) uses the 'under' color (black)
            im_true = ax.imshow(true_labeled_mask, cmap=cmap_labels, vmin=0.1, vmax=max(1, n_true))
            ax.set_title(f'Ground Truth Grains (N={n_true})')
            ax.axis('off')
            # plt.colorbar(im_true, ax=ax, fraction=0.046, pad=0.04) # Optional colorbar

            # Plot Predicted Labeled Mask
            ax = axes[1]
            im_pred = ax.imshow(pred_labeled_mask, cmap=cmap_labels, vmin=0.1, vmax=max(1, n_pred))
            ax.set_title(f'Predicted Grains (N={n_pred})')
            ax.axis('off')
            # plt.colorbar(im_pred, ax=ax, fraction=0.046, pad=0.04) # Optional colorbar

            # Save figure
            save_path = os.path.join(self.visualization_dir, f'labeled_masks_{vis_identifier}.png')
            plt.suptitle(f'Detected Grain Comparison - ID: {vis_identifier}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_path)
            plt.close(fig)

        except Exception as e:
            print(f"Error during labeled mask visualization for ID {vis_identifier}: {e}")


    def calculate_grain_similarity(self, pred_masks, true_masks, visualize=True, visualize_example_image=True):
        """
        Calculates grain similarity metrics between predicted and ground truth masks,
        comparing distributions of raw grain areas (in pixels). Optionally visualizes
        distributions and an example pair of labeled masks.

        Args:
            pred_masks (torch.Tensor): Predicted masks tensor (B, H, W) or (B, 1, H, W).
                                       Values should be probabilities or logits if not boolean.
                                       Assumes 0=grain, 1=boundary after thresholding.
            true_masks (torch.Tensor): Ground truth masks tensor (B, H, W) or (B, 1, H, W).
                                       Assumes 0=grain, 1=boundary (or convertible to boolean).
            visualize (bool): If True and `visualization_dir` is set, saves plots
                              of the aggregated grain size distributions.
            visualize_example_image (bool): If True (and visualize=True), saves a plot
                                            comparing the labeled masks for the first
                                            image in the batch.

        Returns:
            dict: A dictionary containing various grain similarity metrics calculated
                  over the entire batch.
        """

        batch_size = pred_masks.shape[0]
        results = defaultdict(float)
        all_true_areas_list = []
        all_pred_areas_list = []
        img_grain_counts_true = []
        img_grain_counts_pred = []

        # Store data for the first image if example visualization is requested
        first_img_labeled_true = None
        first_img_labeled_pred = None
        first_img_processed = False # Flag to ensure we only store the first

        # --- Process each image in the batch ---
        for i in range(batch_size):
            pred_mask = pred_masks[i].squeeze()
            true_mask = true_masks[i].squeeze()

            # Convert to boolean mask (0=grain, 1=boundary)
            if pred_mask.dtype != torch.bool and pred_mask.dtype != torch.uint8:
                 pred_mask_bin = (pred_mask > 0.5)
            else:
                 pred_mask_bin = pred_mask.bool()

            if true_mask.dtype != torch.bool and true_mask.dtype != torch.uint8:
                 true_mask_bin = (true_mask > 0.5)
            else:
                 true_mask_bin = true_mask.bool()

            # Extract grains (returns labeled mask and region properties)
            labeled_mask_true, true_regions = self._extract_grains(true_mask_bin)
            labeled_mask_pred, pred_regions = self._extract_grains(pred_mask_bin)

            # Store first image's labeled masks for visualization
            if visualize and visualize_example_image and not first_img_processed:
                first_img_labeled_true = labeled_mask_true
                first_img_labeled_pred = labeled_mask_pred
                first_img_processed = True

            # Get Grain Areas (Raw Pixel Counts)
            true_areas_img = np.array([region.area for region in true_regions if region.area > 0])
            pred_areas_img = np.array([region.area for region in pred_regions if region.area > 0])

            # Accumulate stats
            count_true = len(true_areas_img)
            count_pred = len(pred_areas_img)
            img_grain_counts_true.append(count_true)
            img_grain_counts_pred.append(count_pred)
            results['avg_grain_size_true_accum'] += np.sum(true_areas_img)
            results['avg_grain_size_pred_accum'] += np.sum(pred_areas_img)
            if count_true > 0:
                results['median_grain_size_true_img_sum'] += np.median(true_areas_img)
                results['valid_imgs_median_true'] += 1
            if count_pred > 0:
                results['median_grain_size_pred_img_sum'] += np.median(pred_areas_img)
                results['valid_imgs_median_pred'] += 1

            all_true_areas_list.extend(true_areas_img)
            all_pred_areas_list.extend(pred_areas_img)

        # --- Aggregate Batch Statistics ---
        # (Calculation code remains the same as previous version)
        results['grain_count_true_total'] = sum(img_grain_counts_true)
        results['grain_count_pred_total'] = sum(img_grain_counts_pred)
        results['grain_count_true_avg_per_img'] = np.mean(img_grain_counts_true) if img_grain_counts_true else 0
        results['grain_count_pred_avg_per_img'] = np.mean(img_grain_counts_pred) if img_grain_counts_pred else 0
        total_true_area = results['avg_grain_size_true_accum']
        total_pred_area = results['avg_grain_size_pred_accum']
        results['avg_grain_size_true'] = total_true_area / results['grain_count_true_total'] if results['grain_count_true_total'] > 0 else 0
        results['avg_grain_size_pred'] = total_pred_area / results['grain_count_pred_total'] if results['grain_count_pred_total'] > 0 else 0
        results['median_grain_size_true_avg_per_img'] = results['median_grain_size_true_img_sum'] / results['valid_imgs_median_true'] if results['valid_imgs_median_true'] > 0 else 0
        results['median_grain_size_pred_avg_per_img'] = results['median_grain_size_pred_img_sum'] / results['valid_imgs_median_pred'] if results['valid_imgs_median_pred'] > 0 else 0
        del results['avg_grain_size_true_accum'], results['avg_grain_size_pred_accum']
        del results['median_grain_size_true_img_sum'], results['median_grain_size_pred_img_sum']
        del results['valid_imgs_median_true'], results['valid_imgs_median_pred']

        # Convert collected areas to numpy arrays
        all_true_areas = np.array(all_true_areas_list)
        all_pred_areas = np.array(all_pred_areas_list)

        # --- Distribution Similarity Calculations ---
        vis_id = f"call_{self.vis_counter}" # Unique identifier for this call

        if len(all_true_areas) == 0 or len(all_pred_areas) == 0:
            print(f"Warning (ID: {vis_id}): No grains found in true or predicted masks for the batch. Distribution metrics will be zero.")
            # Set similarity metrics to 0
            results['wasserstein_similarity'] = 0.0
            results['ks_similarity'] = 0.0
            results['histogram_similarity'] = 0.0
            results['percentile_similarity'] = 0.0
            results['mean_similarity'] = 0.0
            results['variance_similarity'] = 0.0
            results['moments_similarity'] = 0.0
            results['grain_distribution_similarity'] = 0.0
            results['grain_count_similarity'] = 0.0
            results['wasserstein_distance_raw'] = np.nan
            for p in [10, 25, 50, 75, 90]: results[f'p{p}_similarity'] = 0.0

             # Visualize example image even if distributions are empty (shows empty masks)
            if visualize and visualize_example_image and first_img_labeled_true is not None:
                 self._visualize_labeled_masks(first_img_labeled_true, first_img_labeled_pred, f"{vis_id}_example_img0")

        else:
            # --- Visualize aggregated distributions ---
            if visualize and self.visualization_dir:
                self._visualize_distributions(all_true_areas, all_pred_areas, vis_id)

            # --- Visualize example labeled masks (first image) ---
            if visualize and visualize_example_image and first_img_labeled_true is not None:
                self._visualize_labeled_masks(first_img_labeled_true, first_img_labeled_pred, f"{vis_id}_example_img0")

            # --- Metric Calculations (Code remains the same as previous version) ---

            # 1. Wasserstein
            emd_raw = wasserstein_distance(all_true_areas, all_pred_areas)
            results['wasserstein_distance_raw'] = emd_raw
            scale_factor = np.mean(all_true_areas)
            if scale_factor < 1e-8:
                 scaled_emd = np.inf if emd_raw > 1e-8 else 0.0
            else:
                 scaled_emd = emd_raw / scale_factor
            results['wasserstein_similarity'] = 1.0 / (1.0 + scaled_emd)

            # 2. Kolmogorov-Smirnov
            ks_stat, ks_p_value = ks_2samp(all_true_areas, all_pred_areas)
            results['ks_statistic'] = ks_stat
            results['ks_p_value'] = ks_p_value
            results['ks_similarity'] = 1.0 - ks_stat

            # 3. Histogram
            min_area_hist = max(1, min(np.min(all_true_areas), np.min(all_pred_areas)))
            max_area_hist = max(np.max(all_true_areas), np.max(all_pred_areas))
            num_bins = 30
            if max_area_hist / min_area_hist > 100:
                 bins = np.logspace(np.log10(min_area_hist), np.log10(max_area_hist + 1), num_bins + 1)
            else:
                 bins = np.linspace(min_area_hist, max_area_hist, num_bins + 1)
            true_hist, _ = np.histogram(all_true_areas, bins=bins, density=True)
            pred_hist, _ = np.histogram(all_pred_areas, bins=bins, density=True)
            bin_widths = np.diff(bins)
            results['histogram_similarity'] = np.sum(np.minimum(true_hist, pred_hist) * bin_widths)

            # 4. Percentiles
            percentiles = [10, 25, 50, 75, 90]
            true_percentiles = np.percentile(all_true_areas, percentiles)
            pred_percentiles = np.percentile(all_pred_areas, percentiles)
            percentile_sims = []
            for i, p in enumerate(percentiles):
                 p_true, p_pred = true_percentiles[i], pred_percentiles[i]
                 if max(p_true, p_pred) > 1e-8: ratio = min(p_true, p_pred) / max(p_true, p_pred)
                 elif min(p_true, p_pred) <= 1e-8: ratio = 1.0
                 else: ratio = 0.0
                 results[f'p{p}_similarity'] = ratio
                 percentile_sims.append(ratio)
            results['percentile_similarity'] = np.mean(percentile_sims) if percentile_sims else 0.0

            # 5. Moments
            true_mean, pred_mean = np.mean(all_true_areas), np.mean(all_pred_areas)
            true_var, pred_var = np.var(all_true_areas), np.var(all_pred_areas)
            if max(true_mean, pred_mean) > 1e-8: results['mean_similarity'] = min(true_mean, pred_mean) / max(true_mean, pred_mean)
            elif min(true_mean, pred_mean) <= 1e-8: results['mean_similarity'] = 1.0
            else: results['mean_similarity'] = 0.0
            if max(true_var, pred_var) > 1e-8: results['variance_similarity'] = min(true_var, pred_var) / max(true_var, pred_var)
            elif min(true_var, pred_var) <= 1e-8: results['variance_similarity'] = results['mean_similarity']
            else: results['variance_similarity'] = 0.0
            results['moments_similarity'] = 0.5 * (results['mean_similarity'] + results['variance_similarity'])

            # 6. Count Similarity
            count_true_total = results['grain_count_true_total']
            count_pred_total = results['grain_count_pred_total']
            if max(count_true_total, count_pred_total) > 0:
                 results['grain_count_similarity'] = min(count_true_total, count_pred_total) / max(count_true_total, count_pred_total)
            else: results['grain_count_similarity'] = 1.0

            # 7. Overall Distribution Similarity
            main_metrics = [ ('wasserstein_similarity', 0.4), ('ks_similarity', 0.2),
                             ('histogram_similarity', 0.2), ('percentile_similarity', 0.2) ]
            weighted_sum = sum(results[metric] * weight for metric, weight in main_metrics)
            total_weight = sum(weight for _, weight in main_metrics)
            results['grain_distribution_similarity'] = weighted_sum / total_weight if total_weight > 0 else 0.0

        # --- Finalization ---
        # Increment the visualization counter for the next call
        self.vis_counter += 1

        # Convert defaultdict back to a regular dict for return
        return dict(results)