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
    def __init__(self, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 visualization_dir=r"C:\Users\lorenzo.francesia\OneDrive - Swerim\Desktop\viz", 
                 counter=0,
                 min_area_threshold=100):
        self.device = device
        
        self.visualization_dir = visualization_dir
        # Create visualization directory if specified
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        
        self.vis_counter = counter
        self.min_area_threshold = min_area_threshold
        
    def _extract_grains(self, mask):
        
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        inverted_mask = ~mask.astype(bool)
        
        labeled_mask = measure.label(inverted_mask, connectivity=2)
        
        regions = measure.regionprops(labeled_mask)
        
        filtered_regions = []
        for region in regions:
            if region.area > self.min_area_threshold:
                filtered_regions.append(region)
        
        return labeled_mask, filtered_regions
    
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
    
    
class GrainMetricsExtended():
    def __init__(self, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 visualization_dir=None, 
                 counter=0, 
                 eps=1e-8,
                 min_area_threshold=100):
        
        self.device = device
        self.visualization_dir = visualization_dir
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        self.vis_counter = counter
        self.eps = eps
        self.min_area_threshold = min_area_threshold

    def _extract_grains(self, mask):
        if isinstance(mask, torch.Tensor):
            # Ensure mask is on CPU and boolean/uint8 before numpy conversion
            mask_np = mask.cpu().bool().numpy()
        elif isinstance(mask, np.ndarray):
            mask_np = mask.astype(bool)
        else:
            raise TypeError("Input mask must be a torch.Tensor or numpy.ndarray")

        grains_mask = ~mask_np

        labeled_mask, num_labels = measure.label(grains_mask, connectivity=2, return_num=True)

        regions = measure.regionprops(labeled_mask)

        return labeled_mask, regions

    def _visualize_distributions(self, data_dict, vis_identifier):
        """ Visualize distributions for multiple metrics (Area, Aspect Ratio, Circularity). """
        if not self.visualization_dir: return

        metrics_to_plot = [
            ('Area', 'Grain Area (pixels)', data_dict.get('true_areas', None), data_dict.get('pred_areas', None)),
            ('AspectRatio', 'Aspect Ratio (Minor/Major)', data_dict.get('true_aspect_ratios', None), data_dict.get('pred_aspect_ratios', None)),
            ('Circularity', 'Circularity (4*pi*A/P^2)', data_dict.get('true_circularities', None), data_dict.get('pred_circularities', None)),
        ]

        num_metrics = len([m for m in metrics_to_plot if m[2] is not None and m[3] is not None and len(m[2]) > 0 and len(m[3]) > 0])
        if num_metrics == 0:
            print(f"Skipping distribution visualization for {vis_identifier}: No valid grain data for any metric.")
            return

        try:
            # Adjust layout based on number of metrics with valid data
            fig, axes = plt.subplots(num_metrics, 2, figsize=(16, 6 * num_metrics), squeeze=False)
            plot_row = 0

            for name, xlabel, true_data, pred_data in metrics_to_plot:
                if true_data is None or pred_data is None or len(true_data) == 0 or len(pred_data) == 0:
                    continue # Skip if no data for this metric

                ax1 = axes[plot_row, 0]
                ax2 = axes[plot_row, 1]

                # --- Histogram ---
                min_val = max(self.eps, min(np.min(true_data), np.min(pred_data))) # Avoid log(0)
                max_val = max(np.max(true_data), np.max(pred_data))
                use_log = (name == 'Area' and max_val / min_val > 100) # Only log for area potentially
                if use_log:
                    bins = np.logspace(np.log10(min_val), np.log10(max_val + self.eps), 30)
                    ax1.set_xscale('log')
                else:
                    bins = np.linspace(min_val, max_val, 30)

                ax1.hist(true_data, bins=bins, alpha=0.7, label=f'Truth (N={len(true_data)})', density=True)
                ax1.hist(pred_data, bins=bins, alpha=0.7, label=f'Pred (N={len(pred_data)})', density=True)
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel('Density')
                ax1.set_title(f'{name} Distribution (Histogram)')
                ax1.legend()
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

                # --- CDF ---
                true_sorted = np.sort(true_data)
                pred_sorted = np.sort(pred_data)
                true_cdf = np.arange(1, len(true_sorted)+1) / len(true_sorted)
                pred_cdf = np.arange(1, len(pred_sorted)+1) / len(pred_sorted)

                if use_log:
                    ax2.set_xscale('log')

                ax2.step(true_sorted, true_cdf, label='Truth CDF')
                ax2.step(pred_sorted, pred_cdf, label='Pred CDF')
                ax2.set_xlabel(xlabel)
                ax2.set_ylabel('Cumulative Probability')
                ax2.set_title(f'{name} Cumulative Distribution')
                ax2.legend()
                ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

                plot_row += 1 # Move to next row for next metric

            save_path = os.path.join(self.visualization_dir, f'grain_multi_distribution_{vis_identifier}.png')
            plt.suptitle(f'Grain Property Distribution Comparison - ID: {vis_identifier}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect slightly
            plt.savefig(save_path)
            plt.close(fig)

        except Exception as e:
            print(f"Error during multi-distribution visualization for ID {vis_identifier}: {e}")


    def _visualize_labeled_masks(self, true_labeled_mask, pred_labeled_mask, vis_identifier):
        """ Visualize the detected grains (labeled masks) for ground truth and prediction. """
        if not self.visualization_dir: return
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            cmap_labels = plt.cm.nipy_spectral
            cmap_labels.set_bad(color='black')
            cmap_labels.set_under(color='black')
            n_true = true_labeled_mask.max()
            n_pred = pred_labeled_mask.max()

            im_true = axes[0].imshow(true_labeled_mask, cmap=cmap_labels, vmin=0.1, vmax=max(1, n_true))
            axes[0].set_title(f'Ground Truth Grains (N={n_true})')
            axes[0].axis('off')

            im_pred = axes[1].imshow(pred_labeled_mask, cmap=cmap_labels, vmin=0.1, vmax=max(1, n_pred))
            axes[1].set_title(f'Predicted Grains (N={n_pred})')
            axes[1].axis('off')

            save_path = os.path.join(self.visualization_dir, f'labeled_masks_{vis_identifier}.png')
            plt.suptitle(f'Detected Grain Comparison - ID: {vis_identifier}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error during labeled mask visualization for ID {vis_identifier}: {e}")


    def calculate_grain_similarity(self, pred_masks, true_masks, visualize=True, visualize_example_image=True):
        """ Calculates grain similarity including shape metrics (aspect ratio, circularity). """
        batch_size = pred_masks.shape[0]
        results = defaultdict(float)

        # Store all properties across the batch
        all_true_areas, all_pred_areas = [], []
        all_true_aspect_ratios, all_pred_aspect_ratios = [], []
        all_true_circularities, all_pred_circularities = [], []
        img_grain_counts_true, img_grain_counts_pred = [], []

        first_img_labeled_true, first_img_labeled_pred = None, None
        first_img_processed = False

        for i in range(batch_size):
            pred_mask = pred_masks[i].squeeze()
            true_mask = true_masks[i].squeeze()

            # Binarize based on threshold (assuming 0=grain, 1=boundary after thresh)
            pred_mask_bin = (pred_mask > 0.5) # Use appropriate threshold
            true_mask_bin = (true_mask > 0.5) # Use appropriate threshold

            labeled_mask_true, true_regions = self._extract_grains(true_mask_bin)
            labeled_mask_pred, pred_regions = self._extract_grains(pred_mask_bin)

            if visualize and visualize_example_image and not first_img_processed:
                first_img_labeled_true = labeled_mask_true
                first_img_labeled_pred = labeled_mask_pred
                first_img_processed = True

            # --- Extract properties for this image ---
            true_areas_img, true_ar_img, true_circ_img = [], [], []
            for region in true_regions:
                area = region.area
                if area <= self.min_area_threshold: continue
                true_areas_img.append(area)
                # Aspect Ratio
                minor_ax = region.minor_axis_length
                major_ax = region.major_axis_length
                aspect_ratio = (minor_ax / major_ax) if major_ax > self.eps else 0.0
                true_ar_img.append(aspect_ratio)
                # Circularity
                perimeter = region.perimeter
                # Perimeter might be 0 for single pixel object, also regionprops perimeter needs care
                if perimeter is not None and perimeter > self.eps:
                     circularity = (4 * np.pi * area) / (perimeter**2)
                     # Clamp circularity to [0, 1] as theoretical max is 1, but pixel approximations can exceed
                     true_circ_img.append(np.clip(circularity, 0.0, 1.0))
                else:
                     true_circ_img.append(0.0) # Assign 0 if perimeter is invalid


            pred_areas_img, pred_ar_img, pred_circ_img = [], [], []
            for region in pred_regions:
                area = region.area
                if area <= 0: continue
                pred_areas_img.append(area)
                minor_ax = region.minor_axis_length
                major_ax = region.major_axis_length
                aspect_ratio = (minor_ax / major_ax) if major_ax > self.eps else 0.0
                pred_ar_img.append(aspect_ratio)
                perimeter = region.perimeter
                if perimeter is not None and perimeter > self.eps:
                     circularity = (4 * np.pi * area) / (perimeter**2)
                     pred_circ_img.append(np.clip(circularity, 0.0, 1.0))
                else:
                     pred_circ_img.append(0.0)


            # --- Accumulate Batch Stats ---
            img_grain_counts_true.append(len(true_areas_img))
            img_grain_counts_pred.append(len(pred_areas_img))
            all_true_areas.extend(true_areas_img)
            all_pred_areas.extend(pred_areas_img)
            all_true_aspect_ratios.extend(true_ar_img)
            all_pred_aspect_ratios.extend(pred_ar_img)
            all_true_circularities.extend(true_circ_img)
            all_pred_circularities.extend(pred_circ_img)

        # --- Convert lists to numpy arrays ---
        all_true_areas = np.array(all_true_areas) if all_true_areas else np.array([0.0])
        all_pred_areas = np.array(all_pred_areas) if all_pred_areas else np.array([0.0])
        all_true_ar = np.array(all_true_aspect_ratios) if all_true_aspect_ratios else np.array([0.0])
        all_pred_ar = np.array(all_pred_aspect_ratios) if all_pred_aspect_ratios else np.array([0.0])
        all_true_circ = np.array(all_true_circularities) if all_true_circularities else np.array([0.0])
        all_pred_circ = np.array(all_pred_circularities) if all_pred_circularities else np.array([0.0])

        # --- Aggregate Batch Statistics (Count, Avg/Median Area - Optional to keep) ---
        # ... (keep or remove simple avg/median calculations as needed) ...
        results['grain_count_true_total'] = sum(img_grain_counts_true)
        results['grain_count_pred_total'] = sum(img_grain_counts_pred)

        # --- Calculate Distribution Similarity Metrics ---
        vis_id = f"call_{self.vis_counter}"

        # Helper function to calculate distribution similarity
        def get_distribution_similarity(true_dist, pred_dist, metric_prefix):
            sim_results = {}
            if len(true_dist) == 0 or len(pred_dist) == 0:
                sim_results[f'{metric_prefix}_wasserstein_similarity'] = 0.0
                sim_results[f'{metric_prefix}_ks_similarity'] = 0.0
                sim_results[f'{metric_prefix}_histogram_similarity'] = 0.0
                return sim_results

            # Wasserstein
            emd_raw = wasserstein_distance(true_dist, pred_dist)
            scale_factor = np.mean(true_dist)
            scaled_emd = (emd_raw / scale_factor) if scale_factor > self.eps else np.inf
            sim_results[f'{metric_prefix}_wasserstein_similarity'] = 1.0 / (1.0 + scaled_emd)

            # KS Test
            ks_stat, _ = ks_2samp(true_dist, pred_dist)
            sim_results[f'{metric_prefix}_ks_similarity'] = 1.0 - ks_stat

            # Histogram Intersection (using min/max of combined data for bins)
            combined = np.concatenate((true_dist, pred_dist))
            min_val_hist = max(self.eps, np.min(combined))
            max_val_hist = np.max(combined)
            num_bins = 30
            if metric_prefix == 'area' and max_val_hist / min_val_hist > 100: # Log scale for area only
                 bins = np.logspace(np.log10(min_val_hist), np.log10(max_val_hist + self.eps), num_bins + 1)
            elif max_val_hist > min_val_hist:
                 bins = np.linspace(min_val_hist, max_val_hist, num_bins + 1)
            else: # Handle case where all values are the same
                 bins = np.array([min_val_hist, min_val_hist + self.eps]) # Single bin

            true_hist, _ = np.histogram(true_dist, bins=bins, density=True)
            pred_hist, _ = np.histogram(pred_dist, bins=bins, density=True)
            bin_widths = np.diff(bins)
            sim_results[f'{metric_prefix}_histogram_similarity'] = np.sum(np.minimum(true_hist, pred_hist) * bin_widths)

            return sim_results

        # Calculate for Area, Aspect Ratio, Circularity
        results.update(get_distribution_similarity(all_true_areas, all_pred_areas, "area"))
        results.update(get_distribution_similarity(all_true_ar, all_pred_ar, "aspect_ratio"))
        results.update(get_distribution_similarity(all_true_circ, all_pred_circ, "circularity"))

        weights = {'area': 0.4, 'aspect_ratio': 0.3, 'circularity': 0.3}
        overall_sim = 0.0
        total_weight = 0.0
        for prefix, weight in weights.items():
             # Use a key metric like Wasserstein or average the similarities
             metric_key = f'{prefix}_wasserstein_similarity'
             if metric_key in results:
                 overall_sim += results[metric_key] * weight
                 total_weight += weight
        results['grain_overall_similarity'] = overall_sim / total_weight if total_weight > 0 else 0.0

        if visualize:
            vis_data = {
                'true_areas': all_true_areas, 'pred_areas': all_pred_areas,
                'true_aspect_ratios': all_true_ar, 'pred_aspect_ratios': all_pred_ar,
                'true_circularities': all_true_circ, 'pred_circularities': all_pred_circ
            }
            if self.visualization_dir:
                self._visualize_distributions(vis_data, vis_id)
            if visualize_example_image and first_img_labeled_true is not None:
                self._visualize_labeled_masks(first_img_labeled_true, first_img_labeled_pred, f"{vis_id}_example_img0")

        self.vis_counter += 1
        return dict(results) # Return results dictionary