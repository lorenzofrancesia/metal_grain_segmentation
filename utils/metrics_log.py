# Necessary Imports
import os
import torch
import numpy as np
from skimage import measure
from scipy.stats import wasserstein_distance, ks_2samp, lognorm, kstest # Added lognorm, kstest back
from collections import defaultdict
import matplotlib.pyplot as plt
import math

class GrainMetrics():
    """
    Calculates and compares grain statistics between predicted and true masks.

    Focuses on comparing the distributions of grain properties like Area,
    Equivalent Diameter, and Aspect Ratio using metrics like Wasserstein distance,
    Kolmogorov-Smirnov test, and histogram intersection. Fits a log-normal
    distribution to the diameter data and reports parameters and goodness-of-fit.
    Also provides overall grain counts and an aggregated similarity score.
    Includes options for visualizing distributions (with fits) and labeled masks.
    """

    # Define the properties whose distributions will be compared
    # 'circularity' is still commented out based on the last user input
    _DISTRIBUTION_PREFIXES = ['area', "diameter", 'aspect_ratio'] #, 'circularity']
    _DISTRIBUTION_METRICS_SUFFIXES = ['wasserstein_similarity', 'ks_similarity', 'histogram_similarity']
    # *** Metrics related to lognormal fit for diameter ***
    _DIAMETER_FIT_METRICS_SUFFIXES = ['fit_s', 'fit_loc', 'fit_scale', 'fit_ks_stat', 'fit_ks_pval']
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
        # Distribution similarity metrics
        for prefix in GrainMetrics._DISTRIBUTION_PREFIXES:
            for suffix in GrainMetrics._DISTRIBUTION_METRICS_SUFFIXES:
                names.append(f"{prefix}_{suffix}")
        # *** Diameter lognormal fit metrics ***
        for suffix in GrainMetrics._DIAMETER_FIT_METRICS_SUFFIXES:
            names.append(f"diameter_true_{suffix}")
            names.append(f"diameter_pred_{suffix}")
        # Overall metric
        names.append(GrainMetrics._OVERALL_METRIC)
        return names

    def __init__(self,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 visualization_dir=None,
                 counter=0,
                 eps=1e-8,
                 min_area_threshold=100):
        """ Initializes the metrics calculation class. (Same as before) """
        self.device = device
        self.visualization_dir = visualization_dir
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        self.vis_counter = counter
        self.eps = eps
        self.min_area_threshold = min_area_threshold

    def _extract_grains(self, mask):
        """ Extracts grains, returns labeled mask and region props. (Same as before) """
        # ... (Implementation from previous correct version) ...
        if isinstance(mask, torch.Tensor):
            mask_cpu = mask.detach().float().cpu()
        else:
            mask_cpu = torch.as_tensor(mask, dtype=torch.float32)
        mask_bin = (mask_cpu > 0.5).float()
        if torch.mean(mask_bin) < 0.5:
            mask_bin = 1.0 - mask_bin
        mask_np = mask_bin.numpy().astype(bool)
        labeled_mask = measure.label(mask_np, connectivity=2)
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


    def _calculate_shape_properties(self, regions):
        """ Calculates area, diameter, aspect ratio, circularity. (Same as before) """
        # ... (Implementation from previous correct version) ...
        areas, diameters, aspect_ratios, circularities = [], [], [], []
        for region in regions:
            area = region.area
            if area < self.eps: continue
            areas.append(area)
            diameter = 2 * np.sqrt(area / np.pi)
            diameters.append(diameter)
            minor_ax = region.minor_axis_length
            major_ax = region.major_axis_length
            aspect_ratio = (minor_ax / major_ax) if major_ax > self.eps else 0.0
            aspect_ratios.append(aspect_ratio)
            perimeter = region.perimeter
            if perimeter is not None and perimeter > self.eps:
                 circularity = (4 * np.pi * area) / (perimeter**2)
                 circularities.append(np.clip(circularity, 0.0, 1.0))
            else:
                 circularities.append(0.0)
        return (np.array(areas) if areas else np.array([]),
                np.array(diameters) if diameters else np.array([]),
                np.array(aspect_ratios) if aspect_ratios else np.array([]),
                np.array(circularities) if circularities else np.array([]))


    def _fit_lognormal_and_test(self, data, data_label):
        """
        Fits a log-normal distribution to the data and performs a KS test.

        Args:
            data (np.ndarray): 1D array of data points (e.g., diameters). Must contain positive values.
            data_label (str): Label for the data source ('true' or 'pred').

        Returns:
            dict: Dictionary containing fitted parameters (s, loc, scale)
                  and KS test results (stat, pval), prefixed appropriately.
                  Returns NaNs if fitting or testing fails or data is insufficient.
        """
        fit_results = {}
        prefix = f"diameter_{data_label}"
        # Initialize results with NaN using the class attribute
        for suffix in self._DIAMETER_FIT_METRICS_SUFFIXES:
            fit_results[f"{prefix}_{suffix}"] = np.nan

        # Ensure data is numpy array and filter out non-positive values for lognorm
        data = np.asarray(data)
        data = data[data > self.eps] # Log-normal requires positive data

        # Need at least 2 data points for fitting and meaningful KS test
        if data.size < 2:
            # print(f"LogNormal Fit [{prefix}]: Insufficient positive data (N={data.size}). Skipping.")
            return fit_results

        try:
            # Fit the log-normal distribution.
            # REMOVED incorrect epsilon keyword argument
            s, loc, scale = lognorm.fit(data) # Corrected line
            fit_results[f"{prefix}_fit_s"] = s
            fit_results[f"{prefix}_fit_loc"] = loc
            fit_results[f"{prefix}_fit_scale"] = scale

            # Perform KS test against the *fitted* log-normal distribution
            ks_stat, ks_pvalue = kstest(data, 'lognorm', args=(s, loc, scale))
            fit_results[f"{prefix}_fit_ks_stat"] = ks_stat
            fit_results[f"{prefix}_fit_ks_pval"] = ks_pvalue

        except (ValueError, RuntimeError) as e: # Catch potential errors from fitting/testing
            print(f"\n--- Error during log-normal fitting/testing for {prefix} ---")
            print(f"Filtered data size: {data.size}, First 5 positive: {data[:5]}")
            print(f"Error message: {e}")
            # Results remain NaN on error

        return fit_results


    def _visualize_distributions(self, data_dict, vis_identifier, fit_params_dict=None):
        """ Visualize distributions (Histogram + CDF) with optional lognormal fit overlay for diameter. (Same as before, with fit plotting) """
        if not self.visualization_dir: return
        if fit_params_dict is None: fit_params_dict = {}

        # Dynamically build metrics to plot based on class prefixes
        metrics_to_plot = []
        if 'area' in self._DISTRIBUTION_PREFIXES:
            metrics_to_plot.append(('Area', 'Grain Area (pixels)', data_dict.get('true_areas', None), data_dict.get('pred_areas', None)))
        if 'diameter' in self._DISTRIBUTION_PREFIXES:
             metrics_to_plot.append(('Diameter', 'Equivalent Diameter (pixels)', data_dict.get('true_diameters', None), data_dict.get('pred_diameters', None)))
        if 'aspect_ratio' in self._DISTRIBUTION_PREFIXES:
            metrics_to_plot.append(('AspectRatio', 'Aspect Ratio (Minor/Major)', data_dict.get('true_aspect_ratios', None), data_dict.get('pred_aspect_ratios', None)))
        # if 'circularity' in self._DISTRIBUTION_PREFIXES: ... # Add if re-enabled

        for name, xlabel, true_data, pred_data in metrics_to_plot:
             if true_data is None: true_data = np.array([])
             if pred_data is None: pred_data = np.array([])
             if true_data.size == 0 or pred_data.size == 0: continue

             fig = None
             try:
                 fig, axes = plt.subplots(1, 2, figsize=(17, 6))
                 ax1, ax2 = axes[0], axes[1]
                 combined_data = np.concatenate((true_data, pred_data))
                 min_val = max(self.eps, np.min(combined_data))
                 max_val = np.max(combined_data)
                 if max_val <= min_val + self.eps:
                     delta = max(self.eps, abs(max_val * 0.1))
                     min_val = max(self.eps, max_val - delta)
                     max_val = max_val + delta

                 use_log = (name in ['Area', 'Diameter'] and max_val / min_val > 100)
                 num_bins = 30
                 if use_log:
                     bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)
                     ax1.set_xscale('log')
                     ax2.set_xscale('log')
                 else:
                     bins = np.linspace(min_val, max_val, num_bins + 1)

                 # --- Histogram Plot ---
                 ax1.hist(true_data, bins=bins, alpha=0.6, label=f'Truth (N={len(true_data)})', density=True, color='blue')
                 ax1.hist(pred_data, bins=bins, alpha=0.6, label=f'Pred (N={len(pred_data)})', density=True, color='orange')

                 # <<< Add Fitted PDF Plotting for Diameter >>>
                 if name == 'Diameter' and fit_params_dict:
                     x_pdf = np.linspace(min_val, max_val, 200)
                     true_s = fit_params_dict.get('diameter_true_fit_s', np.nan)
                     if not np.isnan(true_s):
                         true_loc = fit_params_dict.get('diameter_true_fit_loc', 0)
                         true_scale = fit_params_dict.get('diameter_true_fit_scale', 1)
                         try:
                            true_pdf = lognorm.pdf(x_pdf, s=true_s, loc=true_loc, scale=true_scale)
                            ax1.plot(x_pdf, true_pdf, color='blue', linestyle='--', linewidth=2, label='Truth Fit (Lognormal)')
                         except ValueError as ve: print(f"Warn: Could not plot true lognorm PDF {vis_identifier}: {ve}")
                     pred_s = fit_params_dict.get('diameter_pred_fit_s', np.nan)
                     if not np.isnan(pred_s):
                         pred_loc = fit_params_dict.get('diameter_pred_fit_loc', 0)
                         pred_scale = fit_params_dict.get('diameter_pred_fit_scale', 1)
                         try:
                            pred_pdf = lognorm.pdf(x_pdf, s=pred_s, loc=pred_loc, scale=pred_scale)
                            ax1.plot(x_pdf, pred_pdf, color='orange', linestyle=':', linewidth=2, label='Pred Fit (Lognormal)')
                         except ValueError as ve: print(f"Warn: Could not plot pred lognorm PDF {vis_identifier}: {ve}")

                 ax1.set_xlabel(xlabel)
                 ax1.set_ylabel('Density')
                 ax1.set_title(f'{name} Distribution (Histogram)')
                 ax1.legend()
                 ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

                 # --- CDF Plot ---
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

                 # Save and Close
                 save_path = os.path.join(self.visualization_dir, f'grain_distribution_{name.lower()}_{vis_identifier}.png')
                 plt.suptitle(f'{name} Distribution Comparison - ID: {vis_identifier}', fontsize=14)
                 plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                 plt.savefig(save_path)
                 plt.close(fig)

             except Exception as e:
                 print(f"\n--- Error during distribution visualization for {name} (ID {vis_identifier}) ---")
                 print(f"Error message: {e}")
                 if fig is not None and plt.fignum_exists(fig.number): plt.close(fig)


    def _visualize_labeled_masks(self, true_labeled_mask, pred_labeled_mask, vis_identifier):
        """ Visualize labeled masks. (Same as before) """
        # ... (Implementation from previous correct version) ...
        if not self.visualization_dir: return
        fig = None
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            cmap_labels = plt.cm.turbo.copy()
            cmap_labels.set_bad(color='black')
            cmap_labels.set_under(color='black')
            n_true = int(true_labeled_mask.max())
            n_pred = int(pred_labeled_mask.max())
            axes[0].imshow(true_labeled_mask, cmap=cmap_labels, vmin=0.1, vmax=max(1, n_true), interpolation='none')
            axes[0].set_title(f'Ground Truth Grains (N={n_true})')
            axes[0].axis('off')
            axes[1].imshow(pred_labeled_mask, cmap=cmap_labels, vmin=0.1, vmax=max(1, n_pred), interpolation='none')
            axes[1].set_title(f'Predicted Grains (N={n_pred})')
            axes[1].axis('off')
            save_path = os.path.join(self.visualization_dir, f'labeled_masks_{vis_identifier}.png')
            plt.suptitle(f'Detected Grain Comparison - ID: {vis_identifier}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error during labeled mask visualization for ID {vis_identifier}: {e}")
            if fig is not None and plt.fignum_exists(fig.number): plt.close(fig)


    def _get_distribution_similarity(self, true_dist, pred_dist, metric_prefix):
        """ Calculates distribution similarities (Wasserstein, KS, Histogram). (Same as before) """
        # ... (Implementation from previous correct version) ...
        sim_results = {}
        sim_results[f'{metric_prefix}_wasserstein_similarity'] = 0.0
        sim_results[f'{metric_prefix}_ks_similarity'] = 0.0
        sim_results[f'{metric_prefix}_histogram_similarity'] = 0.0
        if true_dist.size == 0 or pred_dist.size == 0: return sim_results
        try:
            true_dist = true_dist[np.isfinite(true_dist)]
            pred_dist = pred_dist[np.isfinite(pred_dist)]
            if true_dist.size == 0 or pred_dist.size == 0: return sim_results
            emd_raw = wasserstein_distance(true_dist, pred_dist)
            scale_factor = np.mean(true_dist)
            scaled_emd = (emd_raw / (scale_factor + self.eps))
            sim_results[f'{metric_prefix}_wasserstein_similarity'] = 1.0 / (1.0 + scaled_emd)
            ks_stat, _ = ks_2samp(true_dist, pred_dist)
            sim_results[f'{metric_prefix}_ks_similarity'] = 1.0 - ks_stat
            combined = np.concatenate((true_dist, pred_dist))
            min_val_hist = max(self.eps, np.min(combined))
            max_val_hist = np.max(combined)
            num_bins = 30
            if max_val_hist <= min_val_hist + self.eps:
                delta = max(self.eps, abs(max_val_hist * 0.1))
                bins = np.array([max(self.eps, max_val_hist - delta), max_val_hist + delta])
            elif metric_prefix in ['area', 'diameter'] and max_val_hist / min_val_hist > 100:
                 bins = np.logspace(np.log10(min_val_hist), np.log10(max_val_hist), num_bins + 1)
            else:
                 bins = np.linspace(min_val_hist, max_val_hist, num_bins + 1)
            true_hist, _ = np.histogram(true_dist, bins=bins, density=True)
            pred_hist, _ = np.histogram(pred_dist, bins=bins, density=True)
            bin_widths = np.diff(bins)
            intersection = np.sum(np.minimum(true_hist, pred_hist) * bin_widths)
            sim_results[f'{metric_prefix}_histogram_similarity'] = np.clip(intersection, 0.0, 1.0)
        except ValueError as ve:
            print(f"\n--- ValueError during similarity calculation for {metric_prefix} ---")
            print(f"Error message: {ve}")
        except Exception as e:
            print(f"\n--- Error calculating distribution similarity for {metric_prefix} ---")
            print(f"Error message: {e}")
        return sim_results


    def calculate_grain_similarity(self, pred_masks, true_masks, visualize=True, visualize_example_image=True):
        """
        Calculates grain similarity metrics and fits log-normal to diameters.

        Compares distributions of properties defined in _DISTRIBUTION_PREFIXES.
        Fits log-normal to diameter data.

        Args: ... (same as before)

        Returns:
            dict: Dictionary containing calculated similarity metrics, fit parameters, and overall score.
        """
        batch_size = pred_masks.shape[0]
        results = defaultdict(float)

        batch_props_true = defaultdict(list)
        batch_props_pred = defaultdict(list)
        first_img_labeled_true, first_img_labeled_pred = None, None
        first_img_processed = False

        for i in range(batch_size):
            pred_mask_single = torch.as_tensor(pred_masks[i], device=self.device).squeeze()
            true_mask_single = torch.as_tensor(true_masks[i], device=self.device).squeeze()
            labeled_mask_true, true_regions = self._extract_grains(true_mask_single)
            labeled_mask_pred, pred_regions = self._extract_grains(pred_mask_single)

            if visualize and visualize_example_image and not first_img_processed:
                if labeled_mask_true is not None and labeled_mask_pred is not None:
                     first_img_labeled_true = labeled_mask_true
                     first_img_labeled_pred = labeled_mask_pred
                first_img_processed = True

            # Note: circularity calculated here but might not be used later
            true_areas_img, true_diam_img, true_ar_img, true_circ_img = self._calculate_shape_properties(true_regions)
            pred_areas_img, pred_diam_img, pred_ar_img, pred_circ_img = self._calculate_shape_properties(pred_regions)

            # Accumulate properties based on _DISTRIBUTION_PREFIXES
            if 'area' in self._DISTRIBUTION_PREFIXES:
                batch_props_true['area'].extend(true_areas_img)
                batch_props_pred['area'].extend(pred_areas_img)
            if 'diameter' in self._DISTRIBUTION_PREFIXES:
                batch_props_true['diameter'].extend(true_diam_img)
                batch_props_pred['diameter'].extend(pred_diam_img)
            if 'aspect_ratio' in self._DISTRIBUTION_PREFIXES:
                batch_props_true['aspect_ratio'].extend(true_ar_img)
                batch_props_pred['aspect_ratio'].extend(pred_ar_img)
            # if 'circularity' in self._DISTRIBUTION_PREFIXES: ... # Add if re-enabled

        # --- Aggregation and Final Calculations ---
        all_props_true = {k: np.array(v) for k, v in batch_props_true.items()}
        all_props_pred = {k: np.array(v) for k, v in batch_props_pred.items()}

        n_true = float(all_props_true.get('area', np.array([])).size)
        n_pred = float(all_props_pred.get('area', np.array([])).size)
        results['grain_count_true_total'] = n_true
        results['grain_count_pred_total'] = n_pred

        vis_id = f"call_{self.vis_counter}"

        # Calculate distribution similarities
        for prefix in self._DISTRIBUTION_PREFIXES:
            true_dist = all_props_true.get(prefix, np.array([]))
            pred_dist = all_props_pred.get(prefix, np.array([]))
            results.update(self._get_distribution_similarity(true_dist, pred_dist, prefix))

        # <<< Fit Log-normal Distribution to Diameters >>>
        diameter_fit_results = {}
        # Only perform fit if 'diameter' is an active prefix
        if 'diameter' in self._DISTRIBUTION_PREFIXES:
            true_diameters = all_props_true.get('diameter', np.array([]))
            pred_diameters = all_props_pred.get('diameter', np.array([]))
            diameter_fit_results.update(self._fit_lognormal_and_test(true_diameters, "true"))
            diameter_fit_results.update(self._fit_lognormal_and_test(pred_diameters, "pred"))
        results.update(diameter_fit_results)
        # <<< End of Log-normal Fitting >>>

        # Calculate Overall Similarity Score
        weights = {'area': 0.75, 'diameter': 0.2, 'aspect_ratio': 0.05} # Matches current prefixes
        shape_similarity_weighted = 0.0
        total_weight = 0.0
        for prefix in self._DISTRIBUTION_PREFIXES:
             metric_key = f'{prefix}_wasserstein_similarity'
             weight = weights.get(prefix, 0.0)
             if metric_key in results and weight > 0:
                 shape_similarity_weighted += results[metric_key] * weight
                 total_weight += weight
        distribution_similarity_avg = (shape_similarity_weighted / total_weight) if total_weight > self.eps else 0.0

        if n_true <= self.eps and n_pred <= self.eps:
             count_similarity = 1.0
             distribution_similarity_avg = 0.0
        elif n_true <= self.eps or n_pred <= self.eps:
             count_similarity = 0.0
             distribution_similarity_avg = 0.0
        else:
             min_count = min(n_true, n_pred)
             max_count = max(n_true, n_pred)
             count_similarity = min_count / max_count
        results[self._OVERALL_METRIC] = distribution_similarity_avg * count_similarity

        # --- Visualization ---
        if visualize and self.visualization_dir:
            vis_data = {}
            for prefix in self._DISTRIBUTION_PREFIXES:
                 # Adjust key names to match visualizer expectations (plural)
                 key_suffix = 's' if prefix != 'aspect_ratio' else 'ratios' # Handle 'aspect_ratio' pluralization
                 if prefix == 'circularity': key_suffix = 'ies' # Handle 'circularity' pluralization
                 vis_data[f'true_{prefix}{key_suffix}'] = all_props_true.get(prefix, np.array([]))
                 vis_data[f'pred_{prefix}{key_suffix}'] = all_props_pred.get(prefix, np.array([]))

            # Pass the actual fit results to the visualizer
            self._visualize_distributions(vis_data, vis_id, fit_params_dict=diameter_fit_results)

            if visualize_example_image and first_img_labeled_true is not None and first_img_labeled_pred is not None:
                self._visualize_labeled_masks(first_img_labeled_true, first_img_labeled_pred, f"{vis_id}_example_img0")
            elif visualize_example_image:
                 print(f"[{vis_id}] Labeled mask visualization requested, but no valid labeled masks were generated/captured for the first image.")

        self.vis_counter += 1
        return dict(results)