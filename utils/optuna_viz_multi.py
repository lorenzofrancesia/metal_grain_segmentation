import os
import optuna
import yaml
import pandas as pd
import numpy as np
import traceback

# --- Try importing visualization libraries ---
try:
    import matplotlib.pyplot as plt
    from optuna.visualization import (
        plot_param_importances,
        plot_optimization_history,
        plot_slice,
        plot_contour,
        plot_parallel_coordinate,
        plot_pareto_front # Import for multi-objective
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False
    print("Error: Visualization libraries (matplotlib, plotly, kaleido) not found.")
    print("Please install them: pip install matplotlib plotly kaleido")
    # Exit if libraries are essential
    # import sys
    # sys.exit(1)

# --- Main Function ---
def create_visualizations_for_multi_study(
    storage_path,
    study_name,
    output_dir,
    # Provide the names used during optimization (MUST match order in study)
    objective_names=("loss", "dice")
    ):
    """
    Create PNG visualizations for an already finished Multi-Objective Optuna study.

    Args:
        storage_path (str): Path to the SQLite database (e.g., "my_study.db").
        study_name (str): Name of the multi-objective study to visualize.
        output_dir (str): Directory to save the visualizations and summary files.
        objective_names (tuple): Tuple of strings representing the names of the
                                 objectives, in the order they were returned
                                 during optimization.
    """
    if not _VISUALIZATION_AVAILABLE:
        print("Cannot proceed without visualization libraries.")
        return

    print(f"Loading multi-objective study '{study_name}' from {storage_path}")

    # Create output directories
    study_dir = os.path.join(output_dir, study_name) # Create subdir for study
    viz_dir = os.path.join(study_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Load the study
    storage = f"sqlite:///{storage_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    # --- Basic Checks ---
    if not study.directions or len(study.directions) <= 1:
        print(f"Warning: Study '{study_name}' does not appear to be multi-objective "
              f"(found {len(study.directions)} direction(s)). Results may be unexpected.")
        # Optionally exit or proceed with caution
        # return
    if len(study.directions) != len(objective_names):
         print(f"Error: Number of study directions ({len(study.directions)}) does not match "
               f"provided objective_names ({len(objective_names)}). Please provide correct names.")
         return

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Successfully loaded study with {len(study.trials)} trials ({len(completed_trials)} completed).")

    if not completed_trials:
        print("No completed trials found in the study. Cannot generate visualizations.")
        return

    # --- Save Study Summary ---
    print("Saving study summary...")
    study_info = {
        "study_name": study.study_name,
        "objective_names": list(objective_names),
        "objective_directions": [d.name for d in study.directions],
        "is_multi_objective": True, # Explicitly state
        "n_trials_total": len(study.trials),
        "n_completed": len(completed_trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
    }
    try:
        pareto_trials = study.best_trials
        study_info["num_pareto_optimal_trials"] = len(pareto_trials)
        study_info["pareto_optimal_trials_summary"] = [
            {"number": t.number, "values": t.values} for t in pareto_trials
        ]
        print(f"Found {len(pareto_trials)} Pareto optimal trials.")
    except Exception as e:
        print(f"Warning: Could not retrieve Pareto optimal trials for summary: {e}")
        study_info["num_pareto_optimal_trials"] = "Error"

    with open(os.path.join(study_dir, "study_summary.yml"), "w") as outfile:
        yaml.dump(study_info, outfile, default_flow_style=False, sort_keys=False)

    # --- Save Trials Dataframe ---
    try:
        trials_df = study.trials_dataframe(multi_index=True) # Use multi-index for values
        trials_df.to_csv(os.path.join(study_dir, "all_trials.csv"), index=False)
        print(f"All trials data saved to all_trials.csv")
    except Exception as e:
        print(f"Warning: Could not save trials dataframe: {e}")

    # --- Generate Visualizations ---
    print("\nGenerating visualizations (PNG images only)...")
    try:
        plot_kwargs = {
             "font": dict(family="Arial, sans-serif", size=12),
             "margin": dict(l=40, r=20, t=60, b=40)
        }
        top_params_combined = set()
        all_importances = {}
        objective_plot_files = {} # Store generated filenames

        # --- Generate Objective-Specific Plots (History, Importance, Slice, Contour) ---
        for target_idx, target_name in enumerate(objective_names):
            print(f"\nGenerating plots relative to objective: '{target_name}' (Index {target_idx})")
            objective_plot_files[target_name] = {}

            target_func = lambda t, idx=target_idx: t.values[idx] if t.values and len(t.values) > idx else float('nan')

            # 1. Optimization History
            try:
                 history_fig = plot_optimization_history(study, target=target_func, target_name=target_name)
                 history_title = f"Optimization History ({target_name})"
                 history_fig.update_layout(title_text=history_title, title_x=0.5, **plot_kwargs)
                 fname_base = f"optimization_history_{target_name}"
                 history_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                 objective_plot_files[target_name]['history'] = fname_base
                 print(f"  Saved plot: {fname_base}.png")
            except Exception as e: print(f"  Error generating optimization history for {target_name}: {e}")

            # 2. Parameter Importance
            importance_dict = {}
            try:
                importance_dict = optuna.importance.get_param_importances(study, target=target_func)
                all_importances[target_name] = importance_dict

                fig_imp = plot_param_importances(study, target=target_func)
                imp_title = f"Parameter Importance (for {target_name})"
                fig_imp.update_layout(title_text=imp_title, title_x=0.5, **plot_kwargs)
                fname_base = f"parameter_importance_{target_name}"
                fig_imp.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                objective_plot_files[target_name]['importance'] = fname_base
                print(f"  Saved plot: {fname_base}.png")

                current_top_params = list(importance_dict.keys())[:min(5, len(importance_dict))]
                top_params_combined.update(current_top_params)
            except Exception as e: print(f"  Error generating parameter importance for {target_name}: {e}")

        # Determine top params across all objectives
        top_params = sorted(list(top_params_combined))
        print(f"\nTop parameters considered for Slice/Contour plots: {top_params}")

        # --- Generate Slice/Contour using top_params, relative to EACH objective ---
        if top_params:
            objective_plot_files['slice'] = {}
            objective_plot_files['contour'] = {}

            for target_idx, target_name in enumerate(objective_names):
                print(f"\nGenerating Slice/Contour plots relative to '{target_name}'...")
                target_func = lambda t, idx=target_idx: t.values[idx] if t.values and len(t.values) > idx else float('nan')

                for param in top_params:
                    # 3. Slice Plot
                    try:
                        slice_fig = plot_slice(study, params=[param], target=target_func, target_name=target_name)
                        slice_title = f"Slice: {param} (vs {target_name})"
                        slice_fig.update_layout(title_text=slice_title, title_x=0.5, **plot_kwargs)
                        fname_base = f"slice_{param}_vs_{target_name}"
                        slice_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                        if param not in objective_plot_files['slice']: objective_plot_files['slice'][param] = {}
                        objective_plot_files['slice'][param][target_name] = fname_base
                    except Exception as e: print(f"  Error generating slice plot for {param} vs {target_name}: {e}")

                    # 4. Contour Plots
                    for other_param in top_params:
                        if param >= other_param: continue
                        pair_key = tuple(sorted((param, other_param)))
                        try:
                            contour_fig = plot_contour(study, params=[param, other_param], target=target_func, target_name=target_name)
                            contour_title = f"Contour: {param} vs {other_param} (Color: {target_name})"
                            contour_fig.update_layout(title_text=contour_title, title_x=0.5, **plot_kwargs)
                            fname_base = f"contour_{param}_vs_{other_param}_vs_{target_name}"
                            contour_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                            if pair_key not in objective_plot_files['contour']: objective_plot_files['contour'][pair_key] = {}
                            objective_plot_files['contour'][pair_key][target_name] = fname_base
                        except ValueError as ve: print(f"  Skipping contour plot for {param} vs {other_param} vs {target_name}: {ve}")
                        except Exception as e: print(f"  Error generating contour plot for {param} vs {other_param} vs {target_name}: {e}")
                print(f"  Finished Slice/Contour plots for '{target_name}'.")

        # --- Generate Plots NOT specific to a single target objective ---
        # 5. Pareto Front
        try:
            pareto_fig = plot_pareto_front(study, target_names=list(objective_names))
            pareto_fig.update_layout(title_text="Pareto Front", title_x=0.5, **plot_kwargs)
            fname_base = "pareto_front"
            pareto_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
            objective_plot_files['pareto'] = fname_base
            print("\nSaved Pareto front plot.")
        except ValueError as ve: print(f"\nCould not generate Pareto front plot: {ve}")
        except Exception as e: print(f"\nError generating Pareto front plot: {e}")

        # 6. Parallel Coordinate
        if top_params:
            try:
                parallel_fig = plot_parallel_coordinate(study, params=top_params)
                parallel_title = "Parallel Coordinate Plot (Top Params vs Objectives)"
                parallel_fig.update_layout(title_text=parallel_title, title_x=0.5, **plot_kwargs)
                fname_base = "parallel_coordinate"
                parallel_fig.write_image(os.path.join(viz_dir, f"{fname_base}.png"))
                objective_plot_files['parallel'] = fname_base
                print("Saved parallel coordinate plot.")
            except Exception as e: print(f"Error generating parallel coordinate plot: {e}")

        # --- Create HTML index ---
        print("\nCreating HTML index page...")
        create_visualization_index_multi(
            viz_dir,
            study,
            objective_names, # Pass objective names
            top_params,
            objective_plot_files
        )
        print(f"\nVisualizations and index saved to {viz_dir}")

    except Exception as e:
        print(f"\nAn error occurred during visualization generation: {e}")
        traceback.print_exc()

# --- HTML Index Generation Function ---
def create_visualization_index_multi(
    viz_dir,
    study,
    objective_names, # Receive objective names
    top_params,
    objective_plot_files # Receive filenames dict
    ):
    """Create an HTML index page linking to multi-objective PNG visualizations."""

    # --- Basic Info ---
    study_name = study.study_name
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    try:
        directions_str = ', '.join(d.name.lower() for d in study.directions)
    except Exception: directions_str = "Error"
    num_pareto_trials = "N/A"
    if n_completed > 0:
        try: num_pareto_trials = len(study.best_trials)
        except: pass

    summary_rows_html = f"""
        <tr><th>Study Name</th><td>{study_name}</td></tr>
        <tr><th>Objective Names</th><td>{', '.join(objective_names)}</td></tr>
        <tr><th>Directions</th><td>{directions_str}</td></tr>
        <tr><th>Total Trials</th><td>{len(study.trials)}</td></tr>
        <tr><th>Completed</th><td>{n_completed}</td></tr>
        <tr><th>Pruned</th><td>{n_pruned}</td></tr>
        <tr><th>Failed</th><td>{n_failed}</td></tr>
        <tr><th>Pareto Optimal Trials</th><td>{num_pareto_trials}</td></tr>
    """

    # --- Generate HTML for Key Plots ---
    key_plots_html = ""
    if 'pareto' in objective_plot_files:
        fname = objective_plot_files['pareto']
        key_plots_html += f'''<div class="plot-item"><h3>Pareto Front</h3><a href="{fname}.png" target="_blank"><img src="{fname}.png" alt="Pareto Front"></a></div>'''
    if 'parallel' in objective_plot_files:
        fname = objective_plot_files['parallel']
        key_plots_html += f'''<div class="plot-item"><h3>Parallel Coordinate</h3><a href="{fname}.png" target="_blank"><img src="{fname}.png" alt="Parallel Coordinate"></a></div>'''
    for obj_name in objective_names:
         if obj_name in objective_plot_files and 'importance' in objective_plot_files[obj_name]:
             fname = objective_plot_files[obj_name]['importance']
             key_plots_html += f'''<div class="plot-item"><h3>Importance ({obj_name})</h3><a href="{fname}.png" target="_blank"><img src="{fname}.png" alt="Importance for {obj_name}"></a></div>'''
    for obj_name in objective_names:
         if obj_name in objective_plot_files and 'history' in objective_plot_files[obj_name]:
             fname = objective_plot_files[obj_name]['history']
             key_plots_html += f'''<div class="plot-item"><h3>History ({obj_name})</h3><a href="{fname}.png" target="_blank"><img src="{fname}.png" alt="History for {obj_name}"></a></div>'''

    # --- Generate HTML for Slice Plots ---
    slice_plots_html = ""
    if 'slice' in objective_plot_files:
         for param in top_params:
             if param in objective_plot_files['slice']:
                 slice_plots_html += f'<div class="plot-item"><h3>Slice Plots: {param}</h3><div class="plot-subcontainer">'
                 for obj_name in objective_names:
                      if obj_name in objective_plot_files['slice'][param]:
                          fname = objective_plot_files['slice'][param][obj_name]
                          slice_plots_html += f'''<div class="plot-subitem"><h4>vs {obj_name}</h4><a href="{fname}.png" target="_blank"><img src="{fname}.png" alt="Slice: {param} vs {obj_name}"></a></div>'''
                 slice_plots_html += '</div></div>'

    # --- Generate HTML for Contour Plots ---
    contour_plots_html = ""
    if 'contour' in objective_plot_files:
         for pair_key in objective_plot_files['contour']:
             param1, param2 = pair_key
             contour_plots_html += f'<div class="plot-item"><h3>Contour Plots: {param1} vs {param2}</h3><div class="plot-subcontainer">'
             for obj_name in objective_names:
                 if obj_name in objective_plot_files['contour'][pair_key]:
                     fname = objective_plot_files['contour'][pair_key][obj_name]
                     contour_plots_html += f'''<div class="plot-subitem"><h4>Color: {obj_name}</h4><a href="{fname}.png" target="_blank"><img src="{fname}.png" alt="Contour: {param1} vs {param2}, Color: {obj_name}"></a></div>'''
             contour_plots_html += '</div></div>'

    # --- HTML Template ---
    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Optuna Results: {study_name}</title><style>
body{{font-family:Arial,sans-serif;max-width:1600px;margin:10px auto;padding:20px;background-color:#fff;color:#333}}h1,h2{{color:#333;border-bottom:1px solid #eee;padding-bottom:5px}}h1{{font-size:1.8em;text-align:center;margin-bottom:25px}}h2{{font-size:1.4em;margin-top:35px}}h3{{margin:0 0 15px 0;font-size:1.15em;color:#444;text-align:center;border-bottom:1px dashed #ddd;padding-bottom:8px}}h4{{margin:10px 0 5px 0;font-size:1.0em;color:#555;text-align:center}}.section{{margin-bottom:30px;background-color:#fdfdfd;padding:20px;border:1px solid #e0e0e0;border-radius:5px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}}.plot-container{{display:grid;grid-template-columns:repeat(auto-fit, minmax(450px, 1fr));gap:25px}}.plot-item{{border:1px solid #e0e0e0;padding:15px;background-color:#f9f9f9;border-radius:4px;transition:box-shadow 0.2s ease}}.plot-item:hover{{box-shadow:0 3px 8px rgba(0,0,0,0.1)}}.plot-item img{{max-width:100%;height:auto;border:1px solid #ddd;display:block;margin-top:5px;border-radius:3px}}.plot-subcontainer{{display:flex;flex-wrap:wrap;gap:15px;justify-content:center}}.plot-subitem{{flex:1 1 45%;min-width:200px;text-align:center}}a{{color:#0066cc;text-decoration:none}}a:hover{{text-decoration:underline}}table{{border-collapse:collapse;width:100%;margin-top:10px;font-size:0.95em}}th,td{{border:1px solid #ddd;padding:10px 12px;text-align:left;vertical-align:top}}th{{background-color:#f2f2f2;font-weight:bold;width:25%}}code{{background-color:#eee;padding:2px 5px;border-radius:3px;font-family:Consolas,monospace;font-size:0.9em;border:1px solid #ddd}}p{{line-height:1.5}}
</style></head><body><h1>Optuna Multi-Objective Results: <code>{study_name}</code></h1>
<div class="section"><h2>Study Summary</h2><table><tbody>{summary_rows_html}</tbody></table></div>
<div class="section"><h2>Key Visualizations</h2><div class="plot-container">{key_plots_html}</div></div>
<div class="section"><h2>Parameter Slice Plots (vs Objectives)</h2><div class="plot-container">{slice_plots_html}</div></div>
<div class="section"><h2>Parameter Contour Plots (vs Objectives)</h2><div class="plot-container">{contour_plots_html}</div></div>
<div class="section"><h2>Downloads</h2>
<p><a href="../all_trials.csv" target="_blank">Download all trials data summary (CSV)</a></p>
<p><a href="../study_summary.yml" target="_blank">Download study summary (YAML)</a></p>
<p><i>Note: Individual trial details (params, user_attrs) can be found in the database. Best parameters are associated with Pareto optimal trials.</i></p>
</div>
<footer style="text-align:center;margin-top:30px;font-size:0.9em;color:#777;">Generated Visualization Index</footer>
</body></html>"""

    # Write the HTML file
    try:
        with open(os.path.join(viz_dir, "index.html"), "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"Created HTML index page: {os.path.join(viz_dir, 'index.html')}")
    except Exception as e:
         print(f"Error writing visualization index.html: {e}")


# --- Example Usage ---
if __name__ == "__main__":

    # --- Configuration ---
    # SET THESE VALUES FOR YOUR STUDY
    STUDY_DB_PATH = r"C:\Users\lorenzo.francesia\Documents\github\runs\optimization_studies.db"
    STUDY_NAME_TO_VISUALIZE = "unetplusplus_FT_lossdice_resnet101_study"
    BASE_OUTPUT_DIR = r"C:\Users\lorenzo.francesia\Documents\github\runs\unetplusplus_FT_lossdice_resnet101_study"
    OBJECTIVE_NAMES_TUPLE = ("val_loss", "dice") 


    # --- Run the visualization ---
    print("-" * 50)
    print(f"Starting visualization for study: {STUDY_NAME_TO_VISUALIZE}")
    print(f"Database: {STUDY_DB_PATH}")
    print(f"Output Dir: {os.path.join(BASE_OUTPUT_DIR, STUDY_NAME_TO_VISUALIZE)}")
    print(f"Objectives: {OBJECTIVE_NAMES_TUPLE}")
    print("-" * 50)

    # Check if DB exists
    if not os.path.exists(STUDY_DB_PATH):
        print(f"Error: Database file not found at {STUDY_DB_PATH}")
    else:
        create_visualizations_for_multi_study(
            storage_path=STUDY_DB_PATH,
            study_name=STUDY_NAME_TO_VISUALIZE,
            output_dir=BASE_OUTPUT_DIR, # Pass the base dir here
            objective_names=OBJECTIVE_NAMES_TUPLE
        )
        print("-" * 50)
        print("Visualization script finished.")