import os
import optuna
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_slice, plot_contour, plot_parallel_coordinate

def create_visualizations_for_study(storage_path, study_name, output_dir):
    """
    Create visualizations for an already finished Optuna study.
    
    Args:
        storage_path (str): Path to the SQLite database containing the study.
        study_name (str): Name of the study to visualize.
        output_dir (str): Directory to save the visualizations.
    """
    print(f"Loading study '{study_name}' from {storage_path}")
    
    # Create output directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load the study
    storage = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    print(f"Successfully loaded study with {len(study.trials)} trials")
    print(f"Best trial: #{study.best_trial.number} with value: {study.best_value}")
    
    # Save best parameters
    best_params = study.best_params
    with open(os.path.join(output_dir, "best_hyperparameters.yml"), "w") as outfile:
        yaml.dump(best_params, outfile, default_flow_style=False)
    
    # Save overall study summary
    study_info = {
        "study_name": study.study_name,
        "direction": study.direction,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }
    
    with open(os.path.join(output_dir, "study_summary.yml"), "w") as outfile:
        yaml.dump(study_info, outfile, default_flow_style=False)
    
    # Create a CSV with all trials information for easy analysis
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(output_dir, "all_trials.csv"))
    
    try:
        # Parameter importance visualization
        print("Generating parameter importance visualization...")
        fig = plot_param_importances(study)
        fig.update_layout(
            width=900, 
            height=600,
            margin=dict(l=20, r=20, t=30, b=20),
            title_text="Parameter Importance",
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=14)
        )
        fig.write_image(os.path.join(viz_dir, "parameter_importance.png"))
        fig.write_html(os.path.join(viz_dir, "parameter_importance.html"))
        
        # Optimization history plot
        print("Generating optimization history plot...")
        history_fig = plot_optimization_history(study)
        history_fig.update_layout(
            width=900, 
            height=500,
            margin=dict(l=20, r=20, t=30, b=20),
            title_text="Optimization History",
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=14)
        )
        history_fig.write_image(os.path.join(viz_dir, "optimization_history.png"))
        history_fig.write_html(os.path.join(viz_dir, "optimization_history.html"))
        
        # Get parameter importance and select top parameters
        importance = optuna.importance.get_param_importances(study)
        top_params = list(importance.keys())[:min(5, len(importance))]
        
        # Generate slice plots for the most important parameters
        print("Generating slice plots for top parameters...")
        for param in top_params:
            try:
                slice_fig = plot_slice(study, params=[param])
                slice_fig.update_layout(
                    width=800, 
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title_text=f"Slice Plot for {param}",
                    title_x=0.5,
                    font=dict(family="Arial, sans-serif", size=14)
                )
                slice_fig.write_image(os.path.join(viz_dir, f"slice_{param}.png"))
                slice_fig.write_html(os.path.join(viz_dir, f"slice_{param}.html"))
            except Exception as e:
                print(f"Error generating slice plot for {param}: {e}")
        
        # Generate contour plots for pairs of top parameters
        print("Generating contour plots for parameter pairs...")
        for i, param1 in enumerate(top_params):
            for param2 in top_params[i+1:]:  # Avoid duplicates
                try:
                    contour_fig = plot_contour(study, params=[param1, param2])
                    contour_fig.update_layout(
                        width=800, 
                        height=600,
                        margin=dict(l=20, r=20, t=40, b=20),
                        title_text=f"Contour Plot: {param1} vs {param2}",
                        title_x=0.5,
                        font=dict(family="Arial, sans-serif", size=14)
                    )
                    contour_fig.write_image(os.path.join(viz_dir, f"contour_{param1}_vs_{param2}.png"))
                    contour_fig.write_html(os.path.join(viz_dir, f"contour_{param1}_vs_{param2}.html"))
                except Exception as e:
                    print(f"Error generating contour plot for {param1} vs {param2}: {e}")
        
        # Parallel coordinate plot for top parameters
        print("Generating parallel coordinate plot...")
        try:
            parallel_fig = plot_parallel_coordinate(study, params=top_params)
            parallel_fig.update_layout(
                width=1000, 
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                title_text="Parallel Coordinate Plot for Top Parameters",
                title_x=0.5,
                font=dict(family="Arial, sans-serif", size=14)
            )
            parallel_fig.write_image(os.path.join(viz_dir, "parallel_coordinate.png"))
            parallel_fig.write_html(os.path.join(viz_dir, "parallel_coordinate.html"))
        except Exception as e:
            print(f"Error generating parallel coordinate plot: {e}")
        
        # Create a summary visualization HTML page
        create_visualization_index(viz_dir, study, top_params)
        print(f"Visualizations successfully saved to {viz_dir}")
        
    except ImportError as e:
        print(f"Could not generate visualization plots. Make sure matplotlib and plotly are installed. Error: {e}")
    except Exception as e:
        print(f"Error generating visualization: {e}")

def create_visualization_index(viz_dir, study, top_params):
    """Create an HTML index page for all visualizations."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Optuna Optimization Results - {study_name}</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
        }}
        h1, h2 {{ 
            color: #333; 
        }}
        .section {{ 
            margin-bottom: 30px; 
        }}
        .plot-container {{ 
            display: flex; 
            flex-wrap: wrap; 
            gap: 20px; 
        }}
        .plot-item {{ 
            margin-bottom: 20px; 
        }}
        .plot-item img {{ 
            max-width: 100%; 
            height: auto; 
            border: 1px solid #ddd; 
        }}
        .plot-item h3 {{ 
            margin: 10px 0; 
        }}
        a {{ 
            color: #0066cc; 
            text-decoration: none; 
        }}
        a:hover {{ 
            text-decoration: underline; 
        }}
    </style>
</head>
<body>
    <h1>Optuna Hyperparameter Optimization Results - {study_name}</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>Study Name: {study_name}</p>
        <p>Best Trial: #{best_trial} (Value: {best_value:.6f})</p>
        <p>Total Trials: {n_trials} (Completed: {n_completed}, Pruned: {n_pruned})</p>
    </div>
    
    <div class="section">
        <h2>Key Visualizations</h2>
        <div class="plot-container">
            <div class="plot-item">
                <h3>Parameter Importance</h3>
                <a href="parameter_importance.html" target="_blank">
                    <img src="parameter_importance.png" alt="Parameter Importance">
                </a>
            </div>
            <div class="plot-item">
                <h3>Optimization History</h3>
                <a href="optimization_history.html" target="_blank">
                    <img src="optimization_history.png" alt="Optimization History">
                </a>
            </div>
            <div class="plot-item">
                <h3>Parallel Coordinate Plot</h3>
                <a href="parallel_coordinate.html" target="_blank">
                    <img src="parallel_coordinate.png" alt="Parallel Coordinate Plot">
                </a>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Parameter Slice Plots</h2>
        <div class="plot-container">
            {slice_plots}
        </div>
    </div>
    
    <div class="section">
        <h2>Parameter Contour Plots</h2>
        <div class="plot-container">
            {contour_plots}
        </div>
    </div>
    
    <div class="section">
        <h2>Trial Data</h2>
        <p>Individual trial data is stored in the trial_X directories.</p>
        <p><a href="../all_trials.csv" target="_blank">Download all trials data (CSV)</a></p>
    </div>
</body>
</html>
"""
    
    # Generate slice plot HTML
    slice_plots_html = ""
    for param in top_params:
        plot_path = f"slice_{param}.png"
        if os.path.exists(os.path.join(viz_dir, plot_path)):
            slice_plots_html += f"""
            <div class="plot-item">
                <h3>Slice Plot: {param}</h3>
                <a href="slice_{param}.html" target="_blank">
                    <img src="{plot_path}" alt="Slice Plot for {param}">
                </a>
            </div>
            """
    
    # Generate contour plot HTML
    contour_plots_html = ""
    for i, param1 in enumerate(top_params):
        for param2 in top_params[i+1:]:
            plot_path = f"contour_{param1}_vs_{param2}.png"
            if os.path.exists(os.path.join(viz_dir, plot_path)):
                contour_plots_html += f"""
                <div class="plot-item">
                    <h3>Contour Plot: {param1} vs {param2}</h3>
                    <a href="contour_{param1}_vs_{param2}.html" target="_blank">
                        <img src="{plot_path}" alt="Contour Plot for {param1} vs {param2}">
                    </a>
                </div>
                """
    
    # Fill in the template
    formatted_html = html_content.format(
        study_name=study.study_name,
        best_trial=study.best_trial.number,
        best_value=study.best_value,
        n_trials=len(study.trials),
        n_completed=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        n_pruned=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        slice_plots=slice_plots_html,
        contour_plots=contour_plots_html
    )
    
    # Write the HTML file
    with open(os.path.join(viz_dir, "index.html"), "w") as f:
        f.write(formatted_html)
    print(f"Created visualization index page at {os.path.join(viz_dir, 'index.html')}")

if __name__ == "__main__":

    # create_visualizations_for_study(study_name="unet_bce_study",
    #                                 storage_path="C://Users//lorenzo.francesia//Documents//github//runs//optimization_studies_test.db",
    #                                 output_dir="C:\\Users\\lorenzo.francesia\\Documents\\github\\runs\\unet_bce_study")
    
    create_visualizations_for_study(study_name="gan_optimization_short",
                                    storage_path="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\gandb\\optuna_gan.db",
                                    output_dir="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\gandb\\viz")