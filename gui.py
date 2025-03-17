import tkinter as tk
import tkinter.ttk as ttk
import customtkinter as ctk
from tkinter import filedialog, messagebox, scrolledtext
import json
import subprocess
import threading
import sys
import queue
import os
import re
import time
from PIL import Image
import torch
from torchvision.transforms import transforms
from data.dataset import SegmentationDataset
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.pyplot as plt

class StringRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.display_output(string)

    def flush(self):
        pass

class TransformWidget(ctk.CTkFrame):
    def __init__(self, master, available_transforms=None, update_callback=None):
        super().__init__(master)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.available_transforms = available_transforms or ["transforms.ToTensor", "transforms.Resize"]
        self.update_callback = update_callback  

        self.label = ctk.CTkLabel(self, text="Transform")
        self.label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.add_btn = ctk.CTkButton(self, text="+", width=30, command=self.add_row)
        self.add_btn.grid(row=0, column=1, padx=5, pady=5, sticky="e")

        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.table_frame.grid_columnconfigure(1, weight=1) # Key change here

        self.rows = []
        self.add_row(default="transforms.Resize", default_size="(512,512)")
        self.add_row(default="transforms.ToTensor")
        
        self._update_widget()

    def add_row(self, default=None, default_size=None):
        row_frame = ctk.CTkFrame(self.table_frame)
        row_frame.pack(fill="x", padx=0, pady=2) # Remove padding here
        row_frame.grid_columnconfigure(1, weight=1)

        delete_btn = ctk.CTkButton(row_frame, text="X", width=30, command=lambda: self.delete_row(row_frame))
        delete_btn.grid(row=0, column=0, padx=(0,2)) # use grid for more precise control
        delete_btn.grid_propagate(False) # Prevent button from affecting row size


        dropdown = ctk.CTkOptionMenu(row_frame, values=self.available_transforms, command=lambda value: self.on_dropdown_change(value, row_frame))
        dropdown.grid(row=0, column=1, sticky="ew") # Use grid and sticky to fill space

        if default:
            dropdown.set(default)

        resize_entry = ctk.CTkEntry(row_frame, placeholder_text="Enter size", width=75)
        resize_entry.grid(row=0, column=2, padx=5)

        if default == "transforms.Resize":
             resize_entry.configure(state="normal")
             if default_size:
                resize_entry.insert(0, default_size)
        else:
            resize_entry.grid_remove()
            resize_entry.configure(state="disabled")
            
        resize_entry.bind("<KeyRelease>", lambda event, item={"dropdown": dropdown, "resize_entry": resize_entry}: self._entry_changed(item))

        self.rows.append({"frame": row_frame, "dropdown": dropdown, "resize_entry": resize_entry, "delete_btn": delete_btn})
        self._update_widget()

    def on_dropdown_change(self, value, row_frame):
        for item in self.rows:
            if item["frame"] == row_frame:
                if value == "transforms.Resize":
                    item["resize_entry"].grid()  # Use grid to show
                    item["resize_entry"].configure(state="normal")
                else:
                    item["resize_entry"].grid_remove()  # Use grid_remove to hide
                    item["resize_entry"].configure(state="disabled")
                break

    def delete_row(self, frame):
        self.rows = [item for item in self.rows if item["frame"] != frame]
        frame.destroy()
        self._update_widget()

    def get_sequence(self):
        sequence = []
        for item in self.rows:
            transform = item["dropdown"].get()
            if transform == "transforms.Resize" and item["resize_entry"].cget("state") == "normal":
                size = item["resize_entry"].get()
                sequence.append(f"{transform}({size})")
            else:
                sequence.append(f"{transform}()")
        return sequence
    
    def load_config(self, config_string):
        # Clear existing rows
        for item in self.rows:
            item["frame"].destroy()
        self.rows = []

        # Parse the config string
        try:
            # Ensure config_string is a string representation of a list
            if isinstance(config_string, list):
                config_string = str(config_string)  # Convert list to its string representation

            transforms = eval(config_string)  # Use eval() with caution!
            for transform_str in transforms:
                match = re.match(r"(\w+\.\w+)\((.*)\)", transform_str)
                if match:
                    transform_name = match.group(1)
                    if transform_name not in self.available_transforms:
                        print(f"Warning: Transform '{transform_name}' is not in available_transforms.")
                        continue

                    if transform_name == "transforms.Resize":
                        args_str = match.group(2)
                        
                        self.add_row(default=transform_name, default_size=args_str)
                    else:
                        self.add_row(default=transform_name)

                else:
                    print(f"Warning: Could not parse transform string: '{transform_str}'")
            self._update_widget()

        except (SyntaxError, ValueError, TypeError) as e:
            print(f"Error: Invalid config string format: {e}")
    
    def _update_widget(self):
        if self.update_callback:
            self.update_callback(self.get_sequence())
            
    def _entry_changed(self, item):
        """Updates the transform sequence when the entry changes."""
        if item["dropdown"].get() == "transforms.Resize":
            try:  # Basic validation
                eval(item["resize_entry"].get()) # Check if it's valid Python syntax (tuple)
                self._update_widget() 
            except Exception:
                pass  

class PlotView(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Set dark style for plots
        plt.style.use('dark_background')

        # Create tabs
        self.add("Losses")
        self.add("Image Evolution")
        self.add("Dataset")

        # --- Losses Tab Setup ---
        self.loss_figure = Figure(figsize=(5, 4), dpi=100)
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_figure, master=self.tab("Losses"))
        self.loss_canvas.draw()
        self.loss_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.loss_data = []  # Store loss values

        # --- Image Evolution Tab Setup ---
        self.image_frame = ctk.CTkFrame(self.tab("Image Evolution"))
        self.image_frame.grid(row=0, column=0, sticky="nsew")

        self.image_figure = Figure(figsize=(5, 4), dpi=100)
        self.image_ax = self.image_figure.add_subplot(111)
        self.image_canvas = FigureCanvasTkAgg(self.image_figure, master=self.image_frame)
        self.image_canvas.draw()
        self.image_canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)

        # Navigation buttons
        self.button_frame = ctk.CTkFrame(self.image_frame)
        self.button_frame.pack(side=ctk.BOTTOM, fill=ctk.X)

        self.prev_button = ctk.CTkButton(self.button_frame, text="< Prev", command=self.show_previous_image)
        self.prev_button.pack(side=ctk.LEFT, padx=5, pady=5)

        self.next_button = ctk.CTkButton(self.button_frame, text="Next >", command=self.show_next_image)
        self.next_button.pack(side=ctk.LEFT, padx=5, pady=5)

        self.image_dir = None  # Directory to watch for images
        self.current_image_index = 0  # Index of the currently displayed image
        self.image_files = []  # List of image files in the directory

        # --- Dataset Tab Setup (using matplotlib) ---
        self.dataset_figure = Figure(figsize=(6, 5), dpi=100) # Increased figure size for text
        self.dataset_ax = self.dataset_figure.add_subplot(111) # Initialize axes for dataset images
        self.dataset_canvas = FigureCanvasTkAgg(self.dataset_figure, master=self.tab("Dataset"))
        self.dataset_canvas.draw()
        self.dataset_canvas_widget = self.dataset_canvas.get_tk_widget() # Store the widget
        self.dataset_canvas_widget.grid(row=0, column=0, sticky="nsew") # Grid place the widget

        # Configure grid weights for resizing
        self.tab("Losses").grid_rowconfigure(0, weight=1)
        self.tab("Losses").grid_columnconfigure(0, weight=1)
        self.tab("Image Evolution").grid_rowconfigure(0, weight=1)
        self.tab("Image Evolution").grid_columnconfigure(0, weight=1)
        self.tab("Dataset").grid_rowconfigure(0, weight=1)
        self.tab("Dataset").grid_columnconfigure(0, weight=1)

        # Clear plots on initialization
        self.clear_plots()

    def set_image_dir(self, image_dir):
        """Sets the directory to watch for images and updates the image list."""
        self.image_dir = image_dir
        self.update_image_list()
        self.current_image_index = len(self.image_files) - 1  # Point to the last image
        self.show_current_image()  # Display the last image in the new directory

    def update_image_list(self):
        """Updates the list of image files from the current directory."""
        if self.image_dir:
            image_files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Sort files numerically based on the first number in the filename
            def sort_key(filename):
                match = re.search(r'\d+', filename)
                if match:
                    return int(match.group(0))
                return 0  # Default sort key if no number is found

            self.image_files = sorted(image_files, key=sort_key)
        else:
            self.image_files = []

    def show_previous_image(self):
        """Displays the previous image in the list."""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

    def show_next_image(self):
        """Displays the next image in the list."""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_current_image()

    def show_current_image(self):
        """Displays the image at the current index."""
        if not self.image_files:
            # print("No images to display.")
            self.image_ax.clear()
            self.image_ax.set_title("Image Evolution")
            self.image_ax.axis("off")
            self.image_canvas.draw()
            return

        # Ensure the index is within valid bounds
        self.current_image_index = max(0, min(self.current_image_index, len(self.image_files) - 1))

        image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
        try:
            img = Image.open(image_path)
            self.image_ax.clear()
            self.image_ax.imshow(img)
            self.image_ax.set_title(f"Image Evolution - Epoch {self.image_files[self.current_image_index][:-4]}")
            self.image_ax.axis("off")
            self.image_canvas.draw()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def plot_loss(self, train_loss_values, val_loss_values):
        """Plots the training and validation loss values on the Losses tab."""
        self.loss_ax.clear()

        if train_loss_values:
            # Use x+1 for epoch index when plotting
            self.loss_ax.plot(range(1, len(train_loss_values) + 1), train_loss_values, label="Train Loss")
        if val_loss_values:
            # Use x+1 for epoch index when plotting
            self.loss_ax.plot(range(1, len(val_loss_values) + 1), val_loss_values, label="Val Loss")

        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.set_title("Training and Validation Loss")
        self.loss_ax.legend()

        # Force integer ticks on x-axis
        self.loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.loss_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

        self.loss_canvas.draw()

    def clear_plots(self):
        """Clears the plots in both tabs."""
        self.loss_ax.clear()
        self.loss_canvas.draw()
        self.image_ax.clear()
        self.image_ax.set_title("Image Evolution")
        self.image_ax.axis("off")
        self.image_canvas.draw()

    def show_empty_image_plot(self):
        """Displays an empty plot for Image Evolution."""
        self.image_ax.clear()
        self.image_ax.set_title("Image Evolution")
        self.image_ax.axis("off")
        self.image_canvas.draw()

    def reset_image_data(self):
        """Resets image-related data for a new training session."""
        self.image_dir = None
        self.current_image_index = 0
        self.image_files = []
        self.clear_plots()

    def visualize_dataset(self, dataset, idx=None):
        """Visualizes dataset samples in the Dataset tab using matplotlib, showing both image and mask."""
        self.switch_to_tab("Dataset")
        self.dataset_ax.clear()  # Clear the *single* axes we were using before

        # Create subplots - now we need 2 axes within the dataset_figure
        self.dataset_figure.clf() # Clear the entire figure to reset subplots
        image_ax = self.dataset_figure.add_subplot(1, 2, 1)  # 1 row, 2 columns, first subplot for image
        mask_ax = self.dataset_figure.add_subplot(1, 2, 2)   # 1 row, 2 columns, second subplot for mask

        if idx is None:
            self.currentidx = np.random.randint(0, len(dataset))
        else:
            self.currentidx = idx
        image, mask = dataset[self.currentidx]

        image_np = image.permute(1, 2, 0).numpy() if isinstance(image, torch.Tensor) else image
        mask_np = mask.squeeze().numpy() if isinstance(mask, torch.Tensor) else mask

        if image_np.max() > 1:
            image_np = image_np / image_np.max()

        # Display Image in the first subplot
        image_ax.imshow(image_np)
        image_ax.set_title(f"Image - Index: {idx}")
        image_ax.axis('off')

        # Display Mask in the second subplot
        mask_ax.imshow(mask_np, cmap='gray')  # Use 'jet' or another suitable colormap for masks
        mask_ax.set_title(f"Mask - Index: {idx}")
        mask_ax.axis('off')

        self.dataset_figure.tight_layout() # Adjust layout to prevent overlap
        self.dataset_canvas.draw()  # Update the canvas

    def class_distribution(self, dataset, classes_names=['Background', 'Foreground']):
        """Calculates and plots the class distribution with percentages in the Dataset tab using matplotlib."""
        self.switch_to_tab("Dataset")
        self.dataset_ax.clear()
        self.dataset_figure.clf()  # Clear the entire figure
        ax = self.dataset_figure.add_subplot(111)  # Add a single subplot

        total_pixels = 0
        foreground_pixels = 0

        for _, mask in dataset:
            mask = mask.permute(1, 2, 0).numpy()
            total_pixels += mask.shape[0] * mask.shape[1]
            foreground_pixels += (mask == 1).sum()

        background_pixels = total_pixels - foreground_pixels

        counts = [background_pixels, foreground_pixels]
        percentages = [background_pixels / total_pixels * 100, foreground_pixels / total_pixels * 100]

        bars = ax.bar(classes_names, counts, color=['blue', 'orange'])
        ax.set_ylabel('Pixel Count')
        ax.set_title('Class Distribution')

        # Add percentage labels on top of each bar
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{percentage:.2f}%",
                    ha="center", va="bottom", fontsize=10, color="white") # Changed color to white for better visibility on dark background

        self.dataset_figure.tight_layout()
        self.dataset_canvas.draw()

    def visualize_overlay(self, dataset, idx=None, alpha=0.35):
        """Visualizes the overlay of an image and its mask in the Dataset tab using matplotlib."""
        self.switch_to_tab("Dataset")
        self.dataset_ax.clear()
        self.dataset_figure.clf() # Clear the entire figure
        ax = self.dataset_figure.add_subplot(111) # Add a single subplot

        if idx is None:
            idx = np.random.randint(0, len(dataset))

        image, mask = dataset[idx]
        image = image.permute(1, 2, 0).numpy() if isinstance(image, torch.Tensor) else image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        ax.imshow(image, alpha=1.0)
        ax.imshow(mask.squeeze(0), cmap="gray", alpha=alpha)
        ax.axis("off")
        ax.set_title(f"Overlay of Image and Mask [Index: {idx}]")

        self.dataset_figure.tight_layout()
        self.dataset_canvas.draw()

    def image_histogram(self, dataset, idx=None):
        """Calculates and plots the histogram of pixel values in an image on the Dataset tab using matplotlib."""
        self.switch_to_tab("Dataset")
        self.dataset_ax.clear()
        self.dataset_figure.clf() # Clear the entire figure
        ax = self.dataset_figure.add_subplot(111) # Add a single subplot
        if idx is None:
            idx = np.random.randint(0, len(dataset))

        image, mask = dataset[idx]
        image = image.permute(1, 2, 0).numpy() if isinstance(image, torch.Tensor) else image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pixel_values = image.flatten()

        ax.hist(pixel_values, bins=256, range=(0, 256), color='blue', alpha=0.7)
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Pixel Values')

        self.dataset_figure.tight_layout()
        self.dataset_canvas.draw()

    def clear_dataset_frame(self):
        """Clears ONLY the matplotlib plot area in Dataset tab."""
        self.dataset_ax.clear()  # Clear matplotlib axes
        self.dataset_figure.clf() # Clear matplotlib figure
        self.dataset_ax.set_facecolor('black') # Keep black background
        self.dataset_ax.axis("off") # Ensure axes are off

    def switch_to_tab(self, tab_name):
        """Switches the view to the specified tab."""
        self.set(tab_name)

class ModeltrainingGUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Model training GUI")
        self.window.geometry("900x550")

        self._init_variables()

        self.message_queue = queue.Queue()  # Create a message queue
        self.previous_line_was_progress_flag = False # Initialize the flag here
        threading.Thread(target=self.process_queue_continuous, daemon=True).start()

        self.int_validation = (self.window.register(self.validate_int_input), '%P')
        self.num_validation = (self.window.register(self.validate_numeric_input), '%P')
        self.create_gui()
        
        # Clear plots on GUI initialization
        self.plot_panel.clear_plots()
        self.plot_panel.show_empty_image_plot()

    def _init_variables(self):
        self.model_var = tk.StringVar(value="U-Net")
        self.attention_var = tk.StringVar(value="None")
        self.batchnorm_var = tk.StringVar(value="True")
        
        self.encoder_var = tk.StringVar(value="resnet50")
        self.pretrained_weights_var = tk.BooleanVar(value=False)
        self.freeze_backbone_var = tk.BooleanVar(value=False)

        self.optimizer_var = tk.StringVar(value="AdamW")
        self.lr_var = tk.StringVar(value="0.0001")
        self.momentum_var = tk.StringVar(value="0.9")
        self.weight_decay_var = tk.StringVar(value="1e-4")
        
        self.warmup_scheduler_var = ctk.StringVar(value="None")
        self.warmup_steps_var = tk.StringVar(value="3")

        self.scheduler_var = tk.StringVar(value="None")
        self.start_factor_var = tk.StringVar(value="1.0")
        self.end_factor_var = tk.StringVar(value="0.3")
        self.iterations_var = tk.StringVar(value="10")
        self.t_max_var = tk.StringVar(value="10")
        self.eta_min_var = tk.StringVar(value="0")
        self.step_size_var = tk.StringVar(value="5")
        self.gamma_lr_var = tk.StringVar(value="0.5")

        self.loss_function_var = tk.StringVar(value="FocalTversky")
        self.loss_function1_var = tk.StringVar(value="FocalTversky")
        self.loss_function2_var = tk.StringVar(value="FocalTversky")
        self.loss_function1_weight_var = tk.StringVar(value="0.5")
        self.loss_function2_weight_var = tk.StringVar(value="0.5")
        self.alpha_var = tk.StringVar(value="0.7")
        self.beta_var = tk.StringVar(value="0.3")
        self.gamma_var = tk.StringVar(value="1.3333")
        self.positive_weight_var = tk.StringVar(value="1")
        self.topoloss_patch_var = tk.StringVar(value="64")
        self.alpha_focal_var = tk.StringVar(value="0.8")
        self.gamma_focal_var = tk.StringVar(value="2")

        self.epochs_var = tk.StringVar(value="1")
        self.batch_size_var = tk.StringVar(value="6")
        self.data_dir_var = tk.StringVar(value="c:\\Users\\lorenzo.francesia\\Documents\\github\\data")
        self.output_dir_var = tk.StringVar(value="c:\\Users\\lorenzo.francesia\\Documents\\github\\runs")
        self.normalize_var = tk.BooleanVar(value=False)
        self.negative_var = tk.BooleanVar(value=False)
        self.transform_var = tk.StringVar(value="")
        
        # testing variables
        self.test_model_path_var = tk.StringVar(value="")
        self.test_data_dir_var = tk.StringVar(value="c:\\Users\\lorenzo.francesia\\Documents\\github\\data\\test")
        self.test_batch_size_var = tk.StringVar(value="6")
        self.test_normalize_var = tk.BooleanVar(value=False)
        self.test_negative_var = tk.BooleanVar(value=False)
        
        # Dataset tab index entry
        self.dataset_dir_var = tk.StringVar(value="c:\\Users\\lorenzo.francesia\\Documents\\github\\data\\val")
        self.dataset_index_var = tk.StringVar() 
        

    def validate_numeric_input(self, new_value):
        if not new_value:
            return True
        try:
            float(new_value)
            return True
        except ValueError:
            return False

    def validate_int_input(self, new_value):
        if not new_value:
            return True
        try:
            int(new_value)
            return True
        except ValueError:
            return False

    def browse_dir(self, entry):
        file_path = filedialog.askdirectory()
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)
            
    def browse_file(self, entry):
        file_path = filedialog.askopenfilename()
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)
    
    def log_message(self, message):
        if self.messages_box:
            self.messages_box.config(state=tk.NORMAL)
            self.messages_box.insert(tk.END, message + "\n")
            self.messages_box.see(tk.END)
            self.messages_box.config(state=tk.DISABLED)
            self.messages_box.update()
            time.sleep(0.05)  # Increased delay to 0.05 seconds in log_message
            #self.window.update_idletasks() # no need for this here, update() is stronger

    def save_config(self):
        """Saves the configuration dynamically."""
        config = {}
        for name, value in self.__dict__.items():
            if not "test" in name and not "dataset" in name:
                if name.endswith("_var") and isinstance(value, (tk.StringVar, tk.IntVar, tk.BooleanVar, ctk.StringVar, ctk.IntVar, ctk.BooleanVar)):  # Add customtkinter types
                    config[name.replace("_var", "")] = value.get()

        # Add any other configurations that aren't Tkinter variables.
        if hasattr(self, 'transforms_widget'):
            config['transform'] = self.transforms_widget.get_sequence()

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
                
    def load_config(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path:  # User cancelled the file dialog
            return
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            # Iterate through the loaded configuration
            for key, value in config.items():
                var_name = key + "_var"  # Construct the expected variable name
                if hasattr(self, var_name):  # Check if the attribute exists
                    var = getattr(self, var_name)  # Get the Tkinter variable
                    if isinstance(var, (tk.StringVar, tk.IntVar, tk.BooleanVar, ctk.StringVar, ctk.IntVar, ctk.BooleanVar)):
                        var.set(value)  # Set the value
                    else:
                        print(f"Warning: Attribute {var_name} is not a Tkinter variable.")
                else:
                    print(f"Warning: No matching variable found for config key: {key}")
               
            if 'transform' in config and hasattr(self, 'transforms_widget'):
                self.transforms_widget.load_config(config_string=config.get("transform", ""))
                     
        except FileNotFoundError:
            messagebox.showerror("Error", "Configuration file not found.")
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON format in the configuration file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
                     
    def toggle_options(self, frame, row, button, update_function=None):

        if frame.winfo_ismapped():
            frame.grid_forget()
            button.configure(text="\u25BC")
        else:
            frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=5, pady=5)
            button.configure(text="\u25B2")
            if update_function:
                update_function()

    def update_model_options(self, calling_frame, *args):
        for widget in calling_frame.winfo_children():
            widget.destroy()

        if self.model_var.get() == "U-Net" or self.model_var.get() == "U-Net++":
            ctk.CTkLabel(calling_frame, text="Attention").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkComboBox(calling_frame, values=["None","scse"], variable=self.attention_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Batchnorm").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkComboBox(calling_frame, values=["True","inplace", "False"], variable=self.batchnorm_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        else:
            ctk.CTkLabel(calling_frame, text="BatchNorm").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkComboBox(calling_frame, values=["True","inplace", "False"], variable=self.batchnorm_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
    def update_optimizer_options(self, *args):
        for widget in self.optimizer_options_frame.winfo_children():
            widget.destroy()

        if self.optimizer_var.get() == "Adam" or self.optimizer_var.get() == "AdamW":
            ctk.CTkLabel(self.optimizer_options_frame, text="Learning Rate").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.lr_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.optimizer_options_frame, text="Momentum").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.momentum_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(self.optimizer_options_frame, text="Weight Decay").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.weight_decay_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        elif self.optimizer_var.get() == "SGD":
            ctk.CTkLabel(self.optimizer_options_frame, text="Learning Rate").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.lr_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(self.optimizer_options_frame, text="Momentum").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.momentum_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(self.optimizer_options_frame, text="Weight Decay").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.weight_decay_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    def update_warmup_options(self, *args):
        """Updates the options displayed based on the selected warmup scheduler."""
        # Destroy all existing widgets in the warmup options frame.
        for widget in self.warmup_options_frame.winfo_children():
            widget.destroy()

        if self.warmup_scheduler_var.get() == "Linear":
            ctk.CTkLabel(self.warmup_options_frame, text="Warmup Steps").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.warmup_options_frame, textvariable=self.warmup_steps_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        else:
            ctk.CTkLabel(self.warmup_options_frame, text="No options for selected scheduler").grid(row=0, column=0, columnspan=3, sticky="e", padx=5, pady=5)
            
    def update_scheduler_options(self, *args):
        for widget in self.scheduler_options_frame.winfo_children():
            widget.destroy()

        if self.scheduler_var.get() == "LinearLR":
            ctk.CTkLabel(self.scheduler_options_frame, text="Start Factor").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.start_factor_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.scheduler_options_frame, text="End Factor").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.end_factor_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.scheduler_options_frame, text="Iterations").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.iterations_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        elif self.scheduler_var.get() == "CosineAnnealingLR":
            ctk.CTkLabel(self.scheduler_options_frame, text="T max").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.t_max_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.scheduler_options_frame, text="Eta min").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.eta_min_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
        elif self.scheduler_var.get() == "StepLR":
            ctk.CTkLabel(self.scheduler_options_frame, text="Step Size").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.step_size_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.scheduler_options_frame, text="Gamma").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.gamma_lr_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        else:
            ctk.CTkLabel(self.scheduler_options_frame, text="No options for selected scheduler").grid(row=0, column=0, columnspan=3, sticky="e", padx=5, pady=5)

    def update_loss_function_options(self, calling_frame, *args):
        # Remove existing traces for submenus if they exist
        if hasattr(self, 'loss_func1_trace_id') and self.loss_func1_trace_id:
            self.loss_function1_var.trace_remove("write", self.loss_func1_trace_id)
            self.loss_func1_trace_id = None
        if hasattr(self, 'loss_func2_trace_id') and self.loss_func2_trace_id:
            self.loss_function2_var.trace_remove("write", self.loss_func2_trace_id)
            self.loss_func2_trace_id = None
        
        for widget in calling_frame.winfo_children():
            widget.destroy()

        if self.loss_function_var.get() == "FocalTversky":
            ctk.CTkLabel(calling_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.alpha_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Beta").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.beta_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Gamma").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.gamma_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        elif self.loss_function_var.get() == "Tversky":
            ctk.CTkLabel(calling_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.alpha_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Beta").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.beta_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
        elif self.loss_function_var.get() == "Focal":
            ctk.CTkLabel(calling_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.alpha_focal_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Gamma").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.gamma_focal_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        elif self.loss_function_var.get() == "BCE":
            ctk.CTkLabel(calling_frame, text="Pos Weight").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.positive_weight_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        elif self.loss_function_var.get() == "Topoloss":
            ctk.CTkLabel(calling_frame, text="Patch Size").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.topoloss_patch_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        elif self.loss_function_var.get() == "Combo":
            ctk.CTkLabel(calling_frame, text="1").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkComboBox(calling_frame, values=["FocalTversky",
                                                    "Tversky",
                                                    "BCE",
                                                    "Dice",
                                                    "LCDice",
                                                    "Focal",
                                                    "Topoloss",
                                                    "IoU"], variable=self.loss_function1_var).grid(row=0, column=1, sticky="e", padx=5, pady=5)
            self.loss_function_options_frame1 = ctk.CTkFrame(calling_frame)
            loss_function_option_button1 = ctk.CTkButton(calling_frame, text="\u25BC", width=30)
            loss_function_option_button1.configure(command=lambda btn=loss_function_option_button1: self.toggle_options(self.loss_function_options_frame1, row=1, button=btn, update_function= lambda: self.update_loss_function_submenu(1)))
            loss_function_option_button1.grid(row=0, column=2, sticky="w", padx=5, pady=5)
            ctk.CTkLabel(calling_frame, text="w").grid(row=0, column=3, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.loss_function1_weight_var, width=40).grid(row=0, column=4, sticky="w", padx=5, pady=5)
            
            
            ctk.CTkLabel(calling_frame, text="2").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkComboBox(calling_frame, values=["FocalTversky",
                                                    "Tversky",
                                                    "BCE",
                                                    "Dice",
                                                    "LCDice",
                                                    "Focal",
                                                    "Topoloss",
                                                    "IoU"], variable=self.loss_function2_var).grid(row=2, column=1, sticky="e", padx=5, pady=5)
            self.loss_function_options_frame2 = ctk.CTkFrame(calling_frame)
            loss_function_option_button2 = ctk.CTkButton(calling_frame, text="\u25BC", width=30, )
            loss_function_option_button2.configure(command=lambda btn=loss_function_option_button2: self.toggle_options(self.loss_function_options_frame2, row=3, button=btn, update_function= lambda: self.update_loss_function_submenu(2)))
            loss_function_option_button2.grid(row=2, column=2, sticky="w", padx=5, pady=5)
            ctk.CTkLabel(calling_frame, text="w").grid(row=2, column=3, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.loss_function2_weight_var, width=40).grid(row=2, column=4, sticky="w", padx=5, pady=5)
            
            self.update_loss_function_submenu(1)
            self.update_loss_function_submenu(2)

        else:
            ctk.CTkLabel(calling_frame, text="No options for selected loss function").grid(row=0, column=0, columnspan=3, sticky="e", padx=5, pady=5)
            
        if self.loss_function_var.get() == "Combo":
            self.loss_func1_trace_id = self.loss_function1_var.trace_add("write", lambda *args: self.update_loss_function_submenu(1))
            self.loss_func2_trace_id = self.loss_function2_var.trace_add("write", lambda *args: self.update_loss_function_submenu(2))
      
    def update_loss_function_submenu(self, option, *args):
        if option == 1:
            calling_frame = self.loss_function_options_frame1
            loss_function_var = self.loss_function1_var
        elif option == 2:
            calling_frame = self.loss_function_options_frame2
            loss_function_var = self.loss_function2_var
        else:
            return
        
        for widget in calling_frame.winfo_children():
            widget.destroy()

        if loss_function_var.get() == "FocalTversky":
            ctk.CTkLabel(calling_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.alpha_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Beta").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.beta_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Gamma").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.gamma_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        elif loss_function_var.get() == "Tversky":
            ctk.CTkLabel(calling_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.alpha_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Beta").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.beta_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
        elif loss_function_var.get() == "Topoloss":
            ctk.CTkLabel(calling_frame, text="Patch").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.topoloss_patch_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
        elif loss_function_var.get() == "Focal":
            ctk.CTkLabel(calling_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.alpha_focal_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(calling_frame, text="Gamma").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.gamma_focal_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
        elif loss_function_var.get() == "BCE":
            ctk.CTkLabel(calling_frame, text="Pos Weight").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(calling_frame, textvariable=self.positive_weight_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        else:
            ctk.CTkLabel(calling_frame, text="No options for selected loss function").grid(row=0, column=0, columnspan=3, sticky="e", padx=5, pady=5)

    def update_transform_variable(self, new_sequence):
        self.transform_var.set(str(new_sequence))
    
    def create_gui(self):
        
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_left_panel(main_frame)
        self.create_right_panel(main_frame)

    def create_left_panel(self, parent):
        
        tab_view = ctk.CTkTabview(parent, anchor="nw")
        tab_view.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        tab_view.add("Training")
        tab_view.add("Dataset")
        tab_view.add("Testing")
        
        self.create_training_tab(tab_view.tab("Training"))
        self.create_dataset_tab(tab_view.tab("Dataset"))
        self.create_testing_tab(tab_view.tab("Testing"))
               
    def create_training_tab(self, parent):
        container_frame = ctk.CTkFrame(parent)
        container_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ctk.CTkScrollableFrame(container_frame, width=300)
        left_frame.pack(fill=tk.BOTH, expand=True)

        ###############################################################################################################
        # Model selection dropdown with toggle button
        ctk.CTkLabel(left_frame, text="Model").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        model_dropdown = ctk.CTkComboBox(left_frame, values=["U-Net",
                                                        "U-Net++",
                                                        "MAnet",
                                                        "LinkNet",
                                                        "FPN",
                                                        "PSPNet",
                                                        "PAN",
                                                        "DeepLabV3",
                                                        "DeepLabV3+"
                                                        ], variable=self.model_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.model_options_frame = ctk.CTkFrame(left_frame)
        model_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30)
        model_options_button.configure(command=lambda btn=model_options_button: self.toggle_options(self.model_options_frame, row=1, button=btn, update_function=self.update_model_options(self.model_options_frame)))
        model_options_button.grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.model_var.trace_add("write",lambda *args:  self.update_model_options(self.model_options_frame, *args))

        ###############################################################################################################
        # Encoder selection and dropdown
        ctk.CTkLabel(left_frame, text="Encoder").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        encoder_dropdown = ctk.CTkComboBox(left_frame, values=["efficientnet_b8",
                                                            "efficientnetv2_xl",
                                                            "resnet152",
                                                            "resnext50_32x4d",
                                                            "hrnet_w64",
                                                            "convnext_xxlarge",
                                                            "swin_large_patch4_window12_384",
                                                            "swinv2_cr_giant_38",
                                                            "vit_giant_patch14_clip_224",
                                                            "vit_giant_patch14_reg4_dinov2"
                                                            ], variable=self.encoder_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.encoder_options_frame = ctk.CTkFrame(left_frame)
        encoder_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30,)
        encoder_options_button.configure( command=lambda btn=encoder_options_button: self.toggle_options(self.encoder_options_frame, row=3, button=btn))
        encoder_options_button.grid(row=2, column=2, sticky="w", padx=5, pady=5)

        # Widgets for encoder options
        ctk.CTkLabel(self.encoder_options_frame, text="Pretrained Weights").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(self.encoder_options_frame, text=None, variable=self.pretrained_weights_var, onvalue=True, offvalue=False).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(self.encoder_options_frame, text="Freeze Backbone").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(self.encoder_options_frame, text=None, variable=self.freeze_backbone_var, onvalue=True, offvalue=False).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ###############################################################################################################
        # Optimizer selection and dropdown
        ctk.CTkLabel(left_frame, text="Optimizer").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        optimizer_dropdown = ctk.CTkComboBox(left_frame, values=["Adam",
                                                                 "AdamW",
                                                                 "SGD"], variable=self.optimizer_var).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        self.optimizer_options_frame = ctk.CTkFrame(left_frame)
        optimizer_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30)
        optimizer_options_button.configure(command=lambda btn=optimizer_options_button: self.toggle_options(self.optimizer_options_frame, row=5, button=btn, update_function=self.update_optimizer_options))
        optimizer_options_button.grid(row=4, column=2, sticky="w", padx=5, pady=5)
        self.optimizer_var.trace_add("write", self.update_optimizer_options)

        ###############################################################################################################
        # Warmup Scheduler selection and dropdown
        ctk.CTkLabel(left_frame, text="Warmup").grid(row=6, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkComboBox(left_frame,
                        values=["None",
                                "Linear"],
                        variable=self.warmup_scheduler_var).grid(row=6, column=1, sticky="w", padx=5, pady=5)

        self.warmup_options_frame = ctk.CTkFrame(left_frame)
        warmup_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30)
        warmup_options_button.configure(command=lambda btn=warmup_options_button: self.toggle_options(self.warmup_options_frame, row=7, button=btn, update_function=self.update_warmup_options))
        warmup_options_button.grid(row=6, column=2, sticky="w", padx=5, pady=5)
        self.warmup_scheduler_var.trace_add("write", self.update_warmup_options)

        ###############################################################################################################
        # Scheduler selection and dropdown
        ctk.CTkLabel(left_frame, text="Scheduler").grid(row=8, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkComboBox(left_frame, values=["None",
                                            "StepLR",
                                            "LinearLR",
                                            "CosineAnnealingLR"], variable=self.scheduler_var).grid(row=8, column=1, sticky="w", padx=5, pady=5)
        self.scheduler_options_frame = ctk.CTkFrame(left_frame)
        scheduler_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30)
        scheduler_options_button.configure(command=lambda btn=scheduler_options_button: self.toggle_options(self.scheduler_options_frame, row=9, button=btn, update_function=self.update_scheduler_options))
        scheduler_options_button.grid(row=8, column=2, sticky="w", padx=5, pady=5)
        self.scheduler_var.trace_add("write", self.update_scheduler_options)

        ###############################################################################################################
        # Loss function selection and dropdown
        ctk.CTkLabel(left_frame, text="Loss function").grid(row=10, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkComboBox(left_frame, values=["FocalTversky",
                                            "Tversky",
                                            "BCE",
                                            "Dice",
                                            "LCDice",
                                            "IoU",
                                            "Focal",
                                            "Topoloss",
                                            "Combo"], variable=self.loss_function_var).grid(row=10, column=1, sticky="w", padx=5, pady=5)
        self.loss_function_options_frame = ctk.CTkFrame(left_frame)
        loss_function_option_button = ctk.CTkButton(left_frame, text="\u25BC", width=30)
        loss_function_option_button.configure(command=lambda btn=loss_function_option_button: self.toggle_options(self.loss_function_options_frame, row=11, button=btn, update_function=self.update_loss_function_options(self.loss_function_options_frame)))
        loss_function_option_button.grid(row=10, column=2, sticky="w", padx=5, pady=5)
        self.loss_function_var.trace_add("write", lambda *args: self.update_loss_function_options(self.loss_function_options_frame, *args))

        ###############################################################################################################
        # Data Dir and Out Dir with Browse Buttons
        for i, dir_label in enumerate(["Data Dir", "Out Dir"], start=12):
            ctk.CTkLabel(left_frame, text=dir_label).grid(row=i, column=0, sticky="e", padx=5, pady=5)
            dir_entry = ctk.CTkEntry(left_frame, textvariable=self.data_dir_var if dir_label == "Data Dir" else self.output_dir_var)
            dir_entry.grid(row=i, column=1, sticky="w", padx=5, pady=5)
            browse_button = ctk.CTkButton(left_frame, text="...", width=30, command=lambda entry=dir_entry: self.browse_dir(entry))
            browse_button.grid(row=i, column=2, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Epochs").grid(row=14, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(left_frame, textvariable=self.epochs_var, validatecommand=self.int_validation).grid(row=14, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Batch Size").grid(row=15, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(left_frame, textvariable=self.batch_size_var, validatecommand=self.int_validation).grid(row=15, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Normalize").grid(row=16, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(left_frame, text="", variable=self.normalize_var, onvalue=True, offvalue=False).grid(row=16, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Negative").grid(row=17, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(left_frame, text="", variable=self.negative_var, onvalue=True, offvalue=False).grid(row=17, column=1, sticky="w", padx=5, pady=5)
        
        available_transforms = ["transforms.ToTensor", "transforms.Resize"]
        self.transforms_widget = TransformWidget(left_frame, available_transforms=available_transforms, update_callback=self.update_transform_variable)
        self.transforms_widget.grid(row=18, column=0, columnspan=3, sticky="we", padx=5, pady=5)

        train_button = ctk.CTkButton(left_frame, text="Start training", command=self.start_training, height=30)
        train_button.grid(row=19, columnspan=3, sticky="we", padx=5, pady=5)
        
        config_frame = ctk.CTkFrame(left_frame)
        config_frame.grid(row=20, columnspan=3, sticky="we", padx=5, pady=5)
        
        save_config = ctk.CTkButton(config_frame, text="Save Configs", command=self.save_config)
        save_config.grid(row=0, column=0,  sticky="w", padx=5, pady=5)
        
        load_config = ctk.CTkButton(config_frame, text="Load Configs", command=self.load_config)
        load_config.grid(row=0, column=1,  sticky="e", padx=5, pady=5)

        slider = ctk.CTkSlider(left_frame, from_=0.8, to=1.6, number_of_steps=10, command=self.change_scaling_event)
        slider.grid(row=21,columnspan=3, sticky="we", padx=5, pady=5)
  
    def create_testing_tab(self, parent):
        container_frame = ctk.CTkFrame(parent)
        container_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ctk.CTkScrollableFrame(container_frame, width=300)
        left_frame.pack(fill=tk.BOTH, expand=True)
        
        ###############################################################################################################
        # Model selection dropdown with toggle button
        ctk.CTkLabel(left_frame, text="Model").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        model_dropdown = ctk.CTkComboBox(left_frame, values=["U-Net",
                                                        "U-Net++",
                                                        "MAnet",
                                                        "LinkNet",
                                                        "FPN",
                                                        "PSPNet",
                                                        "PAN",
                                                        "DeepLabV3",
                                                        "DeepLabV3+"
                                                        ], variable=self.model_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.test_model_options_frame = ctk.CTkFrame(left_frame)
        model_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30)
        model_options_button.configure(command=lambda btn=model_options_button: self.toggle_options(self.test_model_options_frame, row=1, button=btn, update_function=self.update_model_options(self.test_model_options_frame)))
        model_options_button.grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.model_var.trace_add("write",lambda *args:  self.update_model_options(self.test_model_options_frame, *args))
        
        ###############################################################################################################
        # Encoder selection and dropdown
        ctk.CTkLabel(left_frame, text="Encoder").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        encoder_dropdown = ctk.CTkComboBox(left_frame, values=["efficientnet_b8",
                                                            "efficientnetv2_xl",
                                                            "resnet152",
                                                            "resnext50_32x4d",
                                                            "hrnet_w64",
                                                            "convnext_xxlarge",
                                                            "swin_large_patch4_window12_384",
                                                            "swinv2_cr_giant_38",
                                                            "vit_giant_patch14_clip_224",
                                                            "vit_giant_patch14_reg4_dinov2"
                                                            ], variable=self.encoder_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.test_encoder_options_frame = ctk.CTkFrame(left_frame)
        encoder_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30,)
        encoder_options_button.configure( command=lambda btn=encoder_options_button: self.toggle_options(self.test_encoder_options_frame, row=3, button=btn))
        encoder_options_button.grid(row=2, column=2, sticky="w", padx=5, pady=5)

        # Widgets for encoder options
        ctk.CTkLabel(self.test_encoder_options_frame, text="Pretrained Weights").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(self.test_encoder_options_frame, text=None, variable=self.pretrained_weights_var, onvalue=True, offvalue=False).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(self.test_encoder_options_frame, text="Freeze Backbone").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(self.test_encoder_options_frame, text=None, variable=self.freeze_backbone_var, onvalue=True, offvalue=False).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ###############################################################################################################
        # Model path
        ctk.CTkLabel(left_frame, text="Model Path").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        dir_entry = ctk.CTkEntry(left_frame, textvariable=self.test_model_path_var)
        dir_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        browse_button = ctk.CTkButton(left_frame, text="...", width=30, command=lambda entry=dir_entry: self.browse_file(entry))
        browse_button.grid(row=4, column=2, sticky="w", padx=5, pady=5)
        
        ###############################################################################################################
        # Loss function
        ctk.CTkLabel(left_frame, text="Loss function").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        loss_function_dropdown = ctk.CTkComboBox(left_frame, values=["FocalTversky",
                                                                     "Tversky",
                                                                     "IoU",
                                                                     "Combo"], variable=self.loss_function_var).grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.test_loss_function_options_frame = ctk.CTkFrame(left_frame)
        loss_function_option_button = ctk.CTkButton(left_frame, text="\u25BC", width=30, )
        loss_function_option_button.configure(command=lambda btn=loss_function_option_button: self.toggle_options(self.test_loss_function_options_frame, row=6, button=btn, update_function=self.update_loss_function_options(self.test_loss_function_options_frame)))
        loss_function_option_button.grid(row=5, column=2, sticky="w", padx=5, pady=5)
        self.loss_function_var.trace_add("write", lambda *args: self.update_loss_function_options(self.test_loss_function_options_frame, *args))
        
        ###############################################################################################################
        # Data dir and other
        ctk.CTkLabel(left_frame, text="Data Dir").grid(row=6, column=0, sticky="e", padx=5, pady=5)
        dir_entry = ctk.CTkEntry(left_frame, textvariable=self.test_data_dir_var)
        dir_entry.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        browse_button = ctk.CTkButton(left_frame, text="...", width=30, command=lambda entry=dir_entry: self.browse_dir(entry))
        browse_button.grid(row=6, column=2, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(left_frame, text="Batch Size").grid(row=7, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(left_frame, textvariable=self.test_batch_size_var, validatecommand=self.validate_int_input).grid(row=7, column=1, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(left_frame, text="Normalize").grid(row=8, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(left_frame, text="", variable=self.test_normalize_var, onvalue=True, offvalue=False).grid(row=8, column=1, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(left_frame, text="Negative").grid(row=9, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(left_frame, text="", variable=self.test_negative_var, onvalue=True, offvalue=False).grid(row=9, column=1, sticky="w", padx=5, pady=5)

        available_transforms = ["transforms.ToTensor", "transforms.Resize"]
        self.test_transforms_widget = TransformWidget(left_frame, available_transforms=available_transforms, update_callback=self.update_transform_variable)
        self.test_transforms_widget.grid(row=10, column=0, columnspan=3, sticky="we", padx=5, pady=5)

        test_button = ctk.CTkButton(left_frame, text="Start testing", command=self.start_testing, height=30)
        test_button.grid(row=11, columnspan=3, sticky="we", padx=5, pady=5)
        
        slider = ctk.CTkSlider(left_frame, from_=0.8, to=1.6, number_of_steps=10, command=self.change_scaling_event)
        slider.grid(row=12,columnspan=3, sticky="we", padx=5, pady=5)
    
    def create_dataset_tab(self, parent):
        container_frame = ctk.CTkFrame(parent)
        container_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ctk.CTkScrollableFrame(container_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        
        # Dataset Directory Selection
        ctk.CTkLabel(left_frame, text="Dataset Dir").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        dataset_dir_entry = ctk.CTkEntry(left_frame, textvariable=self.dataset_dir_var)
        dataset_dir_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        browse_button = ctk.CTkButton(left_frame, text="...", width=30, command=lambda entry=dataset_dir_entry: self.browse_dir(entry))
        browse_button.grid(row=0, column=2, sticky="w", padx=5, pady=5)

        # Index Input
        ctk.CTkLabel(left_frame, text="Index:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        index_entry = ctk.CTkEntry(left_frame, textvariable=self.dataset_index_var, width=50)
        index_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Negative").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(left_frame, text="", variable=self.negative_var, onvalue=True, offvalue=False).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Visualization Buttons
        button_width = 20  # Adjust button width as needed

        visualize_button = ctk.CTkButton(left_frame, text="Visualize Dataset", width=button_width, command=self.visualize_dataset_clicked)
        visualize_button.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        class_dist_button = ctk.CTkButton(left_frame, text="Class Distribution", width=button_width, command=self.class_distribution_clicked)
        class_dist_button.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        overlay_button = ctk.CTkButton(left_frame, text="Visualize Overlay", width=button_width, command=self.visualize_overlay_clicked)
        overlay_button.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        histogram_button = ctk.CTkButton(left_frame, text="Image Histogram", width=button_width, command=self.image_histogram_clicked)
        histogram_button.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        slider = ctk.CTkSlider(left_frame, from_=0.8, to=1.6, number_of_steps=10, command=self.change_scaling_event)
        slider.grid(row=7,columnspan=3, sticky="we", padx=5, pady=5)
    
    def visualize_dataset_clicked(self):
        self.load_and_visualize("visualize_dataset")

    def class_distribution_clicked(self):
        self.load_and_visualize("class_distribution")

    def visualize_overlay_clicked(self):
        self.load_and_visualize("visualize_overlay")

    def image_histogram_clicked(self):
        self.load_and_visualize("image_histogram")

    def load_and_visualize(self, visualization_type):
        dataset_dir = self.dataset_dir_var.get()
        if not dataset_dir:
            messagebox.showerror("Error", "Please select a dataset directory.")
            return

        image_dir = os.path.join(dataset_dir, "images")
        mask_dir = os.path.join(dataset_dir, "masks")

        # Check if image and mask directories exist
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            messagebox.showerror("Error", "Dataset directory must contain 'images' and 'masks' subdirectories.")
            return

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        try:
            dataset = SegmentationDataset(image_dir, mask_dir, image_transform=transform, mask_transform=transform, negative=bool(self.negative_var.get()))
            index_str = self.dataset_index_var.get()
            index = None # Default index is None, for random selection

            if index_str:
                try:
                    index = int(index_str)
                    if not 0 <= index < len(dataset):
                        messagebox.showerror("Error", f"Index out of range. Dataset size: {len(dataset)}.")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Invalid index. Please enter an integer.")
                    return

            if visualization_type == "visualize_dataset":
                self.plot_panel.visualize_dataset(dataset, idx=index) # Pass index
            elif visualization_type == "class_distribution":
                self.plot_panel.class_distribution(dataset)
            elif visualization_type == "visualize_overlay":
                self.plot_panel.visualize_overlay(dataset, idx=index) # Pass index
            elif visualization_type == "image_histogram":
                self.plot_panel.image_histogram(dataset, idx=index) # Pass index
            elif visualization_type == "inspect":
                self.plot_panel.inspect(dataset) # No index for inspect

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or visualize dataset: {e}")
    
    def change_scaling_event(self, new_scaling: str):
        new_scaling = float(new_scaling)
        ctk.set_widget_scaling(new_scaling)

    def create_right_panel(self, parent):
        right_frame = ctk.CTkFrame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        content_frame = ctk.CTkFrame(right_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_frame = ctk.CTkFrame(content_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_panel = PlotView(self.plot_frame, anchor="s")
        self.plot_panel.pack(fill=tk.BOTH, expand=True)
        self.image_evolution_dir = None

        message_frame = ctk.CTkFrame(content_frame)
        message_frame.pack(fill=tk.BOTH, expand=True)

        self.messages_box = tk.Text(
            message_frame,
            wrap=tk.WORD,
            bg="#1e1e1e",  # Dark grey background
            fg="#d4d4d4",  # Light grey text
            font=("Courier New", 16),
            insertbackground="#d4d4d4",
            selectbackground="#555555",
            selectforeground="#d4d4d4",
            height=8,
            relief=tk.FLAT,
            bd=0
        )

        # --- ttk Scrollbar (Dark Mode, handles disabled state) ---
        style = ttk.Style()
        style.theme_use('clam')

        # --- Define custom elements and layout ---
        style.element_create('Custom.Vertical.Scrollbar.trough', 'from', 'default')
        style.layout('Custom.Vertical.TScrollbar', [
            ('Custom.Vertical.Scrollbar.trough', {'sticky': 'ns', 'children': [
                ('Vertical.Scrollbar.thumb', {'expand': True, 'sticky': 'nswe'}),
                # IMPORTANT: Arrows are *separate* elements
                ('Vertical.Scrollbar.uparrow', {'side': 'top', 'sticky': 'we'}),
                ('Vertical.Scrollbar.downarrow', {'side': 'bottom', 'sticky': 'we'}),
            ]}),
        ])

        # --- Configure colors, including for disabled state ---
        style.configure("Custom.Vertical.TScrollbar",
            background="#2a2a2a",  # Thumb color
            troughcolor="#121212",  # Trough color (what we want)
            bordercolor="#121212",
            arrowcolor="#d4d4d4",
            darkcolor="#121212",
            lightcolor="#121212",
            gripcount=0,
        )

        # --- Map disabled state to custom colors ---
        style.map("Custom.Vertical.TScrollbar",
            background=[('disabled', '#1e1e1e')],  # Thumb when disabled
            troughcolor=[('disabled', '#121212')], # Trough when disabled
            bordercolor=[('disabled', '#121212')],
            arrowcolor=[('disabled', '#3a3a3a')],  # Darker arrows when disabled
            darkcolor=[('disabled', '#121212')],
            lightcolor=[('disabled', '#121212')],
        )

        scrollbar = ttk.Scrollbar(message_frame, orient="vertical",
                                   command=self.messages_box.yview,
                                   style="Custom.Vertical.TScrollbar")
        self.messages_box.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.messages_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.messages_box.config(state=tk.DISABLED)
        

        def adjust_height(event=None):
            total_height = content_frame.winfo_height()

            # Calculate a 70% height for plot_frame and 30% for message_frame
            plot_height = int(total_height * 0.7)
            message_height = int(total_height * 0.3)

            # Set heights to the containers that are then used to allocate space
            self.plot_frame.configure(height=plot_height)
            message_frame.configure(height=message_height)

        # Initial height setup
        right_frame.bind('<Configure>', lambda e: right_frame.after(10, adjust_height))

    def run(self):
        self.window.mainloop()
    
    def process_queue_continuous(self):
        while True:
            try:
                message = self.message_queue.get()
                if message is None:
                    break

                # Check if the *previous* line was a progress line
                if self.previous_line_was_progress_flag:
                    self.clear_last_message()

                # Update the flag for the next iteration based on the *current* message
                self.previous_line_was_progress_flag = bool(re.match(r"^Epoch \d+/\d+:", message) and not "100%" in message)

                # Log the message
                self.log_message(message)
                self.messages_box.update_idletasks()

                if r"\exp" in message:
                    self.image_dir(message)

                # Check if the message indicates the end of an epoch and update plots
                if "Train Loss" in message:
                    self.update_plots_after_epoch()
                

            except Exception as e:
                print(f"Error in process_queue_continuous: {e}")
                break
    
    def image_dir(self, message):
        try:
            # Extract the part after \exp from the message
            exp_part = message.split("\\exp")[1].split("\\")[0]  # Get the number after \exp
            new_dir = os.path.join(self.output_dir_var.get(), "exp" + exp_part, "results", "image_evolution")
            self.image_evolution_dir = new_dir # Update the directory
            self.plot_panel.set_image_dir(new_dir)  # Set the new directory for the plot panel

        except Exception as e:
            print(f"Error updating output directory: {e}")

    def update_plots_after_epoch(self):
        train_loss_values, val_loss_values = self.extract_loss_values()

        # Update loss plot with both train and val loss
        if train_loss_values or val_loss_values:
            self.plot_panel.plot_loss(train_loss_values, val_loss_values)

        # Update image plot using the current directory
        if self.image_evolution_dir:
            self.plot_panel.update_image_list()
            self.plot_panel.show_current_image()
    
    def extract_loss_values(self):
        train_loss_values = []  # Local variables to store extracted values
        val_loss_values = []
        content = self.messages_box.get("1.0", tk.END)
        lines = content.split("\n")
        for line in lines:
            match = re.search(r"Epoch \d+/\d+ - Train Loss: (\d+\.\d+) - Val Loss: (\d+\.\d+)", line)
            if match:
                try:
                    train_loss = float(match.group(1))
                    val_loss = float(match.group(2))
                    train_loss_values.append(train_loss)
                    val_loss_values.append(val_loss)
                except ValueError:
                    print(f"Could not convert loss values to float: {match.group(1)}, {match.group(2)}")
        return train_loss_values, val_loss_values
    
    def run_training_in_thread(self):
        try:
            args = ["python", "train.py"]  

            # Dynamically build the argument list based on _var attributes
            for name, var in self.__dict__.items():
                if not "test" in name and not "dataset" in name:
                    if name.endswith("_var") and isinstance(var, (tk.StringVar, tk.IntVar, tk.BooleanVar, ctk.StringVar, ctk.IntVar, ctk.BooleanVar)):
                        key = name.replace("_var", "")  # Remove "_var" for the argument name

                        # Handle boolean variables differently (append only if True)
                        if isinstance(var, (tk.BooleanVar, ctk.BooleanVar)):
                            if var.get():  # Only add the argument if it's True
                                if key == "pretrained_weights":
                                    args.append("--pretrained_weights")
                                if key == "freeze_backbone":
                                    args.append("--freeze_backbone")
                                elif key == "normalize":
                                    args.append("--normalize")
                                elif key == "negative":
                                    args.append("--negative")
                               
                        else: # Handle all stringVars
                            args.extend([f"--{key}", str(var.get())])
            
            # self.message_queue.put(f"Starting training with arguments: {' '.join(args)}")
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.getcwd()) #added cwd

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    output = output.replace(u'\u00A0', ' ')

                    self.message_queue.put(output) # Still put the output in the queue

            stderr = process.stderr.read()
            if stderr:
                self.message_queue.put(f"Standard Error:\n{stderr}") #debug

            return_code = process.poll()
            self.message_queue.put(f"Training process finished with return code: {return_code}") #
            if return_code != 0:
                self.message_queue.put("Training failed.")
            else:
                self.message_queue.put("Training Complete!")

        except ValueError as ve:
            self.message_queue.put(f"Input Error: {ve}")
        except FileNotFoundError:
            self.message_queue.put("Error: Training script 'train.py' not found. Make sure it's in the correct directory or provide the full path.")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info() #get detailed error info
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.message_queue.put(f"An unexpected error occurred:\n{e}\nType: {exc_type}\nFile: {fname}\nLine: {exc_tb.tb_lineno}")
    
    def run_testing_in_thread(self):
        try:
            # Retrieve values DIRECTLY from GUI variables
            model = self.model_var.get()
            attention = self.attention_var.get()
            batchnorm = self.batchnorm_var.get()
            encoder = self.encoder_var.get()
            weights = self.pretrained_weights_var.get()
            freeze = self.freeze_backbone_var.get()
            model_path = self.test_model_path_var.get()
            
            loss_func = self.loss_function_var.get()
            alpha = self.alpha_var.get()
            beta = self.beta_var.get()  
            gamma = self.gamma_var.get()
            
            # Combo loss
            loss_func1 = self.loss_function1_var.get()
            loss_func1_weight = self.loss_function1_weight_var.get()
            loss_func2 = self.loss_function2_var.get()
            loss_func2_weight = self.loss_function2_weight_var.get()
    
            data_dir = self.test_data_dir_var.get()
            batch_size = int(self.test_batch_size_var.get())
            normalize = self.normalize_var.get()
            negative = self.test_negative_var.get()
            transform = self.test_transforms_widget.get_sequence()


            # Construct the command
            args = ["python", "test.py"]  # Replace with your script name
            args.extend(["--model", model])
            args.extend(["--attention", attention])
            args.extend(["--batchnorm", batchnorm])
            args.extend(["--encoder", encoder])
            if bool(weights):
                args.append("--pretrained_weights")
            if bool(freeze):
                args.append("--freeze_backbone")
            args.extend(["--test_model_path", model_path])
            
            args.extend(["--loss_function", loss_func])
            args.extend(["--alpha", str(alpha)])
            args.extend(["--beta", str(beta)])
            args.extend(["--gamma", str(gamma)])
            
            args.extend(["--loss_function1", loss_func1])
            args.extend(["--loss_function1_weight", loss_func1_weight])
            args.extend(["--loss_function2", loss_func2])
            args.extend(["--loss_function2_weight", loss_func2_weight])
            
            args.extend(["--batch_size", str(batch_size)])
            args.extend(["--data_dir", data_dir])
            args.extend(["--transform", str(transform)])
            if normalize:
                args.append("--normalize")
            if negative:
                args.append("--negative")

            # self.message_queue.put(f"Starting testing with arguments: {' '.join(args)}") #debug
            # self.message_queue.put("Current Working Directory: " + os.getcwd()) #debug

            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.getcwd()) #added cwd

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    output = output.replace(u'\u00A0', ' ')

                    self.message_queue.put(output) # Still put the output in the queue

            stderr = process.stderr.read()
            if stderr:
                self.message_queue.put(f"Standard Error:\n{stderr}") #debug

            return_code = process.poll()
            self.message_queue.put(f"Testing process finished with return code: {return_code}") #
            if return_code != 0:
                self.message_queue.put("Testing failed.")
            else:
                self.message_queue.put("Testing Complete!")

        except ValueError as ve:
            self.message_queue.put(f"Input Error: {ve}")
        except FileNotFoundError:
            self.message_queue.put("Error: Testing script 'test.py' not found. Make sure it's in the correct directory or provide the full path.")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info() #get detailed error info
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.message_queue.put(f"An unexpected error occurred:\n{e}\nType: {exc_type}\nFile: {fname}\nLine: {exc_tb.tb_lineno}")
    
    def clear_last_message(self):
        self.messages_box.config(state=tk.NORMAL)

        # Get the total number of lines
        num_lines = int(self.messages_box.index('end-1c').split('.')[0])

        if num_lines >= 2:
            # Delete the second-to-last line
            last_line_start = f"{num_lines - 1}.0"
            last_line_end = f"{num_lines - 1}.end+1c"

            self.messages_box.delete(last_line_start, last_line_end)
            self.messages_box.update()
            time.sleep(0.1)

        self.messages_box.config(state=tk.DISABLED)

    def start_training(self):
        self.plot_panel.reset_image_data()
        
        # Clear the message box
        self.messages_box.config(state=tk.NORMAL)
        self.messages_box.delete("1.0", tk.END)
        self.messages_box.config(state=tk.DISABLED)
        
        self.message_queue.put("Starting training...")
        thread = threading.Thread(target=self.run_training_in_thread)
        thread.start()
        
    def start_testing(self):
        # Clear the message box
        self.messages_box.config(state=tk.NORMAL)
        self.messages_box.delete("1.0", tk.END)
        self.messages_box.config(state=tk.DISABLED)
        
        self.message_queue.put("Starting testing...")
        thread = threading.Thread(target=self.run_testing_in_thread)
        thread.start()


if __name__ == "__main__":
    app = ModeltrainingGUI()
    app.run()