import customtkinter as ctk
import os 
import re
from PIL import Image
import numpy as np
import torch

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.pyplot as plt

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
        """
        Set the directory to watch for images and update the image list.

        Args:
            image_dir (str): Path to the directory containing images.
        """
        self.image_dir = image_dir
        self.update_image_list()
        self.current_image_index = len(self.image_files) - 1  # Point to the last image
        self.show_current_image()  # Display the last image in the new directory

    def update_image_list(self):
        """
        Update the list of image files from the current directory.

        Filters files by supported image extensions and sorts them numerically
        based on the first number in the filename.
        """
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
        """
        Display the previous image in the list.

        Decreases the current image index and updates the displayed image.
        """
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

    def show_next_image(self):
        """
        Display the next image in the list.

        Increases the current image index and updates the displayed image.
        """
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_current_image()

    def show_current_image(self):
        """
        Display the image at the current index.

        If no images are available, an empty plot is shown instead.
        """
        if not self.image_files:
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
        """
        Plot the training and validation loss values on the Losses tab.

        Args:
            train_loss_values (list): List of training loss values.
            val_loss_values (list): List of validation loss values.
        """
        self.loss_ax.clear()

        if train_loss_values:
            self.loss_ax.plot(range(1, len(train_loss_values) + 1), train_loss_values, label="Train Loss")
        if val_loss_values:
            self.loss_ax.plot(range(1, len(val_loss_values) + 1), val_loss_values, label="Val Loss")

        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.set_title("Training and Validation Loss")
        self.loss_ax.legend()

        self.loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.loss_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

        self.loss_canvas.draw()

    def clear_plots(self):
        """
        Clear the plots in both the Losses and Image Evolution tabs.
        """
        self.loss_ax.clear()
        self.loss_canvas.draw()
        self.image_ax.clear()
        self.image_ax.set_title("Image Evolution")
        self.image_ax.axis("off")
        self.image_canvas.draw()

    def show_empty_image_plot(self):
        """
        Display an empty plot for the Image Evolution tab.
        """
        self.image_ax.clear()
        self.image_ax.set_title("Image Evolution")
        self.image_ax.axis("off")
        self.image_canvas.draw()

    def reset_image_data(self):
        """
        Reset image-related data for a new training session.

        Clears the image directory, resets the image index, and clears plots.
        """
        self.image_dir = None
        self.current_image_index = 0
        self.image_files = []
        self.clear_plots()

    def visualize_dataset(self, dataset, idx=None):
        """
        Visualize dataset samples in the Dataset tab.

        Displays both the image and its corresponding mask.

        Args:
            dataset (Dataset): The dataset containing images and masks.
            idx (int, optional): Index of the sample to visualize. If None, a random index is used.
        """
        self.switch_to_tab("Dataset")
        self.dataset_ax.clear()

        self.dataset_figure.clf()
        image_ax = self.dataset_figure.add_subplot(1, 2, 1)
        mask_ax = self.dataset_figure.add_subplot(1, 2, 2)

        if idx is None:
            self.currentidx = np.random.randint(0, len(dataset))
        else:
            self.currentidx = idx
        image, mask = dataset[self.currentidx]

        image_np = image.permute(1, 2, 0).numpy() if isinstance(image, torch.Tensor) else image
        mask_np = mask.squeeze().numpy() if isinstance(mask, torch.Tensor) else mask

        if image_np.max() > 1:
            image_np = image_np / image_np.max()

        image_ax.imshow(image_np)
        image_ax.set_title(f"Image - Index: {idx}")
        image_ax.axis('off')

        mask_ax.imshow(mask_np, cmap='gray')
        mask_ax.set_title(f"Mask - Index: {idx}")
        mask_ax.axis('off')

        self.dataset_figure.tight_layout()
        self.dataset_canvas.draw()

    def class_distribution(self, dataset, classes_names=['Background', 'Foreground']):
        """
        Calculate and plot the class distribution with percentages in the Dataset tab.

        Args:
            dataset (Dataset): The dataset containing masks.
            classes_names (list): List of class names for the distribution.
        """
        self.switch_to_tab("Dataset")
        self.dataset_ax.clear()
        self.dataset_figure.clf()
        ax = self.dataset_figure.add_subplot(111)

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

        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{percentage:.2f}%",
                    ha="center", va="bottom", fontsize=10, color="white")

        self.dataset_figure.tight_layout()
        self.dataset_canvas.draw()

    def visualize_overlay(self, dataset, idx=None, alpha=0.35):
        """
        Visualize the overlay of an image and its mask in the Dataset tab.

        Args:
            dataset (Dataset): The dataset containing images and masks.
            idx (int, optional): Index of the sample to visualize. If None, a random index is used.
            alpha (float): Transparency level for the mask overlay.
        """
        self.switch_to_tab("Dataset")
        self.dataset_ax.clear()
        self.dataset_figure.clf()
        ax = self.dataset_figure.add_subplot(111)

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
        """
        Calculate and plot the histogram of pixel values in an image on the Dataset tab.

        Args:
            dataset (Dataset): The dataset containing images.
            idx (int, optional): Index of the sample to visualize. If None, a random index is used.
        """
        self.switch_to_tab("Dataset")
        self.dataset_ax.clear()
        self.dataset_figure.clf()
        ax = self.dataset_figure.add_subplot(111)
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
        """
        Clear only the matplotlib plot area in the Dataset tab.

        Keeps the black background and ensures axes are turned off.
        """
        self.dataset_ax.clear()
        self.dataset_figure.clf()
        self.dataset_ax.set_facecolor('black')
        self.dataset_ax.axis("off")

    def switch_to_tab(self, tab_name):
        """
        Switch the view to the specified tab.

        Args:
            tab_name (str): Name of the tab to switch to.
        """
        self.set(tab_name)