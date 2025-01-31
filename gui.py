import tkinter as tk
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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StringRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.display_output(string)

    def flush(self):
        pass

class TransformWidget(ctk.CTkFrame):
    def __init__(self, master, available_transforms=None):
        super().__init__(master)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.available_transforms = available_transforms or ["transforms.ToTensor", "transforms.Resize"]

        self.label = ctk.CTkLabel(self, text="Transform")
        self.label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.add_btn = ctk.CTkButton(self, text="+", width=30, command=self.add_row)
        self.add_btn.grid(row=0, column=1, padx=5, pady=5, sticky="e")

        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.table_frame.grid_columnconfigure(1, weight=1) # Key change here

        self.rows = []
        self.add_row(default="transforms.ToTensor")

    def add_row(self, default=None):
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

        resize_entry = ctk.CTkEntry(row_frame, placeholder_text="Enter size", width=50)
        resize_entry.grid(row=0, column=2, padx=5)
        resize_entry.grid_remove()  # Use grid_remove for cleaner hiding
        resize_entry.configure(state="disabled")

        self.rows.append({"frame": row_frame, "dropdown": dropdown, "resize_entry": resize_entry, "delete_btn": delete_btn})

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

class PlotView(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # create tabs
        self.add("Losses")
        self.add("Image Evolution")

        # add widgets on tabs
        self.label = ctk.CTkLabel(master=self.tab("Image Evolution"))
        self.label.grid(row=0, column=0, padx=20, pady=10)

class ModeltrainingGUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Model training GUI")
        self.window.geometry("900x550")

        self._init_variables()

        self.message_queue = queue.Queue()  # Create a message queue
        self.previous_line_was_progress_flag = False # Initialize the flag here
        threading.Thread(target=self.process_queue_continuous, daemon=True).start()

        self.create_gui()

    def _init_variables(self):
        self.model_var = tk.StringVar(value="U-Net")
        self.dropout_prob_var = tk.StringVar(value="0.5")
        self.pooling_type_var = tk.StringVar(value="Max")

        self.encoder_var = tk.StringVar(value="resnet50")
        self.pretrained_weights_var = tk.BooleanVar(value=False)

        self.optimizer_var = tk.StringVar(value="Adam")
        self.lr_var = tk.StringVar(value="0.001")
        self.momentum_var = tk.StringVar(value="0.9")
        self.weight_decay_var = tk.StringVar(value="1e-4")

        self.scheduler_var = tk.StringVar(value="None")
        self.start_factor_var = tk.StringVar(value="0.3")
        self.end_factor_var = tk.StringVar(value="1")
        self.iterations_var = tk.StringVar(value="10")
        self.t_max_var = tk.StringVar(value="10")
        self.eta_min_var = tk.StringVar(value="0")
        self.step_size_var = tk.StringVar(value="5")
        self.gamma_lr_var = tk.StringVar(value="0.5")

        self.loss_func_var = tk.StringVar(value="FocalTversky")
        self.alpha_var = tk.StringVar(value="0.7")
        self.beta_var = tk.StringVar(value="0.3")
        self.gamma_var = tk.StringVar(value="1.3333")

        self.epochs_var = tk.StringVar(value="1")
        self.batch_size_var = tk.StringVar(value="6")
        self.data_dir_var = tk.StringVar(value="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\data")
        self.out_dir_var = tk.StringVar(value="C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Documents\\Project\\runs")
        self.normalize_var = tk.BooleanVar(value=False)

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

    def log_message(self, message):
        if self.messages_box:
            self.messages_box.config(state=tk.NORMAL)
            self.messages_box.insert(tk.END, message + "\n")
            self.messages_box.see(tk.END)
            self.messages_box.config(state=tk.DISABLED)
            self.messages_box.update()
            time.sleep(0.05)  # Increased delay to 0.05 seconds in log_message
            #self.window.update_idletasks() # no need for this here, update() is stronger

    def show_dummy_plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([0, 1, 2, 3, 4], [10, 12, 8, 15, 13])
        ax.set_title("Dummy Plot")
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")

        canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def save_config(self):
        config = {
            "model": self.model_var.get(),
            "dropout_prob": self.dropout_prob_var.get(),
            "pooling_tpe": self.pooling_type_var.get(),
            "encoder": self.encoder_var.get(),
            "pretrained_weights": self.pretrained_weights_var.get(),
            "optimizer": self.optimizer_var.get(),
            "lr": self.lr_var.get(),
            "momentum": self.momentum_var.get(),
            "weight_decay": self.weight_decay_var.get(),
            "scheduler": self.scheduler_var.get(),
            "start_factor": self.start_factor_var.get(),
            "end_factor": self.end_factor_var.get(),
            "iterations": self.iterations_var.get(),
            "t_max": self.t_max_var.get(),
            "eta_min": self.eta_min_var.get(),
            "step_size": self.step_size_var.get(),
            "gamma_lr": self.gamma_lr_var.get(),
            "loss_func": self.loss_func_var.get(),
            "alpha": self.alpha_var.get(),
            "beta": self.beta_var.get(),
            "gamma": self.gamma_var.get(),
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_size_var.get(),
            "data_dir": self.data_dir_var.get(),
            "out_dir": self.out_dir_var.get(),
        }

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
                
    def load_config(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                self.model_var.set(config.get("model", ""))
                self.dropout_prob_var.set(config.get("dropout_prob", ""))
                self.pooling_type_var.set(config.get("pooling_type"))
                self.encoder_var.set(config.get("encoder", ""))
                self.pretrained_weights_var.set(config.get("pretrained_weights", False))
                self.optimizer_var.set(config.get("optimizer", ""))
                self.lr_var.set(config.get("lr", ""))
                self.momentum_var.set(config.get("momentum", ""))
                self.weight_decay_var.set(config.get("weight_decay", ""))
                self.scheduler_var.set(config.get("scheduler", ""))
                self.start_factor_var.set(config.get("start_factor", ""))
                self.end_factor_var.set(config.get("end_factor", ""))
                self.iterations_var.set(config.get("iterations", ""))
                self.t_max_var.set(config.get("t_max", ""))
                self.eta_min_var.set(config.get("eta_min", ""))
                self.step_size_var.set(config.get("step_size", ""))
                self.gamma_lr_var.set(config.get("gamma_lr", ""))
                self.loss_func_var.set(config.get("loss_func", ""))
                self.alpha_var.set(config.get("alpha", ""))
                self.beta_var.set(config.get("beta", ""))
                self.gamma_var.set(config.get("gamma", ""))
                self.epochs_var.set(config.get("epochs", ""))
                self.batch_size_var.set(config.get("batch_size", ""))
                self.data_dir_var.set(config.get("data_dir", ""))
                self.out_dir_var.set(config.get("out_dir", ""))
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

    def update_optimizer_options(self, *args):
        for widget in self.optimizer_options_frame.winfo_children():
            widget.destroy()

        if self.optimizer_var.get() == "Adam":
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

    def update_loss_function_options(self, *args):
        for widget in self.loss_function_options_frame.winfo_children():
            widget.destroy()

        if self.loss_func_var.get() == "FocalTversky":
            ctk.CTkLabel(self.loss_function_options_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.alpha_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.loss_function_options_frame, text="Beta").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.beta_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.loss_function_options_frame, text="Gamma").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.gamma_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        elif self.loss_func_var.get() == "Tversky":
            ctk.CTkLabel(self.loss_function_options_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.alpha_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.loss_function_options_frame, text="Beta").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.beta_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

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
                                                        ], variable=self.model_var).grid(row=0, column=1, sticky="e", padx=5, pady=5)
        self.model_options_frame = ctk.CTkFrame(left_frame)
        model_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30)
        model_options_button.configure(command=lambda btn=model_options_button: self.toggle_options(self.model_options_frame, row=1, button=btn))
        model_options_button.grid(row=0, column=2, sticky="w", padx=5, pady=5)

        # Widgets for model options
        ctk.CTkLabel(self.model_options_frame, text="Dropout Prob").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(self.model_options_frame, textvariable=self.dropout_prob_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(self.model_options_frame, text="Pooling Type").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkComboBox(self.model_options_frame, values=["Max", "Avg"], variable=self.pooling_type_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

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
                                                            ], variable=self.encoder_var).grid(row=2, column=1, sticky="e", padx=5, pady=5)
        self.encoder_options_frame = ctk.CTkFrame(left_frame)
        encoder_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30,)
        encoder_options_button.configure( command=lambda btn=encoder_options_button: self.toggle_options(self.encoder_options_frame, row=3, button=btn))
        encoder_options_button.grid(row=2, column=2, sticky="w", padx=5, pady=5)

        # Widgets for encoder options
        ctk.CTkLabel(self.encoder_options_frame, text="Pretrained Weights").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(self.encoder_options_frame, text=None, variable=self.pretrained_weights_var, onvalue=True, offvalue=False).grid(row=0, column=1, sticky="w", padx=5, pady=5)


        ###############################################################################################################
        # Optimizer selection and dropdown
        ctk.CTkLabel(left_frame, text="Optimizer").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        optimizer_dropdown = ctk.CTkComboBox(left_frame, values=["Adam",
                                                                 "SGD"], variable=self.optimizer_var).grid(row=4, column=1, sticky="e", padx=5, pady=5)
        self.optimizer_options_frame = ctk.CTkFrame(left_frame)
        optimizer_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30)
        optimizer_options_button.configure(command=lambda btn=optimizer_options_button: self.toggle_options(self.optimizer_options_frame, row=5, button=btn, update_function=self.update_optimizer_options))
        optimizer_options_button.grid(row=4, column=2, sticky="w", padx=5, pady=5)
        self.optimizer_var.trace_add("write", self.update_optimizer_options)


        ###############################################################################################################
        # Scheduler selection and dropdown
        ctk.CTkLabel(left_frame, text="Scheduler").grid(row=6, column=0, sticky="e", padx=5, pady=5)
        scheduler_dropdown = ctk.CTkComboBox(left_frame, values=["None",
                                                                 "StepLR",
                                                                 "LinearLR",
                                                                 "CosineAnnealingLR"], variable=self.scheduler_var).grid(row=6, column=1, sticky="e", padx=5, pady=5)
        self.scheduler_options_frame = ctk.CTkFrame(left_frame)
        scheduler_options_button = ctk.CTkButton(left_frame, text="\u25BC", width=30, )
        scheduler_options_button.configure(command=lambda btn=scheduler_options_button: self.toggle_options(self.scheduler_options_frame, row=7, button=btn, update_function=self.update_scheduler_options))
        scheduler_options_button.grid(row=6, column=2, sticky="w", padx=5, pady=5)
        self.scheduler_var.trace_add("write", self.update_scheduler_options)

        ###############################################################################################################
        # Loss function selection and dropdown
        ctk.CTkLabel(left_frame, text="Loss function").grid(row=8, column=0, sticky="e", padx=5, pady=5)
        loss_function_dropdown = ctk.CTkComboBox(left_frame, values=["FocalTversky",
                                                                     "Tversky",
                                                                     "IoU"], variable=self.loss_func_var).grid(row=8, column=1, sticky="e", padx=5, pady=5)
        self.loss_function_options_frame = ctk.CTkFrame(left_frame)
        loss_function_option_button = ctk.CTkButton(left_frame, text="\u25BC", width=30, )
        loss_function_option_button.configure(command=lambda btn=loss_function_option_button: self.toggle_options(self.loss_function_options_frame, row=9, button=btn, update_function=self.update_loss_function_options))
        loss_function_option_button.grid(row=8, column=2, sticky="w", padx=5, pady=5)
        self.loss_func_var.trace_add("write", self.update_loss_function_options)

        ###############################################################################################################

        # Data Dir and Out Dir with Browse Buttons
        for i, dir_label in enumerate(["Data Dir", "Out Dir"], start=10):
            ctk.CTkLabel(left_frame, text=dir_label).grid(row=i, column=0, sticky="e", padx=5, pady=5)
            dir_entry = ctk.CTkEntry(left_frame, textvariable=self.data_dir_var if dir_label == "Data Dir" else self.out_dir_var)
            dir_entry.grid(row=i, column=1, sticky="w", padx=5, pady=5)
            browse_button = ctk.CTkButton(left_frame, text="...", width=30, command=lambda entry=dir_entry: self.browse_file(entry))
            browse_button.grid(row=i, column=2, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Epochs").grid(row=12, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(left_frame, textvariable=self.epochs_var, validatecommand=self.validate_int_input).grid(row=12, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Batch Size").grid(row=13, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(left_frame, textvariable=self.batch_size_var, validatecommand=self.validate_int_input).grid(row=13, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Normalize").grid(row=14, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(left_frame, text="", variable=self.normalize_var, onvalue=True, offvalue=False).grid(row=14, column=1, sticky="w", padx=5, pady=5)

        available_transforms = ["transforms.ToTensor", "transforms.Resize"]
        self.transforms_widgets = TransformWidget(left_frame, available_transforms=available_transforms)
        self.transforms_widgets.grid(row=15, column=0, columnspan=3, sticky="we", padx=5, pady=5)

        train_button = ctk.CTkButton(left_frame, text="Start training", command=self.start_training, height=30)
        train_button.grid(row=16, columnspan=3, sticky="we", padx=5, pady=5)
        
        config_frame = ctk.CTkFrame(left_frame)
        config_frame.grid(row=17, columnspan=3, sticky="we", padx=5, pady=5)
        
        save_config = ctk.CTkButton(config_frame, text="Save Configs", command=self.save_config)
        save_config.grid(row=0, column=0,  sticky="w", padx=5, pady=5)
        
        load_config = ctk.CTkButton(config_frame, text="Load Configs", command=self.load_config)
        load_config.grid(row=0, column=1,  sticky="e", padx=5, pady=5)

        slider = ctk.CTkSlider(left_frame, from_=0.8, to=1.6, number_of_steps=10, command=self.change_scaling_event)
        slider.grid(row=18,columnspan=3, sticky="we", padx=5, pady=5)
    
    def change_scaling_event(self, new_scaling: str):
        new_scaling = float(new_scaling)
        ctk.set_widget_scaling(new_scaling)

    def create_right_panel(self, parent):
        right_frame = ctk.CTkFrame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        content_frame = ctk.CTkFrame(right_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        plot_frame = ctk.CTkFrame(content_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        plot_panel = PlotView(plot_frame, anchor="s")
        plot_panel.pack(fill=tk.BOTH, expand=True)

        message_frame = ctk.CTkFrame(content_frame)
        message_frame.pack(fill=tk.BOTH, expand=True)

        # Terminal-style Text widget
        self.messages_box = tk.Text(
            message_frame,
            wrap=tk.WORD,
            bg="black",  # Black background
            fg="white",  # White foreground (text color)
            font=("Courier New", 16),  # Monospace font, larger size
            insertbackground="white",  # cursor color
            selectbackground="#333333",  # selected text color
            selectforeground="white",
            height=8
        )
        scrollbar = tk.Scrollbar(message_frame, command=self.messages_box.yview)
        self.messages_box.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.messages_box.pack(fill=tk.BOTH, expand=True)
        self.messages_box.config(state=tk.DISABLED)

        def adjust_height(event=None):
            total_height = content_frame.winfo_height()

            # Calculate a 70% height for plot_frame and 30% for message_frame
            plot_height = int(total_height * 0.7)
            message_height = int(total_height * 0.3)

            # Set heights to the containers that are then used to allocate space
            plot_frame.configure(height=plot_height)
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

            except Exception as e:
                print(f"Error in process_queue_continuous: {e}")
                break

    def run_training_in_thread(self):
        try:
            # Retrieve values DIRECTLY from GUI variables
            model = self.model_var.get()
            dropout = float(self.dropout_prob_var.get())
            pooling = self.pooling_type_var.get().lower()

            encoder = self.encoder_var.get()
            pretrained_weights = self.pretrained_weights_var.get()

            optimizer = self.optimizer_var.get()
            lr = float(self.lr_var.get())
            momentum = self.momentum_var.get()
            weight_decay = self.weight_decay_var.get()

            scheduler = self.scheduler_var.get()
            start_factor = self.start_factor_var.get()
            end_factor = self.end_factor_var.get()
            iterations = self.iterations_var.get()
            t_max = self.t_max_var.get()
            eta_min = self.eta_min_var.get()
            step_size = self.step_size_var.get()
            gamma_lr = self.gamma_lr_var.get()

            loss_func = self.loss_func_var.get()
            alpha = self.alpha_var.get()
            beta = self.beta_var.get()
            gamma = self.gamma_var.get()


            data_dir = self.data_dir_var.get()
            batch_size = int(self.batch_size_var.get())
            epochs = int(self.epochs_var.get())
            output_dir = self.out_dir_var.get()
            normalize = self.normalize_var.get()
            transform = self.transforms_widgets.get_sequence()


            # Construct the command
            args = ["python", "train.py"]  # Replace with your script name
            args.extend(["--model", model])
            args.extend(["--dropout", str(dropout)])
            args.extend(["--pooling", pooling])
            
            args.extend(["--encoder", encoder])
            if pretrained_weights:
                args.append("--weights")
            
            args.extend(["--optimizer", optimizer])
            args.extend(["--lr", str(lr)])
            args.extend(["--momentum", str(momentum)])
            args.extend(["--weight_decay", weight_decay])
            
            args.extend(["--scheduler", scheduler])
            args.extend(["--start_factor", str(start_factor)])
            args.extend(["--end_factor", str(end_factor)])
            args.extend(["--iterations", str(iterations)])
            args.extend(["--t_max", str(t_max)])
            args.extend(["--eta_min", str(eta_min)])
            args.extend(["--step_size", str(step_size)])
            args.extend(["--gamma_lr", str(gamma_lr)])
            
            args.extend(["--loss_func", loss_func])
            args.extend(["--alpha", str(alpha)])
            args.extend(["--beta", str(beta)])
            args.extend(["--gamma", str(gamma)])
            
            args.extend(["--batch_size", str(batch_size)])
            args.extend(["--epochs", str(epochs)])
            
            args.extend(["--data_dir", data_dir])
            args.extend(["--output_dir", output_dir])

            args.extend(["--transform", str(transform)])
            if normalize:
                args.append("--normalize")



            self.message_queue.put(f"Starting training with arguments: {' '.join(args)}") #debug
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
            self.message_queue.put(f"Training process finished with return code: {return_code}") #
            if return_code != 0:
                self.message_queue.put("Training failed.")
            else:
                self.message_queue.put("Training Complete!")

        except ValueError as ve:
            self.message_queue.put(f"Input Error: {ve}")
        except FileNotFoundError:
            self.message_queue.put("Error: Training script 'your_training_script.py' not found. Make sure it's in the correct directory or provide the full path.")
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
        self.message_queue.put("Starting training...")
        thread = threading.Thread(target=self.run_training_in_thread)
        thread.start()


if __name__ == "__main__":
    app = ModeltrainingGUI()
    app.run()