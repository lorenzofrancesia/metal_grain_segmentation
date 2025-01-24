import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json


class ModeltrainingGUI:

    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Model training GUI")
        self.window.geometry("1100x800")

        # define textvariables
        self.model_var = tk.StringVar(value="U-Net")
        self.dropout_prob_var = tk.StringVar(value="0.5")

        self.encoder_var = tk.StringVar(value="efficientnet_b8")
        self.pretrained_weights_var = tk.StringVar(value="False")

        self.optimizer_var = tk.StringVar(value="Adam")
        self.lr_var = tk.StringVar(value="0.001")
        self.momentum_var = tk.StringVar(value="0.9")

        self.scheduler_var = tk.StringVar(value="LinearLR")
        self.start_factor_var = tk.StringVar(value="0.3")
        self.end_factor_var = tk.StringVar(value="1")
        self.iterations_var = tk.StringVar(value="10")
        self.t_max_var = tk.StringVar(value="10")
        self.eta_min_var = tk.StringVar(value="0")

        self.loss_func_var = tk.StringVar(value="FocalTversky")
        self.alpha_var = tk.StringVar(value="0.7")
        self.beta_var = tk.StringVar(value="0.3")
        self.gamma_var = tk.StringVar(value="4/3")

        self.epochs_var = tk.StringVar(value="50")
        self.batch_size_var = tk.StringVar(value="32")
        self.data_dir_var = tk.StringVar()
        self.out_dir_var = tk.StringVar()

        #config variables
        self.config_file_var = tk.StringVar()

        self.create_gui()

    def browse_file(self, entry):
        file_path = filedialog.askdirectory()
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)
            
    def browse_config_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.config_file_var.set(file_path)
            self.load_config(file_path)

    def toggle_options(self, frame, row, update_function=None):
        if frame.winfo_ismapped():
            frame.grid_forget()
        else:
            frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=5)
            if update_function:
                update_function()

    def update_optimizer_options(self, *args):
        for widget in self.optimizer_options_frame.winfo_children():
            widget.destroy()

        if self.optimizer_var.get() == "Adam":
            ctk.CTkLabel(self.optimizer_options_frame, text="Learning Rate").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.lr_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.optimizer_options_frame, text="Momentum").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.momentum_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=1, column=1, sticky="w", padx=5, pady=5)


        elif self.optimizer_var.get() == "SGD":
            ctk.CTkLabel(self.optimizer_options_frame, text="Learning Rate").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.optimizer_options_frame, textvariable=self.lr_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=0, column=1, sticky="w", padx=5, pady=5)

    def update_scheduler_options(self, *args):
        for widget in self.scheduler_options_frame.winfo_children():
            widget.destroy()

        if self.scheduler_var.get() == "LinearLR":
            ctk.CTkLabel(self.scheduler_options_frame, text="Start Factor").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.start_factor_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.scheduler_options_frame, text="End Factor").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.end_factor_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=1, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.scheduler_options_frame, text="Iterations").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.iterations_var, validate="key", validatecommand=(self.window.register(self.validate_int_input), '%P')).grid(row=2, column=1, sticky="w", padx=5, pady=5)


        elif self.scheduler_var.get() == "CosineAnnealingLR":
            ctk.CTkLabel(self.scheduler_options_frame, text="T max").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.t_max_var, validate="key", validatecommand=(self.window.register(self.validate_int_input), '%P')).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.scheduler_options_frame, text="Eta min").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.scheduler_options_frame, textvariable=self.eta_min_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=1, column=1, sticky="w", padx=5, pady=5)

    def update_loss_function_options(self, *args):
        for widget in self.loss_function_options_frame.winfo_children():
            widget.destroy()

        if self.loss_func_var.get() == "FocalTversky":
            ctk.CTkLabel(self.loss_function_options_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.alpha_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.loss_function_options_frame, text="Beta").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.beta_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=1, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.loss_function_options_frame, text="Gamma").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.gamma_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        if self.loss_func_var.get() == "Tversky":
            ctk.CTkLabel(self.loss_function_options_frame, text="Alpha").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.alpha_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            ctk.CTkLabel(self.loss_function_options_frame, text="Beta").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(self.loss_function_options_frame, textvariable=self.beta_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=1, column=1, sticky="w", padx=5, pady=5)

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
    
    def start_training(self):
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.get_tk_widget().destroy()
                del self.canvas
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
                del self.fig
            self.clear_messages()
            self.log_message("Starting training...")
    
        # Retrieve all the settings
            training_settings = {
                "model": self.model_var.get(),
                "dropout_prob": self.dropout_prob_var.get(),
                "encoder": self.encoder_var.get(),
                "pretrained_weights": self.pretrained_weights_var.get(),
                "optimizer": self.optimizer_var.get(),
                "learning_rate": self.lr_var.get(),
                "momentum": self.momentum_var.get(),
                "scheduler": self.scheduler_var.get(),
                "start_factor": self.start_factor_var.get(),
                "end_factor": self.end_factor_var.get(),
                "iterations": self.iterations_var.get(),
                "t_max": self.t_max_var.get(),
                "eta_min": self.eta_min_var.get(),
                "loss_function": self.loss_func_var.get(),
                "alpha": self.alpha_var.get(),
                "beta": self.beta_var.get(),
                "gamma": self.gamma_var.get(),
                "epochs": self.epochs_var.get(),
                "batch_size": self.batch_size_var.get(),
                "data_dir": self.data_dir_var.get(),
                "out_dir": self.out_dir_var.get(),
            }
    
            # Displaying the settings in the message box
            self.log_message("Training Settings:")
            for key, value in training_settings.items():
                self.log_message(f"{key}: {value}")
        
            # Simulate some training progress
            for i in range(10):
                self.log_message(f"Epoch {i+1}/10 Complete")
    
            self.log_message("Training Complete!")
            self.show_dummy_plot()

    def log_message(self, message):
        self.messages_box.insert(tk.END, message + "\n")
        self.messages_box.see(tk.END)  # Scroll to the end

    def clear_messages(self):
        self.messages_box.delete("1.0", tk.END)

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
        
        self.canvas = canvas  # Store a reference to the canvas to destroy later
        self.fig = fig    # Store a reference to the figure to destroy later
        
    def save_config(self):
        config = {
            "model": self.model_var.get(),
            "dropout_prob": self.dropout_prob_var.get(),
            "encoder": self.encoder_var.get(),
            "pretrained_weights": self.pretrained_weights_var.get(),
            "optimizer": self.optimizer_var.get(),
            "lr": self.lr_var.get(),
            "momentum": self.momentum_var.get(),
            "scheduler": self.scheduler_var.get(),
            "start_factor": self.start_factor_var.get(),
            "end_factor": self.end_factor_var.get(),
            "iterations": self.iterations_var.get(),
            "t_max": self.t_max_var.get(),
            "eta_min": self.eta_min_var.get(),
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
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self, file_path):
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    self.model_var.set(config.get("model", "U-Net"))
                    self.dropout_prob_var.set(config.get("dropout_prob", "0.5"))
                    self.encoder_var.set(config.get("encoder", "efficientnet_b8"))
                    self.pretrained_weights_var.set(config.get("pretrained_weights", "False"))
                    self.optimizer_var.set(config.get("optimizer", "Adam"))
                    self.lr_var.set(config.get("lr", "0.001"))
                    self.momentum_var.set(config.get("momentum", "0.9"))
                    self.scheduler_var.set(config.get("scheduler", "LinearLR"))
                    self.start_factor_var.set(config.get("start_factor", "0.3"))
                    self.end_factor_var.set(config.get("end_factor", "1"))
                    self.iterations_var.set(config.get("iterations", "10"))
                    self.t_max_var.set(config.get("t_max", "10"))
                    self.eta_min_var.set(config.get("eta_min", "0"))
                    self.loss_func_var.set(config.get("loss_func", "FocalTversky"))
                    self.alpha_var.set(config.get("alpha", "0.7"))
                    self.beta_var.set(config.get("beta", "0.3"))
                    self.gamma_var.set(config.get("gamma", "4/3"))
                    self.epochs_var.set(config.get("epochs", "50"))
                    self.batch_size_var.set(config.get("batch_size", "32"))
                    self.data_dir_var.set(config.get("data_dir", ""))
                    self.out_dir_var.set(config.get("out_dir", ""))
                    
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
            


    def create_gui(self):
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ctk.CTkScrollableFrame(main_frame, width=400, height=700)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Menu Buttons (Moved to a separate frame above other left frame elements)
        menu_frame = ctk.CTkFrame(left_frame)
        menu_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(menu_frame, text="Load Config", command=self.browse_config_file).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkButton(menu_frame, text="Save Config", command=self.save_config).grid(row=0, column=1, padx=5, pady=5)


        ###############################################################################################################
        # Model selection dropdown with toggle button
        ctk.CTkLabel(left_frame, text="Model").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        model_dropdown = ctk.CTkComboBox(left_frame, values=["U-Net",
                                                            "U-Net++",
                                                            "MAnet",
                                                            "LinkNet",
                                                            "FPN",
                                                            "PSPNet",
                                                            "PAN",
                                                            "DeepLabV3",
                                                            "DeepLabV3+"], variable=self.model_var).grid(row=1, column=1, sticky="e", padx=5, pady=5)
        self.model_options_frame = ctk.CTkFrame(left_frame)
        ctk.CTkButton(left_frame, text="\u25BC", width=30, command=lambda: self.toggle_options(self.model_options_frame, row=2)).grid(row=1, column=2, sticky="w", padx=5, pady=5)

        # Widgets for model options
        ctk.CTkLabel(self.model_options_frame, text="Dropout Prob").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(self.model_options_frame, textvariable=self.dropout_prob_var, validate="key", validatecommand=(self.window.register(self.validate_numeric_input), '%P')).grid(row=0, column=1, sticky="w", padx=5, pady=5)


        ###############################################################################################################
        # Encoder selection and dropdown
        ctk.CTkLabel(left_frame, text="Encoder").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        encoder_dropdown = ctk.CTkComboBox(left_frame, values=["efficientnet_b8",
                                                                "efficientnetv2_xl",
                                                                "resnet152",
                                                                "resnext50_32x4d",
                                                                "hrnet_w64",
                                                                "convnext_xxlarge",
                                                                "swin_large_patch4_window12_384",
                                                                "swinv2_cr_giant_38",
                                                                "vit_giant_patch14_clip_224",
                                                                "vit_giant_patch14_reg4_dinov2"], variable=self.encoder_var).grid(row=3, column=1, sticky="e", padx=5, pady=5)
        self.encoder_options_frame = ctk.CTkFrame(left_frame)
        ctk.CTkButton(left_frame, text="\u25BC", width=30, command=lambda: self.toggle_options(self.encoder_options_frame, row=4)).grid(row=3, column=2, sticky="w", padx=5, pady=5)

        # Widgets for encoder options
        ctk.CTkLabel(self.encoder_options_frame, text="Pretrained Weights").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkSwitch(self.encoder_options_frame, text=None, variable=self.pretrained_weights_var, onvalue="True", offvalue="False").grid(row=0, column=1, sticky="w", padx=5, pady=5)


        ###############################################################################################################
        # Optimizer selection and dropdown
        ctk.CTkLabel(left_frame, text="Optimizer").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        optimizer_dropdown = ctk.CTkComboBox(left_frame, values=["Adam", "SGD"], variable=self.optimizer_var).grid(row=5, column=1, sticky="e", padx=5, pady=5)
        self.optimizer_options_frame = ctk.CTkFrame(left_frame)
        ctk.CTkButton(left_frame, text="\u25BC", width=30, command=lambda: self.toggle_options(self.optimizer_options_frame, row=6, update_function=self.update_optimizer_options)).grid(row=5, column=2, sticky="w", padx=5, pady=5)
        self.optimizer_var.trace_add("write", self.update_optimizer_options)


        ###############################################################################################################
        # Scheduler selection and dropdown
        ctk.CTkLabel(left_frame, text="Scheduler").grid(row=7, column=0, sticky="e", padx=5, pady=5)
        scheduler_dropdown = ctk.CTkComboBox(left_frame, values=["LinearLR", "CosineAnnealingLR"], variable=self.scheduler_var).grid(row=7, column=1, sticky="e", padx=5, pady=5)
        self.scheduler_options_frame = ctk.CTkFrame(left_frame)
        ctk.CTkButton(left_frame, text="\u25BC", width=30, command=lambda: self.toggle_options(self.scheduler_options_frame, row=8, update_function=self.update_scheduler_options)).grid(row=7, column=2, sticky="w", padx=5, pady=5)
        self.scheduler_var.trace_add("write", self.update_scheduler_options)


        ###############################################################################################################
        # Loss function selection and dropdown
        ctk.CTkLabel(left_frame, text="Loss function").grid(row=9, column=0, sticky="e", padx=5, pady=5)
        loss_function_dropdown = ctk.CTkComboBox(left_frame, values=["FocalTversky", "Tversky", "IoU"], variable=self.loss_func_var).grid(row=9, column=1, sticky="e", padx=5, pady=5)
        self.loss_function_options_frame = ctk.CTkFrame(left_frame)
        ctk.CTkButton(left_frame, text="\u25BC", width=30, command=lambda: self.toggle_options(self.loss_function_options_frame, row=10, update_function=self.update_loss_function_options)).grid(row=9, column=2, sticky="w", padx=5, pady=5)
        self.loss_func_var.trace_add("write", self.update_loss_function_options)

        ###############################################################################################################
        # Data Dir and Out Dir with Browse Buttons
        for i, dir_label in enumerate(["Data Dir", "Out Dir"], start=11):
            ctk.CTkLabel(left_frame, text=dir_label).grid(row=i, column=0, sticky="e", padx=5, pady=5)
            dir_entry = ctk.CTkEntry(left_frame, textvariable=self.data_dir_var if dir_label == "Data Dir" else self.out_dir_var)
            dir_entry.grid(row=i, column=1, sticky="w", padx=5, pady=5)
            browse_button = ctk.CTkButton(left_frame, text="...", width=30, command=lambda entry=dir_entry: self.browse_file(entry))
            browse_button.grid(row=i, column=2, sticky="w", padx=5, pady=5)

        ###############################################################################################################
        # Training Parameters
        ctk.CTkLabel(left_frame, text="Epochs").grid(row=13, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(left_frame, textvariable=self.epochs_var, validate="key", validatecommand=(self.window.register(self.validate_int_input), '%P')).grid(row=13, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(left_frame, text="Batch Size").grid(row=14, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(left_frame, textvariable=self.batch_size_var, validate="key", validatecommand=(self.window.register(self.validate_int_input), '%P')).grid(row=14, column=1, sticky="w", padx=5, pady=5)


        ###############################################################################################################
        # Start Training Button
        ctk.CTkButton(left_frame, text="Start Training", command=self.start_training).grid(row=15, column=0, columnspan=3, pady=20)


        ###############################################################################################################
        # Right Panel (Plots and Messages)
        
        # Create subframes for plot and messages
        self.plots_frame = ctk.CTkFrame(right_frame)
        self.plots_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP,  pady=(0, 10))

        plots_label = ctk.CTkLabel(self.plots_frame, text="Plots will be displayed here")
        plots_label.pack(expand=True)
        

        messages_frame = ctk.CTkFrame(right_frame)
        messages_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        
        # message box
        self.messages_box = scrolledtext.ScrolledText(messages_frame, wrap=tk.WORD, bg="#333", fg="white") # Dark mode colors
        self.messages_box.pack(fill=tk.BOTH, expand=True)
        
        
        # Clear messages button
        ctk.CTkButton(messages_frame, text="Clear Messages", command=self.clear_messages).pack(pady=5)

        self.window.update_idletasks()
        right_frame_height = right_frame.winfo_reqheight()
        self.plots_frame.configure(height=int(right_frame_height * 0.8))
        messages_frame.configure(height=int(right_frame_height * 0.2))

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = ModeltrainingGUI()
    app.run()