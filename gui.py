import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog

def browse_file(entry):
    file_path = filedialog.askdirectory()
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def toggle_options(frame, row):
    if frame.winfo_ismapped():
        frame.grid_forget()
    else:
        frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=5)

def update_optimizer_options(optimizer_options_frame, optimizer_var):
    for widget in optimizer_options_frame.winfo_children():
        widget.destroy()
    
    if optimizer_var.get() == "Adam":
        ctk.CTkLabel(optimizer_options_frame, text="Learning Rate").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(optimizer_options_frame, textvariable=lr_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(optimizer_options_frame, text="Momentum").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(optimizer_options_frame, textvariable=momentum_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
    elif optimizer_var.get() == "SGD":
        ctk.CTkLabel(optimizer_options_frame, text="Learning Rate").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ctk.CTkEntry(optimizer_options_frame, textvariable=lr_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)

def create_gui():
    window = ctk.CTk()
    window.title("Model Training GUI")
    window.geometry("900x650")

    main_frame = ctk.CTkFrame(window)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = ctk.CTkFrame(main_frame, width=400)  # Increased width
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    right_frame = ctk.CTkFrame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Define StringVars for textvariables
    model_var = tk.StringVar(value="U-Net")
    dropout_prob_var = tk.StringVar(value="0.5")
    
    encoder_var = tk.StringVar(value="efficientnet_b8")
    pretrained_weights_var = tk.StringVar(value="False")
    
    optimizer_var = tk.StringVar(value="Adam")
    lr_var = tk.StringVar(value="0.001")
    momentum_var = tk.StringVar(value="0.9")
    
    loss_func_var = tk.StringVar(value="CrossEntropy")
    scheduler_var = tk.StringVar(value="StepLR")
    epochs_var = tk.StringVar(value="50")
    batch_size_var = tk.StringVar(value="32")
    data_dir_var = tk.StringVar()
    out_dir_var = tk.StringVar()
    
    
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
                                                      ], variable=model_var).grid(row=0, column=1, sticky="e", padx=5, pady=5)
    model_options_frame = ctk.CTkFrame(left_frame)
    ctk.CTkButton(left_frame, text="Options", width=50, command=lambda: toggle_options(model_options_frame, row=1)).grid(row=0, column=2, sticky="w", padx=5, pady=5)
    
    # Widgets for model options
    ctk.CTkLabel(model_options_frame, text="Dropout Prob").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(model_options_frame, textvariable=dropout_prob_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
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
                                                        ], variable=encoder_var).grid(row=2, column=1, sticky="e", padx=5, pady=5)
    encoder_options_frame = ctk.CTkFrame(left_frame)
    ctk.CTkButton(left_frame, text="Options", width=50, command=lambda: toggle_options(encoder_options_frame, row=3)).grid(row=2, column=2, sticky="w", padx=5, pady=5)
    
    # Widgets for encoder options
    ctk.CTkLabel(encoder_options_frame, text="Pretrained Weights").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkSwitch(encoder_options_frame, text=None, variable=pretrained_weights_var, onvalue="True", offvalue="False").grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
    
    ###############################################################################################################
    # Optimizer selection and dropdown
    ctk.CTkLabel(left_frame, text="Optimizer").grid(row=4, column=0, sticky="e", padx=5, pady=5)
    optimizer_dropdown = ctk.CTkComboBox(left_frame, values=["Adam",
                                                           "SGD",
                                                           "RMSprop"], variable=optimizer_var).grid(row=4, column=1, sticky="e", padx=5, pady=5)
    optimizer_options_frame = ctk.CTkFrame(left_frame)
    ctk.CTkButton(left_frame, text="Options", width=50, command=lambda: toggle_options(optimizer_options_frame, row=5)).grid(row=4, column=2, sticky="w", padx=5, pady=5)
    optimizer_var.trace_add("write", update_optimizer_options)
    
    # # Widgets for model options
    # if optimizer_var.get() == "Adam":
    #     ctk.CTkLabel(optimizer_options_frame, text="Learning Rate").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    #     ctk.CTkEntry(optimizer_options_frame, textvariable=lr_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
    #     ctk.CTkLabel(optimizer_options_frame, text="Momentum").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    #     ctk.CTkEntry(optimizer_options_frame, textvariable=momentum_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
    # if optimizer_var.get() == "SGD":
    #     ctk.CTkLabel(optimizer_options_frame, text="Learning Rate").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    #     ctk.CTkEntry(optimizer_options_frame, textvariable=lr_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
    ###############################################################################################################

    # Data Dir and Out Dir with Browse Buttons
    for i, dir_label in enumerate(["Data Dir", "Out Dir"], start=6):
        ctk.CTkLabel(left_frame, text=dir_label).grid(row=i, column=0, sticky="e", padx=5, pady=5)
        dir_entry = ctk.CTkEntry(left_frame, textvariable=data_dir_var if dir_label == "Data Dir" else out_dir_var)
        dir_entry.grid(row=i, column=1, sticky="w", padx=5, pady=5)
        browse_button = ctk.CTkButton(left_frame, text="...", width=30, command=lambda entry=dir_entry: browse_file(entry))
        browse_button.grid(row=i, column=2, sticky="w", padx=5, pady=5)

    ctk.CTkLabel(left_frame, text="Normalize").grid(row=i+2, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkSwitch(left_frame, text=None).grid(row=i+2, column=1, sticky="w", padx=5, pady=5)
    ctk.CTkLabel(left_frame, text="Transform").grid(row=i+3, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkSwitch(left_frame, text=None).grid(row=i+3, column=1, sticky="w", padx=5, pady=5)

    # Right Panel (Plots and Messages)
    plots_frame = ctk.CTkFrame(right_frame)
    plots_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    plots_label = ctk.CTkLabel(plots_frame, text="Plots will be displayed here")
    plots_label.pack(expand=True)

    messages_frame = ctk.CTkFrame(right_frame)
    messages_frame.pack(fill=tk.BOTH, expand=True)
    messages_label = ctk.CTkLabel(messages_frame, text="Messages will be displayed here")
    messages_label.pack(expand=True)

    window.mainloop()
if __name__ == "__main__":
    create_gui()