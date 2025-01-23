import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog

def browse_file(entry):
    file_path = filedialog.askdirectory()
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def create_gui():
    window = ctk.CTk()
    window.title("Model Training GUI")
    window.geometry("800x600")

    main_frame = ctk.CTkFrame(window)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = ctk.CTkFrame(main_frame, width=400)  # Increased width
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    right_frame = ctk.CTkFrame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    input_data = [
        ("Model", ctk.CTkEntry(left_frame)),
        ("Encoder", ctk.CTkEntry(left_frame)),
        ("Weights", ctk.CTkEntry(left_frame)),
        ("Optimizer", ctk.CTkComboBox(left_frame, values=["Adam", "SGD", "RMSprop"])),
        ("LR", ctk.CTkEntry(left_frame)),
        ("Loss Func", ctk.CTkEntry(left_frame)),
        ("Scheduler", ctk.CTkEntry(left_frame)),
        ("Epochs", ctk.CTkEntry(left_frame)),
    ]

    for i, (label_text, widget) in enumerate(input_data):
        ctk.CTkLabel(left_frame, text=label_text).grid(row=i, column=0, sticky="e", padx=5, pady=5)
        widget.grid(row=i, column=1, sticky="w", padx=5, pady=5)

    # Data Dir and Out Dir with Browse Buttons
    for i, dir_label in enumerate(["Data Dir", "Out Dir"], start=len(input_data)):
        ctk.CTkLabel(left_frame, text=dir_label).grid(row=i, column=0, sticky="e", padx=5, pady=5)
        dir_entry = ctk.CTkEntry(left_frame)
        dir_entry.grid(row=i, column=1, sticky="w", padx=5, pady=5)
        browse_button = ctk.CTkButton(left_frame, text="...", width=30, command=lambda entry=dir_entry: browse_file(entry))
        browse_button.grid(row=i, column=2, sticky="w", padx=5, pady=5)

    ctk.CTkLabel(left_frame, text="Batch Size").grid(row=i+1, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(left_frame).grid(row=i+1, column=1, sticky="w", padx=5, pady=5)
    ctk.CTkLabel(left_frame, text="Normalize").grid(row=i+2, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkSwitch(left_frame).grid(row=i+2, column=1, sticky="w", padx=5, pady=5)
    ctk.CTkLabel(left_frame, text="Transform").grid(row=i+3, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkSwitch(left_frame).grid(row=i+3, column=1, sticky="w", padx=5, pady=5)

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