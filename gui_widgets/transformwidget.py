import customtkinter as ctk
import re


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
        """
        Add a new row to the transform table.

        Args:
            default (str, optional): Default transform to select in the dropdown.
            default_size (str, optional): Default size for the resize entry, if applicable.
        """
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
        """
        Handle changes in the dropdown selection.

        Args:
            value (str): Selected transform value.
            row_frame (CTkFrame): The frame containing the dropdown.
        """
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
        """
        Delete a row from the transform table.

        Args:
            frame (CTkFrame): The frame of the row to delete.
        """
        self.rows = [item for item in self.rows if item["frame"] != frame]
        frame.destroy()
        self._update_widget()

    def get_sequence(self):
        """
        Get the current sequence of transforms as a list of strings.

        Returns:
            list: List of transform strings in the format "transform(args)".
        """
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
        """
        Load a configuration string and populate the transform table.

        Args:
            config_string (str or list): String representation of a list of transforms or a list itself.
        """
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
        """
        Update the widget by invoking the update callback with the current sequence.
        """
        if self.update_callback:
            self.update_callback(self.get_sequence())
            
    def _entry_changed(self, item):
        """
        Handle changes in the resize entry field.

        Args:
            item (dict): Dictionary containing the dropdown and resize entry widgets.
        """
        if item["dropdown"].get() == "transforms.Resize":
            try:  # Basic validation
                eval(item["resize_entry"].get()) # Check if it's valid Python syntax (tuple)
                self._update_widget() 
            except Exception:
                pass