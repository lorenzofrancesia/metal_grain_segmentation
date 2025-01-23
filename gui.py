from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.button import MDFillRoundFlatIconButton
from kivymd.uix.selectioncontrol import MDSwitch
from kivy.core.window import Window

Window.size = (1200, 800)

class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"  # "Dark"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"

        main_layout = MDBoxLayout(orientation='horizontal', padding=10)

        # Left Panel
        left_panel = MDBoxLayout(orientation='vertical', size_hint_x=0.3, padding=10, spacing=10)

        input_data = [
            ("Model", MDTextField()),
            ("Encoder", MDTextField()),
            ("Weights", MDTextField()),
            ("Optimizer", self.create_dropdown(["Adam", "SGD", "RMSprop"])),
            ("LR", MDTextField()),
            ("Loss Func", MDTextField()),
            ("Scheduler", MDTextField()),
            ("Epochs", MDTextField()),
            ("Data Dir", self.create_file_selector()),
            ("Batch Size", MDTextField()),
            ("Normalize", MDSwitch()),
            ("Transform", MDSwitch()),
            ("Out Dir", self.create_file_selector()),
        ]

        for label_text, widget in input_data:
            left_panel.add_widget(MDLabel(text=label_text))
            left_panel.add_widget(widget)

        main_layout.add_widget(left_panel)

        # Right Panel
        right_panel = MDBoxLayout(orientation='vertical', size_hint_x=0.7, padding=10, spacing=10)
        right_panel.add_widget(MDLabel(text='Plots go here', size_hint_y=0.7, halign="center", valign="middle"))
        right_panel.add_widget(MDLabel(text='Messages go here', size_hint_y=0.3, halign="center", valign="middle"))
        main_layout.add_widget(right_panel)

        return main_layout

    def create_dropdown(self, items):
        menu_items = [
            {"text": f"{item}", "viewclass": "OneLineListItem",
             "on_release": lambda x=item: self.set_item(x, menu)} for item in items
        ]
        menu = MDDropdownMenu(
            items=menu_items,
            width_mult=4,
        )
        return menu

    def set_item(self, item, menu):
        menu.caller.text = item
        menu.dismiss()

    def create_file_selector(self):
        box = MDBoxLayout(orientation="horizontal")
        text_field = MDTextField(hint_text="Select Path")
        button = MDFillRoundFlatIconButton(icon="folder", size_hint_x=None, width=30)
        box.add_widget(text_field)
        box.add_widget(button)
        return box


if __name__ == '__main__':
    MainApp().run()