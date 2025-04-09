import tkinter as tk
from tkinter import ttk

def on_slider_change(event):
    # Update label with current slider value, formatted to one decimal point
    value = slider.get()
    formatted_value = f"{value:.1f}"
    value_label.config(text=f"Value: {formatted_value}")

# Create the main application window
root = tk.Tk()
root.title("Slider Example")

# Create a slider widget with range from 0 to 20
slider = ttk.Scale(root, from_=0, to=20, orient='horizontal', command=on_slider_change)
slider.pack(padx=10, pady=10)

# Create a label to display the slider's value
value_label = ttk.Label(root, text="Value: 0.0")
value_label.pack(padx=10, pady=10)

# Run the main event loop
root.mainloop()