# matplotlib_embed.py
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import logging

class MatplotlibEmbed:
    """Manages an embedded Matplotlib plot within a Tkinter frame."""

    def __init__(self, master_frame):
        self.master = master_frame
        self.fig = Figure(figsize=(8, 6), dpi=100) # Create Figure
        self.ax = self.fig.add_subplot(111)      # Add Axes
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master) # Link Figure to Tkinter Canvas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=False) # Add navigation toolbar
        self.toolbar.update()

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Display canvas
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)                  # Display toolbar

        self.click_cid = None
        self.key_cid = None
        self.onclick_callback = None # Function to call when plot is clicked
        self.onkey_callback = None   # Function to call on key press

    def connect_events(self, onclick_func, onkey_func):
        """Connects MPL event handlers."""
        self.disconnect_events() # Disconnect old ones first
        self.onclick_callback = onclick_func
        self.onkey_callback = onkey_func
        if self.onclick_callback:
            self.click_cid = self.canvas.mpl_connect('button_press_event', self.onclick_callback)
        if self.onkey_callback:
            self.key_cid = self.canvas.mpl_connect('key_press_event', self.onkey_callback)

    def disconnect_events(self):
        """Disconnects existing MPL event handlers."""
        if self.click_cid:
            self.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None
        if self.key_cid:
            self.canvas.mpl_disconnect(self.key_cid)
            self.key_cid = None

    def plot_scatter(self, df, x_col, y_col, title):
        """Clears axes and plots new scatter data."""
        self.ax.cla() # Clear previous plot
        if df is not None and not df.empty and x_col in df.columns and y_col in df.columns:
            if not (df[x_col].isnull().all() or df[y_col].isnull().all()):
                 self.ax.scatter(df[x_col], df[y_col], s=10, alpha=0.7, label='Data')
                 logging.info(f"Plotting scatter for {x_col} vs {y_col}")
            else:
                 self.ax.text(0.5, 0.5, "No valid data points for selected axes", ha='center', va='center', transform=self.ax.transAxes)
                 logging.warning(f"Cannot plot scatter: All NaN data for {x_col} or {y_col}")
        else:
            self.ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=self.ax.transAxes)
            logging.warning("Cannot plot scatter: DataFrame or columns invalid.")

        self.ax.set_xlabel(x_col if x_col else "X")
        self.ax.set_ylabel(y_col if y_col else "Y")
        self.ax.set_title(title + "\n(Click points; Press 'r' to reset)")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw_idle()

    def add_line(self, x_coords, y_coords, label, color='red', marker='o', markersize=5, linewidth=2):
        """Adds a line segment to the plot."""
        self.ax.plot(x_coords, y_coords, color=color, marker=marker, markersize=markersize, linewidth=linewidth, label=label)
        self.ax.legend()
        self.canvas.draw_idle()

    def add_temp_marker(self, x, y, color='red', marker='o', markersize=8):
        """Adds a temporary marker (like for P1)."""
        # Store reference to remove later if needed, or just let reset clear it
        self.ax.plot(x, y, color=color, marker=marker, markersize=markersize, label='_nolegend_')
        self.canvas.draw_idle()

    def add_text(self, x, y, text, color='red', fontsize=9, **kwargs):
         """Adds text annotation to the plot."""
         self.ax.text(x, y, text, color=color, fontsize=fontsize, **kwargs)
         self.canvas.draw_idle()

    def redraw(self):
        """Force redraw of the canvas."""
        self.canvas.draw_idle()