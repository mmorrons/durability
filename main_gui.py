# main_gui.py
import tkinter as tk
from tkinter import ttk # Themed widgets
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import logging
import os

# Import from our custom modules
import data_processing as dp
from matplotlib_embed import MatplotlibEmbed

# --- Configuration ---
APP_TITLE = "📊 GUI Segment Analyzer"
LOG_LEVEL = logging.INFO

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s-%(levelname)s-%(message)s')

# --- Main Application Class ---
class SegmentAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x800") # Adjust size as needed

        # Data Storage
        self.processed_data = {} # Key: unique_key, Value: {'metadata':..., 'prepared_df':..., 'segments':[], 'p1':None}
        self.current_selected_key = None
        self.current_df_display = None # Holds the potentially smoothed data for plotting

        # Plotting State (managed within the MatplotlibEmbed now)
        # But we need to track logical state for segment building here
        self.clicked_point_1 = None

        # Build UI
        self._setup_ui()

    def _setup_ui(self):
        # --- Main Layout Frames ---
        self.control_frame = ttk.Frame(self, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.plot_frame = ttk.Frame(self, padding="10")
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # --- Control Frame Widgets ---
        # File Loading
        self.load_button = ttk.Button(self.control_frame, text="Load File(s)", command=self.load_files)
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.dataset_label = ttk.Label(self.control_frame, text="Select Dataset:")
        self.dataset_label.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(self.control_frame, textvariable=self.dataset_var, state="readonly", width=40)
        self.dataset_dropdown.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.dataset_dropdown.bind("<<ComboboxSelected>>", self.on_dataset_select)

        # Parameter Selection
        param_frame = ttk.LabelFrame(self.control_frame, text="Plot Parameters", padding="5")
        param_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        self.smooth_label = ttk.Label(param_frame, text="Smoothing:")
        self.smooth_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.smooth_var = tk.StringVar(value="Raw Data")
        smoothing_opts = ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"] # Example options
        self.smooth_dropdown = ttk.Combobox(param_frame, textvariable=self.smooth_var, values=smoothing_opts, state="readonly", width=18)
        self.smooth_dropdown.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        self.xcol_label = ttk.Label(param_frame, text="X-Axis:")
        self.xcol_label.grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.xcol_var = tk.StringVar()
        self.xcol_dropdown = ttk.Combobox(param_frame, textvariable=self.xcol_var, state="readonly", width=15)
        self.xcol_dropdown.grid(row=0, column=3, padx=5, pady=2, sticky="w")

        self.ycol_label = ttk.Label(param_frame, text="Y-Axis:")
        self.ycol_label.grid(row=0, column=4, padx=5, pady=2, sticky="w")
        self.ycol_var = tk.StringVar()
        self.ycol_dropdown = ttk.Combobox(param_frame, textvariable=self.ycol_var, state="readonly", width=15)
        self.ycol_dropdown.grid(row=0, column=5, padx=5, pady=2, sticky="w")

        self.plot_button = ttk.Button(param_frame, text="Update Plot", command=self.update_plot)
        self.plot_button.grid(row=0, column=6, padx=10, pady=2, sticky="e")

        # Reset Buttons Frame
        reset_frame = ttk.Frame(self.control_frame)
        reset_frame.grid(row=2, column=0, columnspan=3, pady=5)
        self.reset_p1_button = ttk.Button(reset_frame, text="Clear Current Click (P1)", command=self.clear_current_click)
        self.reset_p1_button.pack(side=tk.LEFT, padx=5)
        self.reset_segs_button = ttk.Button(reset_frame, text="Reset All Segments", command=self.reset_all_segments)
        self.reset_segs_button.pack(side=tk.LEFT, padx=5)

        # Status Bar (Optional)
        self.status_var = tk.StringVar(value="Ready. Load file(s) to begin.")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)


        # --- Plot Frame Setup ---
        self.plot_manager = MatplotlibEmbed(self.plot_frame)
        self.plot_manager.connect_events(self.on_plot_click, self.on_plot_key)

    # --- Callback Functions ---
    def load_files(self):
        filepaths = filedialog.askopenfilenames(
            title="Select one or more Excel/CSV files",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepaths:
            self.set_status("File loading cancelled.")
            return

        new_files_count = 0
        error_files = []

        # --- Corrected Progress Window Handling ---
        progress_win = tk.Toplevel(self) # Create the window normally
        progress_win.title("Loading...")
        progress_win.geometry("300x50+100+100") # Position near top-left
        progress_win.resizable(False, False)    # Prevent resizing
        progress_win.transient(self)           # Associate with main window
        prog_label = ttk.Label(progress_win, text="Processing files...")
        prog_label.pack(pady=10, padx=10)
        progress_win.update() # Make window visible
        # --- End Correction ---

        try: # Use try...finally to ensure window is closed
            for fpath in filepaths:
                filename = os.path.basename(fpath)
                # Update progress label
                prog_label.config(text=f"Processing: {filename}...")
                progress_win.update() # Allow GUI to refresh

                self.set_status(f"Processing: {filename}...") # Update main status bar

                metadata, unique_key = dp.parse_filename(filename)
                if unique_key is None: unique_key = filename

                if unique_key not in self.processed_data:
                    raw_df = dp.load_data(fpath)
                    if raw_df is not None:
                        # Add filename attribute for cached functions if needed
                        # (Already done inside load_data/prepare_data)
                        prep_df = dp.prepare_data(raw_df)
                        if prep_df is not None and not prep_df.empty:
                            self.processed_data[unique_key] = {
                                'metadata': metadata if metadata else {'filename': filename, 'display_name': filename},
                                'prepared_df': prep_df,
                                'segments': [],
                                'p1': None # Initialize point 1 state
                            }
                            new_files_count += 1
                            logging.info(f"Processed key: {unique_key}")
                        else:
                            error_files.append(filename)
                            logging.error(f"Preparation failed for {filename}")
                    else:
                        error_files.append(filename)
                        logging.error(f"Loading failed for {filename}")
                else:
                     logging.info(f"Skipping already loaded file: {filename} (Key: {unique_key})")

        finally:
            # --- Ensure progress window is destroyed ---
            progress_win.destroy() # Explicitly close the window
            # --- End Ensure ---

        # --- Update UI after processing ---
        if error_files:
             messagebox.showerror("Loading Errors", f"Failed to load or prepare:\n- " + "\n- ".join(error_files))

        self._update_dataset_dropdown() # Refresh dropdown list

        if new_files_count > 0:
             self.set_status(f"Loaded {new_files_count} new file(s). Select a dataset.")
        elif not error_files:
             self.set_status("No new valid files loaded (already processed or cancelled).")
        else:
             self.set_status(f"Load complete with errors for {len(error_files)} file(s).")

    def _update_dataset_dropdown(self):
        """Updates the dataset dropdown based on loaded data."""
        display_options = {
            key: data['metadata'].get('display_name', key)
            for key, data in self.processed_data.items()
        }
        sorted_keys = sorted(display_options, key=display_options.get)
        display_names = [display_options[key] for key in sorted_keys]

        self.dataset_dropdown['values'] = display_names
        self.dataset_keys = sorted_keys # Store keys in order

        if display_names:
            if self.current_selected_key is None or self.current_selected_key not in self.processed_data:
                # Select the first one if nothing is selected or previous selection gone
                self.dataset_var.set(display_names[0])
                self.on_dataset_select(event=None) # Trigger update for the first item
            else:
                 # Reselect the current one if it still exists
                 try:
                      current_display_name = self.processed_data[self.current_selected_key]['metadata']['display_name']
                      current_index = display_names.index(current_display_name)
                      self.dataset_dropdown.current(current_index)
                 except (ValueError, KeyError):
                      # Fallback if current key/name somehow missing
                      self.dataset_var.set(display_names[0])
                      self.on_dataset_select(event=None)

            self.dataset_dropdown.config(state="readonly")
        else:
            self.dataset_var.set("")
            self.dataset_dropdown.config(state="disabled")
            self.xcol_dropdown.config(values=[], state="disabled")
            self.ycol_dropdown.config(values=[], state="disabled")

    def on_dataset_select(self, event=None):
        """Handles selection change in the dataset dropdown."""
        try:
            selected_display_name = self.dataset_var.get()
            if not selected_display_name: return

            # Find the key corresponding to the display name
            selected_index = self.dataset_dropdown.current()
            if selected_index < 0 or selected_index >= len(self.dataset_keys):
                logging.warning("Dataset dropdown index out of sync.")
                return
            self.current_selected_key = self.dataset_keys[selected_index]

            logging.info(f"Dataset selected: {selected_display_name} (Key: {self.current_selected_key})")
            self.set_status(f"Selected: {selected_display_name}. Choose parameters and Update Plot.")

            # Reset analysis state for this dataset
            self.clicked_point_1 = None
            # Segments are stored per dataset, no need to clear globally

            # Update column dropdowns
            self._update_column_dropdowns()
            # Trigger initial plot update for the selected dataset
            self.update_plot()

        except Exception as e:
             messagebox.showerror("Error", f"Failed to handle dataset selection: {e}")
             logging.error("Dataset selection error", exc_info=True)


    def _update_column_dropdowns(self):
        """Populates X and Y column dropdowns based on selected dataset."""
        if self.current_selected_key and self.current_selected_key in self.processed_data:
            df = self.processed_data[self.current_selected_key]['prepared_df']
            if df is not None:
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols:
                    self.xcol_dropdown.config(values=numeric_cols, state="readonly")
                    # Try to set default X to time_seconds
                    try:
                         default_x_idx = numeric_cols.index(dp.TIME_COL_SECONDS)
                         self.xcol_dropdown.current(default_x_idx)
                    except ValueError:
                         self.xcol_dropdown.current(0) # Default to first if time not found
                    self.xcol_var.set(self.xcol_dropdown.get()) # Update variable

                    # Update Y dropdown based on X selection
                    self._update_y_dropdown()
                    return # Success

        # Disable if no data or no numeric columns
        self.xcol_dropdown.config(values=[], state="disabled"); self.xcol_var.set("")
        self.ycol_dropdown.config(values=[], state="disabled"); self.ycol_var.set("")

    def _update_y_dropdown(self):
         """Updates Y dropdown based on current X selection."""
         x_selection = self.xcol_var.get()
         all_numeric = self.xcol_dropdown['values']
         if x_selection and all_numeric:
              y_options = [col for col in all_numeric if col != x_selection]
              self.ycol_dropdown.config(values=y_options, state="readonly")
              if y_options:
                   # Try setting a default Y column
                   default_y_index = 0
                   common_y_defaults = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', 'W', 'FC']
                   for i, yc in enumerate(y_options):
                        if yc in common_y_defaults:
                             default_y_index = i
                             break
                   self.ycol_dropdown.current(default_y_index)
                   self.ycol_var.set(self.ycol_dropdown.get()) # Update variable
              else:
                   self.ycol_var.set("") # No options left
         else:
              self.ycol_dropdown.config(values=[], state="disabled")
              self.ycol_var.set("")

    def update_plot(self):
        """Applies smoothing and updates the embedded Matplotlib plot."""
        if not self.current_selected_key:
            self.set_status("No dataset selected.")
            # Optionally clear plot here: self.plot_manager.plot_scatter(None, None, None, "No Dataset Selected")
            return

        selected_data = self.processed_data[self.current_selected_key]
        df_prepared = selected_data['prepared_df']
        smoothing_method = self.smooth_var.get()
        x_col = self.xcol_var.get()
        y_col = self.ycol_var.get()

        if not x_col or not y_col:
            self.set_status("Select X and Y axis columns.")
            # Optionally clear plot: self.plot_manager.plot_scatter(None, None, None, "Select Axes")
            return

        self.set_status(f"Applying {smoothing_method} and plotting {y_col} vs {x_col}...")
        self.update_idletasks() # Update GUI status message

        self.current_df_display = dp.apply_smoothing(df_prepared, smoothing_method, dp.TIME_COL_SECONDS)

        if self.current_df_display is None or self.current_df_display.empty:
            messagebox.showerror("Plot Error", f"No data available after applying smoothing '{smoothing_method}'.")
            self.set_status(f"Error: No data after '{smoothing_method}'.")
            self.plot_manager.plot_scatter(None, None, None, "No Data After Smoothing")
            return

        if x_col not in self.current_df_display.columns or y_col not in self.current_df_display.columns:
             messagebox.showerror("Plot Error", f"Selected columns '{x_col}' or '{y_col}' not found after smoothing.")
             self.set_status("Error: Columns missing after smoothing.")
             self.plot_manager.plot_scatter(self.current_df_display, None, None, "Columns Missing")
             return

        title = f"{y_col} vs {x_col} ({smoothing_method})"
        # Plot initial scatter - this clears previous segments/markers too
        self.plot_manager.plot_scatter(self.current_df_display, x_col, y_col, title)

        # Re-plot existing segments for this dataset
        segments_list = selected_data.get('segments', [])
        if isinstance(segments_list, list):
            for i, segment in enumerate(segments_list):
                 if segment and 'start' in segment and 'end' in segment and 'slope' in segment:
                    p1, p2, slope = segment['start'], segment['end'], segment['slope']
                    label = f'Seg {i+1} (m={slope:.2f})'
                    self.plot_manager.add_line([p1[0], p2[0]], [p1[1], p2[1]], label=label)
                    mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                    self.plot_manager.add_text(mid_x, mid_y, f' m={slope:.2f}', va='bottom')

        # Clear any lingering P1 click state from previous plots
        self.clicked_point_1 = None
        self.set_status(f"Plot updated for {y_col} vs {x_col}. Click points to define segments.")

    def on_plot_click(self, event):
        """Callback for mouse clicks on the embedded Matplotlib plot."""
        if not event.inaxes == self.plot_manager.ax: return # Click outside plot axes
        x, y = event.xdata, event.ydata
        if x is None or y is None: return # Click outside data limits

        current_key = self.current_selected_key
        if not current_key: return # No dataset selected

        print(f"Clicked Plot at: ({x:.2f}, {y:.2f})") # Print to console

        if self.clicked_point_1 is None:
            # First click
            self.clicked_point_1 = (x, y)
            self.plot_manager.add_temp_marker(x, y) # Show visual feedback
            self.set_status("P1 selected. Click second point (P2).")
        else:
            # Second click
            p1 = self.clicked_point_1
            p2 = (x, y)
            self.clicked_point_1 = None # Reset for next segment

            # Check if points are different enough
            if abs(p1[0] - p2[0]) < 1e-6 and abs(p1[1] - p2[1]) < 1e-6:
                self.set_status("P2 cannot be same as P1. Select P1 again.")
                # Re-plot without the temp marker - need a way to remove specific artist
                # For now, just prompt user; reset will clear it.
                return

            slope = dp.calculate_slope(p1, p2)
            segment_info = {'start': p1, 'end': p2, 'slope': slope}

            # Ensure segments list exists
            if 'segments' not in self.processed_data[current_key]:
                 self.processed_data[current_key]['segments'] = []
            self.processed_data[current_key]['segments'].append(segment_info)
            segment_count = len(self.processed_data[current_key]['segments'])

            print(f"Segment {segment_count} added for {current_key}. Slope (m) = {slope:.4f}")

            # Draw the segment on the plot
            label = f'Seg {segment_count} (m={slope:.2f})'
            self.plot_manager.add_line([p1[0], p2[0]], [p1[1], p2[1]], label=label)
            mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            self.plot_manager.add_text(mid_x, mid_y, f' m={slope:.2f}', va='bottom')

            self.set_status(f"Segment {segment_count} added. Select P1 for next segment.")

    def on_plot_key(self, event):
        """Callback for key presses on the embedded Matplotlib plot."""
        print(f"Key pressed on plot: {event.key}") # Console log
        if event.key == 'r' or event.key == 'R':
            self.reset_all_segments()

    def clear_current_click(self):
        """Clears the currently selected first point (P1)."""
        if self.clicked_point_1:
            self.clicked_point_1 = None
            self.set_status("P1 selection cleared. Select P1 again.")
            # Need to redraw plot to remove temporary marker
            self.update_plot() # Replots scatter and existing segments
        else:
            self.set_status("No P1 selected to clear.")

    def reset_all_segments(self):
        """Clears all segments for the currently selected dataset."""
        if not self.current_selected_key: return

        logging.info(f"Resetting segments for {self.current_selected_key}")
        self.clicked_point_1 = None
        if 'segments' in self.processed_data[self.current_selected_key]:
            self.processed_data[self.current_selected_key]['segments'] = []

        self.set_status("Segments reset. Plot updated.")
        # Redraw the plot without the segments
        self.update_plot()

    def set_status(self, message):
        """Updates the status bar message."""
        self.status_var.set(message)
        logging.info(f"Status: {message}")


# --- Run Application ---
if __name__ == "__main__":
    app = SegmentAnalyzerApp()
    app.mainloop()