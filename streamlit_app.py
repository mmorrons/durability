# streamlit_app.py (Rebuilt Version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import os
import io
from datetime import datetime

# --- Import Data Processing Logic ---
# Ensure data_processing.py is in the same directory
try:
    import data_processing as dp
except ImportError:
    st.error("Fatal Error: `data_processing.py` not found.")
    st.stop()

# --- Configuration ---
APP_TITLE = "DUR Split Regression (Rebuilt)"
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s-%(levelname)s-%(message)s')

# --- Helper Function for Plotting ---
def create_plot(df_display, x_col, y_col, smoothing_method, metadata, segments_list):
    """Creates the Plotly figure for display."""
    fig = go.Figure()
    plot_title = f"{y_col} vs {x_col} ({smoothing_method})"
    dataset_name = metadata.get('display_name', 'Unknown Dataset')

    # Basic data validation for plotting
    plot_error = False
    err_msg = ""
    if df_display is None or df_display.empty: plot_error = True; err_msg = "No data after smoothing."
    elif x_col not in df_display.columns: plot_error = True; err_msg = f"X-col '{x_col}' missing."
    elif y_col not in df_display.columns: plot_error = True; err_msg = f"Y-col '{y_col}' missing."
    elif df_display[x_col].isnull().all() or df_display[y_col].isnull().all(): plot_error = True; err_msg = f"All NaN data for axes."

    if plot_error:
        logging.warning(f"Plot Error ({dataset_name}) - {err_msg}")
        fig.add_annotation(text=f"Cannot plot: {err_msg}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f"{plot_title} - {dataset_name} [ERROR]", xaxis_title=x_col, yaxis_title=y_col)
        return fig

    # Add main scatter trace
    try:
        fig.add_trace(go.Scattergl(
            x=df_display[x_col], y=df_display[y_col], mode='markers',
            marker=dict(color='blue', size=5, opacity=0.7), name=f'Data',
            hovertemplate=f"<b>X ({x_col})</b>: %{{x:.2f}}<br><b>Y ({y_col})</b>: %{{y:.2f}}<extra></extra>"
        ))
    except Exception as e:
         logging.error(f"Error adding scatter trace: {e}")
         fig.add_annotation(text=f"Plot Error:\n{e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Add Segments
    if isinstance(segments_list, list):
        for i, segment in enumerate(segments_list):
             # Validate segment structure before plotting
             if isinstance(segment, dict) and all(k in segment for k in ['start', 'end', 'slope']):
                 try:
                     p_s, p_e, m = segment['start'], segment['end'], segment['slope']
                     # Ensure points are tuples/lists of numbers
                     if isinstance(p_s, (list, tuple)) and len(p_s) == 2 and all(isinstance(n, (int, float)) for n in p_s) and \
                        isinstance(p_e, (list, tuple)) and len(p_e) == 2 and all(isinstance(n, (int, float)) for n in p_e):
                         fig.add_trace(go.Scatter(
                             x=[p_s[0], p_e[0]], y=[p_s[1], p_e[1]],
                             mode='lines+markers',
                             line=dict(color='red', width=2),
                             marker=dict(color='red', size=8),
                             name=f'Seg{i+1}(m={m:.2f})'
                         ))
                         mid_x, mid_y = (p_s[0] + p_e[0]) / 2, (p_s[1] + p_e[1]) / 2
                         fig.add_annotation(x=mid_x, y=mid_y, text=f' m={m:.2f}', showarrow=False, font=dict(color='red', size=10), xshift=5)
                     else:
                         logging.warning(f"Invalid point data in segment {i+1}: start={p_s}, end={p_e}")
                 except Exception as e_seg:
                     logging.warning(f"Could not plot segment {i+1}: {e_seg}")
             else:
                 logging.warning(f"Invalid segment structure index {i}: {segment}")

    fig.update_layout(
        title=f"{plot_title} - {dataset_name}",
        xaxis_title=x_col, yaxis_title=y_col,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode='select' # Enable selection tool
    )
    return fig

# --- Streamlit App Initialization ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# --- Session State Initialization ---
def initialize_state():
    # Data Storage (Cumulative)
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {} # {unique_key: {'metadata':..., 'prepared_df':...}}

    # Selections Pending User Confirmation (via Apply button)
    if 'pending_selection' not in st.session_state:
        st.session_state.pending_selection = {
            'key': None,
            'smooth': "Raw Data",
            'x_col': None,
            'y_col': None
        }

    # Selections Currently Applied and Displayed
    if 'applied_selection' not in st.session_state:
        st.session_state.applied_selection = {
            'key': None,
            'smooth': "Raw Data",
            'x_col': None,
            'y_col': None
        }

    # Segments for the *currently applied* dataset
    if 'current_segments' not in st.session_state:
        st.session_state.current_segments = []

    # Manual input state
    if 'manual_input' not in st.session_state:
        st.session_state.manual_input = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
    if 'selection_target' not in st.session_state:
        st.session_state.selection_target = 'P1' # 'P1' or 'P2'

    # Plot state
    if 'last_plot_fig' not in st.session_state:
        st.session_state.last_plot_fig = None
    if 'last_select_event' not in st.session_state:
        st.session_state.last_select_event = None

initialize_state()

# --- File Upload & Processing ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Upload `.xlsx` or `.csv` files",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="file_uploader_rebuilt"
    )

    if uploaded_files:
        new_files_processed_count = 0
        error_files_count = 0
        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                metadata, unique_key = dp.parse_filename(filename)
                if unique_key is None: unique_key = filename # Fallback key

                # Process only if not already processed
                if unique_key not in st.session_state.processed_data:
                    logging.info(f"Processing new file: {filename} (Key: {unique_key})")
                    try:
                        file_content_buffer = io.BytesIO(uploaded_file.getvalue())
                        raw_df = dp.load_data(file_content_buffer, filename) # Pass buffer
                        if raw_df is not None:
                            prepared_df = dp.prepare_data(raw_df)
                            if prepared_df is not None and not prepared_df.empty:
                                st.session_state.processed_data[unique_key] = {
                                    'metadata': metadata or {'filename': filename, 'display_name': filename},
                                    'prepared_df': prepared_df
                                }
                                # If this is the first file loaded, make it the pending selection
                                if st.session_state.pending_selection['key'] is None:
                                    st.session_state.pending_selection['key'] = unique_key
                                new_files_processed_count += 1
                            else:
                                logging.error(f"Preparation failed for {filename}")
                                error_files_count += 1
                        else:
                            logging.error(f"Loading failed for {filename}")
                            error_files_count += 1
                    except Exception as e:
                        logging.error(f"Error processing file {filename}: {e}", exc_info=True)
                        error_files_count += 1
                # else: # Optionally log skipped files
                #     logging.debug(f"Skipping already processed file: {filename}")

        if new_files_processed_count > 0:
            st.success(f"Processed {new_files_processed_count} new file(s).")
        if error_files_count > 0:
            st.warning(f"Failed to process {error_files_count} file(s). Check logs.")
        # Clear the uploader state after processing to allow re-uploading same files if needed
        # st.session_state.file_uploader_rebuilt = [] # Might cause issues, test carefully

# --- Main Application Logic ---
if not st.session_state.processed_data:
    st.info("⬅️ Upload data files using the sidebar to begin.")
else:
    # --- Configuration Sidebar ---
    with st.sidebar:
        st.header("2. Configure Plot")

        # A. Select Dataset (Updates pending_selection['key'])
        dataset_options = {k: v['metadata'].get('display_name', k) for k, v in st.session_state.processed_data.items()}
        sorted_keys = sorted(dataset_options, key=dataset_options.get)
        display_names = [dataset_options[key] for key in sorted_keys]

        # Try to find index of current pending key, default to 0 if not found or None
        current_pending_display_name = dataset_options.get(st.session_state.pending_selection['key'])
        try:
            pending_index = display_names.index(current_pending_display_name) if current_pending_display_name else 0
        except ValueError:
            pending_index = 0

        selected_display_name = st.selectbox(
            "Dataset:", options=display_names, index=pending_index, key="sb_dataset"
        )
        # Find the key corresponding to the selected display name
        new_pending_key = next((key for key, name in dataset_options.items() if name == selected_display_name), None)

        # Update pending key if it changed
        if new_pending_key != st.session_state.pending_selection['key']:
            st.session_state.pending_selection['key'] = new_pending_key
            # Reset axes/smoothing when dataset changes? Optional, maybe confusing.
            # st.session_state.pending_selection['x_col'] = None
            # st.session_state.pending_selection['y_col'] = None
            # st.session_state.pending_selection['smooth'] = "Raw Data"
            st.info("Dataset selection changed. Click 'Apply Changes'.") # Inform user

        # Get data for the PENDING key to populate next dropdowns
        pending_key = st.session_state.pending_selection['key']
        if not pending_key:
            st.warning("No dataset selected in processed data.")
            st.stop() # Cannot proceed without a dataset

        df_prepared_pending = st.session_state.processed_data[pending_key]['prepared_df']

        # B. Select Smoothing (Updates pending_selection['smooth'])
        smoothing_options = ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"]
        current_pending_smooth = st.session_state.pending_selection['smooth']
        smooth_idx = smoothing_options.index(current_pending_smooth) if current_pending_smooth in smoothing_options else 0
        st.session_state.pending_selection['smooth'] = st.selectbox(
            "Smoothing:", options=smoothing_options, index=smooth_idx, key="sb_smooth"
        )

        # Apply smoothing TEMPORARILY to get available columns for axis selection
        # This avoids applying smoothing to the main data until "Apply" is clicked
        try:
             df_for_columns = dp.apply_smoothing(df_prepared_pending, st.session_state.pending_selection['smooth'], dp.TIME_COL_SECONDS)
             if df_for_columns is None or df_for_columns.empty: raise ValueError("Smoothing returned no data")
             numeric_cols = df_for_columns.select_dtypes(include=np.number).columns.tolist()
             if not numeric_cols: raise ValueError("No numeric columns after smoothing")
        except Exception as e:
             st.error(f"Axis selection unavailable: Error during temp smoothing ({e})")
             numeric_cols = df_prepared_pending.select_dtypes(include=np.number).columns.tolist() # Fallback
             if not numeric_cols: st.stop() # Cannot proceed


        # C. Select X-Axis (Updates pending_selection['x_col'])
        current_pending_x = st.session_state.pending_selection['x_col']
        # Set default if current is invalid or None
        default_x = current_pending_x if current_pending_x in numeric_cols else (dp.TIME_COL_SECONDS if dp.TIME_COL_SECONDS in numeric_cols else numeric_cols[0])
        try: default_x_idx = numeric_cols.index(default_x)
        except ValueError: default_x_idx = 0
        st.session_state.pending_selection['x_col'] = st.selectbox(
             "X-Axis:", numeric_cols, index=default_x_idx, key="sb_x_axis"
        )

        # D. Select Y-Axis (Updates pending_selection['y_col'])
        current_pending_y = st.session_state.pending_selection['y_col']
        x_selected = st.session_state.pending_selection['x_col']
        y_options = [c for c in numeric_cols if c != x_selected]
        if not y_options: st.error("Only one numeric column available."); st.stop()

        default_y = None
        if current_pending_y in y_options: default_y = current_pending_y
        else: # Try common defaults
            common_y_defaults = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', dp.WATT_COL, 'FC']
            for yc in y_options:
                if yc in common_y_defaults: default_y = yc; break
            if default_y is None: default_y = y_options[0] # Fallback to first available

        try: default_y_idx = y_options.index(default_y)
        except ValueError: default_y_idx = 0
        st.session_state.pending_selection['y_col'] = st.selectbox(
            "Y-Axis:", y_options, index=default_y_idx, key="sb_y_axis"
        )

        # E. Apply Button
        st.markdown("---")
        apply_button = st.button("Apply Changes", key="btn_apply", type="primary", use_container_width=True)

        if apply_button:
            # Validate pending selections before applying
            pending = st.session_state.pending_selection
            if not pending['key'] or not pending['x_col'] or not pending['y_col']:
                 st.error("Cannot apply: Dataset, X-axis, and Y-axis must be selected.")
            else:
                 # Apply pending selections to the applied state
                 st.session_state.applied_selection = pending.copy()
                 # Reset segments when plot configuration changes
                 st.session_state.current_segments = []
                 # Clear manual input and plot selection state
                 st.session_state.manual_input = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                 st.session_state.selection_target = 'P1'
                 st.session_state.last_select_event = None
                 # Force plot regeneration by clearing the cached figure
                 st.session_state.last_plot_fig = None
                 logging.info(f"Applied new configuration: {st.session_state.applied_selection}")
                 st.success("Configuration applied. Plot updated.")
                 # Rerun is implicit after button click

    # --- Main Area Layout ---
    plot_col, input_col = st.columns([3, 1]) # Plot area larger

    # --- Plot Column ---
    with plot_col:
        st.header("3. Plot and Define Segments")

        # Check if configuration has been applied at least once
        applied_key = st.session_state.applied_selection.get('key')
        if not applied_key:
            st.info("⬅️ Configure plot options in the sidebar and click 'Apply Changes'.")
        else:
            # Regenerate plot if needed (config applied or segments changed)
            # Check if the current applied config matches the one used for the last plot
            should_regenerate = False
            if st.session_state.last_plot_fig is None:
                should_regenerate = True # No plot exists yet
            # Add other conditions if needed, e.g., comparing segment list length/content

            if should_regenerate:
                 logging.info("Regenerating plot...")
                 try:
                    applied = st.session_state.applied_selection
                    df_prepared_applied = st.session_state.processed_data[applied['key']]['prepared_df']
                    metadata_applied = st.session_state.processed_data[applied['key']]['metadata']
                    # Apply the *applied* smoothing
                    df_display = dp.apply_smoothing(df_prepared_applied, applied['smooth'], dp.TIME_COL_SECONDS)
                    # Create plot with *applied* config and *current* segments
                    fig = create_plot(df_display, applied['x_col'], applied['y_col'], applied['smooth'],
                                      metadata_applied, st.session_state.current_segments)
                    st.session_state.last_plot_fig = fig # Cache the generated figure
                 except Exception as e_plot:
                    st.error(f"Error generating plot: {e_plot}")
                    logging.error("Plot generation failed", exc_info=True)
                    st.session_state.last_plot_fig = go.Figure() # Store empty fig on error

            # Display the plot (either regenerated or cached)
            fig_to_display = st.session_state.last_plot_fig
            if fig_to_display:
                # Key ensures Plotly state (zoom, etc.) is preserved unless data changes fundamentally
                plot_key = f"plotly_chart_{applied_key}"
                event_data = st.plotly_chart(fig_to_display, key=plot_key, use_container_width=True, on_select="rerun")

                # --- Handle Plot Selection Event ---
                select_info = event_data.get('select') if event_data else None
                points_data = select_info.get('points', []) if select_info else None
                is_new_event = (points_data is not None and points_data != st.session_state.last_select_event)

                if is_new_event:
                    logging.info(f"Plot selection event: {len(points_data)} point(s).")
                    st.session_state.last_select_event = points_data # Store current event data
                    if len(points_data) >= 1:
                        selected_point = points_data[0] # Use first point of selection
                        x_sel, y_sel = selected_point.get('x'), selected_point.get('y')
                        if x_sel is not None and y_sel is not None:
                            target = st.session_state.selection_target
                            if target == 'P1':
                                st.session_state.manual_input['x1'] = x_sel
                                st.session_state.manual_input['y1'] = y_sel
                                st.session_state.selection_target = 'P2' # Move target
                                logging.info(f"Updated Manual P1 from plot: ({x_sel:.2f}, {y_sel:.2f})")
                            elif target == 'P2':
                                st.session_state.manual_input['x2'] = x_sel
                                st.session_state.manual_input['y2'] = y_sel
                                st.session_state.selection_target = 'P1' # Reset target
                                logging.info(f"Updated Manual P2 from plot: ({x_sel:.2f}, {y_sel:.2f})")
                            st.rerun() # Rerun to update the input widgets
                        else:
                            logging.warning("Selection event point missing coordinate data.")
                    else:
                        logging.info("Selection event contained no points (e.g., drag-select release).")
            else:
                # This case might happen if plot generation failed
                 st.error("Plot could not be displayed.")

    # --- Manual Input Column ---
    with input_col:
        st.header("4. Define Segment")
        st.caption("Click plot points or type coords.")

        target = st.session_state.selection_target
        st.info(f"Next plot click updates: **{target}**")

        mi = st.session_state.manual_input # Shorter alias
        mi['x1'] = st.number_input("P1 X:", value=mi['x1'], format="%.3f", key="num_x1")
        mi['y1'] = st.number_input("P1 Y:", value=mi['y1'], format="%.3f", key="num_y1")
        mi['x2'] = st.number_input("P2 X:", value=mi['x2'], format="%.3f", key="num_x2")
        mi['y2'] = st.number_input("P2 Y:", value=mi['y2'], format="%.3f", key="num_y2")

        add_button = st.button("Add Segment", key="btn_add_seg", use_container_width=True)
        clear_button = st.button("Clear Inputs", key="btn_clear_in", use_container_width=True)
        reset_button = st.button("Reset All Segments", key="btn_reset_seg", use_container_width=True)

        if add_button:
            x1, y1, x2, y2 = mi['x1'], mi['y1'], mi['x2'], mi['y2']
            # Check if applied config exists (needed for context, though slope is point-based)
            if not st.session_state.applied_selection.get('key'):
                 st.error("Cannot add segment: Apply a plot configuration first.")
            elif None in [x1, y1, x2, y2]:
                 st.error("All coordinates (P1 X/Y, P2 X/Y) must be entered.")
            else:
                p1 = (x1, y1); p2 = (x2, y2)
                # Basic check for distinct points
                if abs(p1[0] - p2[0]) > 1e-6 or abs(p1[1] - p2[1]) > 1e-6:
                    slope = dp.calculate_slope(p1, p2)
                    new_segment = {'start': p1, 'end': p2, 'slope': slope}
                    st.session_state.current_segments.append(new_segment)
                    seg_count = len(st.session_state.current_segments)
                    logging.info(f"Added segment {seg_count}: P1={p1}, P2={p2}, m={slope:.4f}")
                    # Clear inputs and reset target
                    st.session_state.manual_input = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                    st.session_state.selection_target = 'P1'
                    # Force plot regeneration to show the new segment
                    st.session_state.last_plot_fig = None
                    st.rerun()
                else:
                    st.error("P1 and P2 coordinates are too close. Select distinct points.")

        if clear_button:
            st.session_state.manual_input = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
            st.session_state.selection_target = 'P1'
            st.rerun()

        if reset_button:
            st.session_state.current_segments = []
            st.session_state.manual_input = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
            st.session_state.selection_target = 'P1'
            # Force plot regeneration
            st.session_state.last_plot_fig = None
            logging.info("Reset all segments for the current view.")
            st.rerun()


    # --- Segment Details Table ---
    st.markdown("---")
    st.header("Segment Details")
    segments = st.session_state.current_segments
    if segments:
        data_to_display = []
        for i, seg in enumerate(segments):
            if isinstance(seg, dict) and all(k in seg for k in ['start', 'end', 'slope']):
                try:
                    data_to_display.append({
                        "Seg #": i + 1,
                        "Start X": f"{seg['start'][0]:.2f}", "Start Y": f"{seg['start'][1]:.2f}",
                        "End X": f"{seg['end'][0]:.2f}", "End Y": f"{seg['end'][1]:.2f}",
                        "Slope (m)": f"{seg['slope']:.4f}"
                    })
                except Exception as e_fmt:
                    logging.warning(f"Error formatting segment {i+1} for display: {e_fmt}")
                    data_to_display.append({"Seg #": i + 1, "Start X": "Err", "Start Y": "Err", "End X": "Err", "End Y": "Err", "Slope (m)": "Err"})

        if data_to_display:
            df_segs = pd.DataFrame(data_to_display).set_index('Seg #')
            st.dataframe(df_segs, use_container_width=True)
        else:
            st.caption("No valid segments to display.")
    else:
        st.caption("No segments defined for the current plot configuration.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"*{APP_TITLE}*")
st.sidebar.info(f"📍 Sassari, Sardinia, Italy")
now_local = datetime.now()
timezone_hint = "CEST" # Replace with dynamic timezone if needed
st.sidebar.caption(f"Timestamp: {now_local.strftime('%Y-%m-%d %H:%M:%S')} {timezone_hint}")