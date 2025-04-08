# streamlit_app.py (3-Column Layout, Selection-to-Manual Transfer)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import os
import io
import base64
from datetime import datetime

# Assuming data_processing.py is in the same folder and contains the necessary functions
try:
    import data_processing as dp
except ImportError:
    st.error("Error: `data_processing.py` not found. Ensure it's in the same directory.")
    st.stop()

# --- Configuration ---
APP_TITLE = "DUR Split Regression"
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s-%(levelname)s-%(message)s')

# --- Helper Function for Plotting ---
def create_plot(df, x_col, y_col, smoothing_method, metadata, segments):
    """Creates the Plotly figure, now without P1 marker"""
    fig = go.Figure()
    plot_title = f"{y_col} vs {x_col} ({smoothing_method})"
    dataset_name = metadata.get('display_name', 'Unknown Dataset')

    plot_error = False
    err_msg = ""
    if df is None or df.empty: plot_error = True; err_msg = "No data."
    elif x_col not in df.columns: plot_error = True; err_msg = f"X-col '{x_col}' missing."
    elif y_col not in df.columns: plot_error = True; err_msg = f"Y-col '{y_col}' missing."
    elif df[x_col].isnull().all() or df[y_col].isnull().all(): plot_error = True; err_msg = f"All NaN data for axes."

    if plot_error:
        logging.warning(f"Cannot plot - {err_msg}")
        fig.add_annotation(text=f"Cannot plot: {err_msg}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f"{plot_title} - {dataset_name} [ERROR]", xaxis_title=x_col, yaxis_title=y_col)
        return fig

    # Add main scatter trace
    try:
        fig.add_trace(go.Scattergl(
            x=df[x_col], y=df[y_col], mode='markers',
            marker=dict(color='blue', size=5, opacity=0.7), name=f'Data',
            hovertemplate=f"<b>X ({x_col})</b>: %{{x:.2f}}<br><b>Y ({y_col})</b>: %{{y:.2f}}<extra></extra>"
        ))
    except Exception as e:
         logging.error(f"Error adding scatter trace: {e}")
         fig.add_annotation(text=f"Plot Error:\n{e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Add Segments
    if isinstance(segments, list):
        for i, segment in enumerate(segments):
             if isinstance(segment, dict) and all(k in segment for k in ['start', 'end', 'slope']):
                 try:
                     p_s, p_e, m = segment['start'], segment['end'], segment['slope']
                     fig.add_trace(go.Scatter(x=[p_s[0], p_e[0]], y=[p_s[1], p_e[1]], mode='lines+markers', line=dict(color='red', width=2), marker=dict(color='red', size=8), name=f'Seg{i+1}(m={m:.2f})'))
                     mid_x, mid_y = (p_s[0] + p_e[0]) / 2, (p_s[1] + p_e[1]) / 2; fig.add_annotation(x=mid_x, y=mid_y, text=f' m={m:.2f}', showarrow=False, font=dict(color='red', size=10), xshift=5)
                 except Exception as e_seg: logging.warning(f"Could not plot segment {i+1}: {e_seg}")
             else: logging.warning(f"Invalid segment structure index {i}: {segment}")

    fig.update_layout(
        title=f"{plot_title} - {dataset_name}", xaxis_title=x_col, yaxis_title=y_col,
        hovermode='closest', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode='select' # IMPORTANT: Keep selection mode enabled
    )
    return fig

# --- Streamlit App ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.info("""
**Instructions:**
1. Load file(s). 2. Configure plot (Right Panel).
3. **To add segments:**
    Type P1(X,Y) and P2(X,Y) coords directly (Right Panel), then click 'Add Manual Segment'.
4. Segments show on plot and details below. Use Reset buttons as needed.
""")

# --- Session State Initialization ---
# Data storage
if 'processed_data' not in st.session_state: st.session_state.processed_data = {}
# Global Selections
if 'current_selected_key' not in st.session_state: st.session_state.current_selected_key = None
if 'current_smoothing' not in st.session_state: st.session_state.current_smoothing = "Raw Data"
if 'current_x_col' not in st.session_state: st.session_state.current_x_col = None
if 'current_y_col' not in st.session_state: st.session_state.current_y_col = None
# Analysis State (for the single active plot)
if 'segments' not in st.session_state: st.session_state.segments = []
# State for manual input fields / selection target
if 'manual_x1' not in st.session_state: st.session_state.manual_x1 = None
if 'manual_y1' not in st.session_state: st.session_state.manual_y1 = None
if 'manual_x2' not in st.session_state: st.session_state.manual_x2 = None
if 'manual_y2' not in st.session_state: st.session_state.manual_y2 = None
if 'selection_target' not in st.session_state: st.session_state.selection_target = 'P1' # Target 'P1' or 'P2'
# Store last event to prevent double processing
if 'last_select_event_data' not in st.session_state: st.session_state.last_select_event_data = None

# --- File Upload & Processing ---
uploaded_files = st.file_uploader("Upload Data File(s)", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="file_uploader_main")
# (File processing logic remains the same as previous version)
if uploaded_files and not st.session_state.get('files_processed_flag', False):
    new_files_count = 0; error_files = []
    with st.spinner("Processing..."):
        st.session_state.processed_data = {} # Clear old data on new upload
        st.session_state.current_selected_key = None
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name; file_content_buffer = io.BytesIO(uploaded_file.getvalue())
            metadata, unique_key = dp.parse_filename(filename)
            if unique_key is None: unique_key = filename
            logging.info(f"Processing: {filename}"); raw_df = dp.load_data(file_content_buffer, filename)
            if raw_df is not None:
                prepared_df = dp.prepare_data(raw_df)
                if prepared_df is not None and not prepared_df.empty:
                    st.session_state.processed_data[unique_key] = {'metadata': metadata or {'filename': filename, 'display_name': filename}, 'prepared_df': prepared_df}
                    if st.session_state.current_selected_key is None: st.session_state.current_selected_key = unique_key # Select first successfully processed file
                    new_files_count += 1
                else: error_files.append(filename)
            else: error_files.append(filename)
    if new_files_count > 0: st.success(f"Processed {new_files_count} file(s).")
    if error_files: st.error(f"Failed: {', '.join(error_files)}")
    st.session_state.files_processed_flag = True
    # Reset analysis state on new upload
    st.session_state.segments = []
    st.session_state.manual_x1, st.session_state.manual_y1 = None, None
    st.session_state.manual_x2, st.session_state.manual_y2 = None, None
    st.session_state.selection_target = 'P1'
    st.session_state.last_select_event_data = None
    st.rerun() # Rerun immediately after processing
elif not uploaded_files and st.session_state.get('files_processed_flag'):
     st.session_state.files_processed_flag = False

# --- Main Area ---
if not st.session_state.processed_data: st.warning("Upload data file(s).")
else:
    # --- Layout ---
    plot_col, manual_col, config_col = st.columns([2, 2, 1]) # Adjust ratios as needed

    # --- Configuration Column ---
    with config_col:
        st.subheader("Configuration")

        # Dataset selection
        dataset_options = {k: v['metadata'].get('display_name', k) for k, v in st.session_state.processed_data.items()}
        sorted_keys = sorted(dataset_options, key=dataset_options.get); display_names = [dataset_options[key] for key in sorted_keys]
        current_display_name = dataset_options.get(st.session_state.current_selected_key, None)
        try: current_idx = display_names.index(current_display_name) if current_display_name else 0
        except ValueError: current_idx = 0
        selected_display_name = st.selectbox("Dataset:", options=display_names, key="dataset_selector_display", index=current_idx)
        new_selected_key = next((key for key, name in dataset_options.items() if name == selected_display_name), None)
        if new_selected_key != st.session_state.current_selected_key:
            st.session_state.current_selected_key = new_selected_key
            st.session_state.segments = [] # Reset segments on dataset change
            st.session_state.manual_x1, st.session_state.manual_y1 = None, None
            st.session_state.manual_x2, st.session_state.manual_y2 = None, None
            st.session_state.selection_target = 'P1'
            st.session_state.last_select_event_data = None
            logging.info(f"Dataset changed to: {st.session_state.current_selected_key}. Reset state."); st.rerun()

        selected_key = st.session_state.current_selected_key
        if not selected_key or selected_key not in st.session_state.processed_data: st.error("Dataset error."); st.stop()
        current_metadata = st.session_state.processed_data[selected_key]['metadata']; df_prepared = st.session_state.processed_data[selected_key]['prepared_df']

        # Smoothing selection
        smoothing_options = ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"]
        st.session_state.current_smoothing = st.selectbox("Smoothing:", options=smoothing_options, index=smoothing_options.index(st.session_state.current_smoothing) if st.session_state.current_smoothing in smoothing_options else 0, key="smoothing_selector")

        # Apply smoothing - This happens here, result used by plot and axes selectors
        df_display = dp.apply_smoothing(df_prepared, st.session_state.current_smoothing, dp.TIME_COL_SECONDS)
        if df_display is None or df_display.empty: st.error(f"No data after smoothing."); st.stop()
        numeric_cols = df_display.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols: st.error("No numeric columns."); st.stop()

        # Axis selection
        current_x = st.session_state.get('current_x_col')
        default_x = current_x if current_x in numeric_cols else (dp.TIME_COL_SECONDS if dp.TIME_COL_SECONDS in numeric_cols else numeric_cols[0])
        st.session_state.current_x_col = st.selectbox("X-Axis:", numeric_cols, index=numeric_cols.index(default_x) if default_x in numeric_cols else 0, key=f"x_select")

        current_y = st.session_state.get('current_y_col')
        y_options = [c for c in numeric_cols if c != st.session_state.current_x_col]
        if not y_options: st.error("Only one numeric column."); st.stop()
        default_y = None
        if current_y in y_options: default_y = current_y
        else:
            common_y = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', dp.WATT_COL, 'FC'];
            for yc in y_options:
                if yc in common_y: default_y = yc; break
            if default_y is None: default_y = y_options[0]
        st.session_state.current_y_col = st.selectbox("Y-Axis:", y_options, index=y_options.index(default_y) if default_y in y_options else 0, key=f"y_select")


    # --- Manual Input Column ---
    with manual_col:
        st.subheader("Manual Segment")
        st.info("Use plot select tool (left) or type coords below.")

        # Display area for selection target
        st.markdown(f"**Next Plot Selection will update:** `{st.session_state.selection_target}`")

        # Use session state values for number inputs
        st.session_state.manual_x1 = st.number_input("P1 X:", value=st.session_state.manual_x1, format="%.3f", key="disp_x1_manual")
        st.session_state.manual_y1 = st.number_input("P1 Y:", value=st.session_state.manual_y1, format="%.3f", key="disp_y1_manual")
        st.session_state.manual_x2 = st.number_input("P2 X:", value=None if st.session_state.selection_target == 'P2' else st.session_state.manual_x2, format="%.3f", key="disp_x2_manual") # Clear P2 if target is P2? Maybe not.
        st.session_state.manual_y2 = st.number_input("P2 Y:", value=None if st.session_state.selection_target == 'P2' else st.session_state.manual_y2, format="%.3f", key="disp_y2_manual")

        add_manual_button = st.button("Add Manual Segment", key="add_manual_btn", use_container_width=True)
        clear_manual_button = st.button("Clear Manual Inputs", key="clear_manual_btn", use_container_width=True)
        reset_segments_button = st.button("Reset All Segments", key="reset_segs_manual", use_container_width=True)

        # Handle button clicks
        if clear_manual_button:
            st.session_state.manual_x1, st.session_state.manual_y1 = None, None
            st.session_state.manual_x2, st.session_state.manual_y2 = None, None
            st.session_state.selection_target = 'P1' # Reset target
            st.rerun()

        if reset_segments_button:
            st.session_state.segments = []
            st.session_state.manual_x1, st.session_state.manual_y1 = None, None
            st.session_state.manual_x2, st.session_state.manual_y2 = None, None
            st.session_state.selection_target = 'P1'
            st.rerun()

        if add_manual_button:
            # Read directly from session state as number_input updates it
            x1, y1, x2, y2 = st.session_state.manual_x1, st.session_state.manual_y1, st.session_state.manual_x2, st.session_state.manual_y2
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                p1 = (x1, y1); p2 = (x2, y2)
                if abs(p1[0] - p2[0]) > 1e-6 or abs(p1[1] - p2[1]) > 1e-6:
                    slope = dp.calculate_slope(p1, p2); new_segment = {'start': p1, 'end': p2, 'slope': slope}
                    st.session_state.segments.append(new_segment)
                    segment_count = len(st.session_state.segments)
                    print(f"Manual Seg {segment_count} added. m={slope:.4f}")
                    st.success(f"Manual Segment {segment_count} added.")
                    # Optionally clear fields after successful add
                    st.session_state.manual_x1, st.session_state.manual_y1 = None, None
                    st.session_state.manual_x2, st.session_state.manual_y2 = None, None
                    st.session_state.selection_target = 'P1' # Reset target
                    st.rerun()
                else: st.error("Manual P1 and P2 too close.")
            else: st.error("Enter numeric coordinates for P1 & P2.")

        # Display current segments count in controls
        st.markdown("---")
        st.metric("Segments Defined", len(st.session_state.segments))


    # --- Plot Column ---
    with plot_col:
        st.subheader("Plot")
        x_col_plot = st.session_state.get('current_x_col')
        y_col_plot = st.session_state.get('current_y_col')

        if x_col_plot and y_col_plot:
            fig = create_plot(df_display, x_col_plot, y_col_plot, st.session_state.current_smoothing,
                              current_metadata, st.session_state.segments) # Pass current segments

            # Display plot and capture selection event
            # IMPORTANT: Use a key that doesn't change unnecessarily to preserve state better
            chart_key = f"main_chart_{selected_key}_{x_col_plot}_{y_col_plot}_{st.session_state.current_smoothing}"
            event_data = st.plotly_chart(fig, key=chart_key, use_container_width=True, on_select="rerun")

            # --- Process Selection Event for Click-to-Transfer ---
            select_info = event_data.get('select') if event_data else None
            current_event_data = select_info.get('points', []) if select_info else None

            # Check if this is a new event compared to last run
            is_new_event = (current_event_data is not None and current_event_data != st.session_state.last_select_event_data)

            if is_new_event:
                logging.info(f"Selection event: {len(current_event_data)} pts.")
                if len(current_event_data) >= 1:
                     # Get coords from first selected point
                     selected_point = current_event_data[0]
                     x_sel, y_sel = selected_point.get('x'), selected_point.get('y')

                     if x_sel is not None and y_sel is not None:
                         print(f"Plot selected: ({x_sel:.2f}, {y_sel:.2f})") # Console log

                         target = st.session_state.selection_target
                         if target == 'P1':
                             st.session_state.manual_x1 = x_sel
                             st.session_state.manual_y1 = y_sel
                             st.session_state.selection_target = 'P2' # Set next target
                             st.info("P1 coordinates updated from plot selection. Select P2.")
                             logging.info("Updated P1 from selection.")
                         elif target == 'P2':
                             st.session_state.manual_x2 = x_sel
                             st.session_state.manual_y2 = y_sel
                             st.session_state.selection_target = 'P1' # Cycle back to P1
                             st.info("P2 coordinates updated from plot selection. Click 'Add Manual Segment'.")
                             logging.info("Updated P2 from selection.")

                         # Store the processed event data to prevent re-processing immediately
                         st.session_state.last_select_event_data = current_event_data
                         st.rerun() # Rerun to update the number_input widgets
                     else:
                         logging.warning("Selection event missing coordinate data.")
                         st.session_state.last_select_event_data = current_event_data # Mark as seen
                         st.rerun() # Rerun even if data is bad to clear event
                else:
                     logging.info("Selection event had no points.")
                     # Clear last event data if selection was empty
                     st.session_state.last_select_event_data = None # Use None to indicate no active event processed
            # If it's not a new event, clear the stored last event so next selection works
            elif not is_new_event and st.session_state.last_select_event_data is not None:
                 st.session_state.last_select_event_data = None


        else:
             st.caption("Configure X and Y axes in the control panel.")


    # --- Display Segment Information (Below Columns) ---
    st.markdown("---")
    st.subheader("Defined Segment Details")
    segments = st.session_state.get('segments', [])
    if segments and isinstance(segments, list):
        data_to_display = []
        for i, seg in enumerate(segments):
            if isinstance(seg, dict) and all(k in seg for k in ['start', 'end', 'slope']):
                data_to_display.append({
                    "Seg #": i + 1,
                    "Start X": f"{seg['start'][0]:.2f}", "Start Y": f"{seg['start'][1]:.2f}",
                    "End X": f"{seg['end'][0]:.2f}", "End Y": f"{seg['end'][1]:.2f}",
                    "Slope (m)": f"{seg['slope']:.4f}" })
        st.dataframe(pd.DataFrame(data_to_display).set_index('Seg #'), use_container_width=True)
    else: st.caption("No segments defined yet.")

# --- Footer ---
st.sidebar.markdown("---"); st.sidebar.markdown(f"*{APP_TITLE}*"); st.sidebar.info(f"📍 Sassari, Sardinia, Italy")
now_local = datetime.now(); timezone_hint = "CEST"; st.sidebar.caption(f"Generated: {now_local.strftime('%Y-%m-%d %H:%M:%S')} {timezone_hint}")