# streamlit_app.py (Multi-Section Version - Full Updated Code)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import os
import io
import time # For unique IDs
from datetime import datetime

# --- Import Data Processing Logic ---
# Ensure data_processing.py is in the same directory
try:
    import data_processing as dp
except ImportError:
    st.error("Fatal Error: `data_processing.py` not found. Please ensure it's in the same directory as this script.")
    st.stop()

# --- Configuration ---
APP_TITLE = "Multi-Analysis Segmenter (v2)"
LOG_LEVEL = logging.INFO
# Configure root logger
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a specific logger for the app

# --- Helper Function for Plotting ---
def create_plot(df_display, x_col, y_col, smoothing_method, metadata, segments_list, section_id):
    """Creates the Plotly figure for display, adding section ID to title."""
    fig = go.Figure()
    dataset_name = metadata.get('display_name', f"Data {section_id}")
    plot_title = f"Section {section_id}: {y_col} vs {x_col} ({smoothing_method})<br>{dataset_name}" # Multi-line title

    # Basic data validation for plotting
    plot_error = False
    err_msg = ""
    if df_display is None or df_display.empty: plot_error = True; err_msg = "No data to plot (apply smoothing?)."
    elif not x_col or not y_col: plot_error = True; err_msg = "X or Y axis not selected."
    elif x_col not in df_display.columns: plot_error = True; err_msg = f"X-col '{x_col}' not found in smoothed data."
    elif y_col not in df_display.columns: plot_error = True; err_msg = f"Y-col '{y_col}' not found in smoothed data."
    elif df_display[x_col].isnull().all(): plot_error = True; err_msg = f"All X-axis ('{x_col}') data is invalid/NaN."
    elif df_display[y_col].isnull().all(): plot_error = True; err_msg = f"All Y-axis ('{y_col}') data is invalid/NaN."

    if plot_error:
        logger.warning(f"Plot Error (Section {section_id}) - {err_msg}")
        fig.add_annotation(text=f"Cannot plot: {err_msg}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f"Section {section_id}: Plot Error", xaxis_title=x_col or "X", yaxis_title=y_col or "Y")
        return fig

    # Add main scatter trace
    try:
        fig.add_trace(go.Scattergl(
            x=df_display[x_col], y=df_display[y_col], mode='markers',
            marker=dict(color='blue', size=5, opacity=0.7), name=f'Data',
            hovertemplate=f"<b>X ({x_col})</b>: %{{x:.2f}}<br><b>Y ({y_col})</b>: %{{y:.2f}}<extra></extra>" # Correct hover template format
        ))
    except Exception as e:
         logger.error(f"Error adding scatter trace (Sect {section_id}): {e}", exc_info=True)
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
                         logger.warning(f"Invalid point data in segment {i+1} (Sect {section_id}): start={p_s}, end={p_e}")
                 except Exception as e_seg:
                     logger.warning(f"Could not plot segment {i+1} (Sect {section_id}): {e_seg}", exc_info=True)
             else:
                 logger.warning(f"Invalid segment structure index {i} (Sect {section_id}): {segment}")

    fig.update_layout(
        title=plot_title,
        xaxis_title=x_col, yaxis_title=y_col,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode='select' # Enable selection tool
    )
    return fig

# --- Streamlit App Initialization ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.info("""
**Instructions:**
1. Click "Add Analysis Section" to create a new independent analysis area.
2. In each section:
    a. Upload **one** `.xlsx` or `.csv` file.
    b. Configure Smoothing, X-Axis, and Y-Axis. The plot updates automatically.
    c. Define segments by clicking points on the plot (fills P1/P2 inputs) or typing coordinates manually.
    d. Click "Add Segment" within the section.
    e. Use "Clear Inputs" or "Reset All Segments" within the section as needed.
    f. Click "Remove Section" to delete an analysis area.
""")

# --- Session State Initialization for Multi-Section ---
if 'analysis_sections' not in st.session_state:
    st.session_state.analysis_sections = [] # List to hold each section's state

def add_new_section():
    """Adds a new dictionary structure for a section to the list."""
    # Create a more robust unique ID using timestamp and list length
    section_id = f"S{int(time.time()*100)}_{len(st.session_state.analysis_sections)}"
    st.session_state.analysis_sections.append({
        'id': section_id,
        'file_info': None, # {'name': ..., 'unique_key': ...}
        'prepared_df': None,
        'metadata': {},
        'plot_config': {'smooth': "Raw Data", 'x_col': None, 'y_col': None},
        'df_display': None, # To store the smoothed data for plotting
        'segments': [],
        'manual_input': {'x1': None, 'y1': None, 'x2': None, 'y2': None},
        'selection_target': 'P1',
        'last_select_event': None,
        'plot_fig': None # Cached plot figure for this section
    })
    logger.info(f"Added new section with ID: {section_id}")

def remove_section(section_id_to_remove):
    """Removes a section from the list by its ID."""
    initial_len = len(st.session_state.analysis_sections)
    st.session_state.analysis_sections = [
        sec for sec in st.session_state.analysis_sections if sec['id'] != section_id_to_remove
    ]
    final_len = len(st.session_state.analysis_sections)
    if final_len < initial_len:
        logger.info(f"Removed section with ID: {section_id_to_remove}")
    else:
        logger.warning(f"Attempted to remove section {section_id_to_remove}, but it was not found.")
    # No explicit rerun needed here, Streamlit handles the change in the next cycle

# --- Control Button to Add Sections ---
st.button("➕ Add Analysis Section", on_click=add_new_section, key="add_section_btn")

# --- Render Each Analysis Section ---
if not st.session_state.analysis_sections:
    st.warning("Click 'Add Analysis Section' to begin.")

# Iterate directly over the list. Streamlit's execution model handles changes.
for section in st.session_state.analysis_sections:
    section_id = section['id']

    # Safely get the file name for the title (Incorporating the fix)
    file_info = section.get('file_info') # Get the file_info dict/None
    file_name_for_title = "None" # Default title part
    if isinstance(file_info, dict): # Check if it's a dictionary
        file_name_for_title = file_info.get('name', "Unknown") # Get name if dict

    expander_title = f"Analysis Section {section_id} (File: {file_name_for_title})"
    with st.expander(expander_title, expanded=True):

        # Use columns within the expander for layout
        col1, col2, col3 = st.columns([1, 3, 1.5]) # Config | Plot | Input

        # --- Column 1: File Upload & Config ---
        with col1:
            st.subheader("1. Load & Configure")
            # A. File Upload (Unique key per section)
            uploaded_file = st.file_uploader(
                f"Upload Data File ({section_id})", # Label includes ID for clarity
                type=["xlsx", "xls", "csv"],
                key=f"uploader_{section_id}", # UNIQUE KEY
                accept_multiple_files=False # Only one file per section
            )

            # Process file if uploaded FOR THIS SECTION
            if uploaded_file is not None:
                # Check if it's a *new* file (based on name) for this section
                current_file_name = section.get('file_info', {}).get('name')
                if current_file_name != uploaded_file.name:
                    logger.info(f"Processing new file '{uploaded_file.name}' for section {section_id}")
                    # Reset section state before processing new file
                    section['prepared_df'] = None
                    section['metadata'] = {}
                    section['plot_config'] = {'smooth': "Raw Data", 'x_col': None, 'y_col': None}
                    section['df_display'] = None
                    section['segments'] = []
                    section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                    section['selection_target'] = 'P1'
                    section['last_select_event'] = None
                    section['plot_fig'] = None
                    # Now process
                    try:
                        filename = uploaded_file.name
                        metadata, unique_key = dp.parse_filename(filename)
                        if unique_key is None: unique_key = filename # Fallback key

                        file_content_buffer = io.BytesIO(uploaded_file.getvalue())
                        # Call load_data (might be cached)
                        raw_df = dp.load_data(file_content_buffer, filename)
                        if raw_df is not None:
                            # Call prepare_data (might be cached)
                            prepared_df = dp.prepare_data(raw_df)
                            if prepared_df is not None and not prepared_df.empty:
                                # Update section state successfully
                                section['file_info'] = {'name': filename, 'unique_key': unique_key}
                                section['prepared_df'] = prepared_df
                                section['metadata'] = metadata or {'filename': filename, 'display_name': filename}
                                st.success(f"Loaded: {filename}")
                                # Rerun to update UI based on new data & reset config
                                st.rerun()
                            else:
                                st.error(f"Data preparation failed for {filename}. Check logs."); section['prepared_df'] = None
                        else:
                            st.error(f"File loading failed for {filename}. Check logs."); section['prepared_df'] = None
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        logger.error(f"File processing error (Sect {section_id}): {e}", exc_info=True)
                        section['prepared_df'] = None
                        # Store filename even if failed, helps title consistency
                        section['file_info'] = {'name': uploaded_file.name, 'unique_key': uploaded_file.name}


            # B. Configuration (Only if data is loaded for this section)
            if section['prepared_df'] is not None:
                st.markdown("---")
                df_prepared_sec = section['prepared_df']
                config = section['plot_config'] # Direct reference to modify section's dict
                needs_plot_update = False # Flag to redraw plot if config changes THIS cycle

                # --- Smoothing Selector ---
                smoothing_options = ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"]
                current_smooth = config['smooth']
                smooth_idx = smoothing_options.index(current_smooth) if current_smooth in smoothing_options else 0
                new_smooth = st.selectbox(
                    f"Smoothing ({section_id}):", options=smoothing_options, index=smooth_idx, key=f"smooth_{section_id}"
                )
                if new_smooth != config['smooth']:
                    logger.debug(f"Config change [Smooth]: {config['smooth']} -> {new_smooth} (Sect {section_id})")
                    config['smooth'] = new_smooth
                    needs_plot_update = True

                # --- Apply smoothing TEMPORARILY to get columns ---
                temp_smoothed_df = None
                numeric_cols = []
                try:
                    temp_smoothed_df = dp.apply_smoothing(df_prepared_sec, config['smooth'], dp.TIME_COL_SECONDS)
                    if temp_smoothed_df is not None and not temp_smoothed_df.empty:
                         numeric_cols = temp_smoothed_df.select_dtypes(include=np.number).columns.tolist()
                    if not numeric_cols:
                         logger.warning(f"No numeric columns after temp smoothing (Sect {section_id}). Falling back.")
                         numeric_cols = df_prepared_sec.select_dtypes(include=np.number).columns.tolist()
                except Exception as e_smooth_cols:
                    st.error(f"Axis selection unavailable due to smoothing error: {e_smooth_cols}")
                    logger.error(f"Temp smoothing failed (Sect {section_id})", exc_info=True)
                    numeric_cols = df_prepared_sec.select_dtypes(include=np.number).columns.tolist() # Fallback

                if not numeric_cols:
                     st.error("No numeric columns found in the prepared data."); continue

                # --- X-Axis Selector ---
                current_x = config['x_col']
                # Try to keep current selection if valid, else default
                default_x = current_x if current_x in numeric_cols else (dp.TIME_COL_SECONDS if dp.TIME_COL_SECONDS in numeric_cols else numeric_cols[0])
                try: x_idx = numeric_cols.index(default_x)
                except ValueError: x_idx = 0 # Should not happen if default_x is from numeric_cols
                new_x = st.selectbox(
                    f"X-Axis ({section_id}):", options=numeric_cols, index=x_idx, key=f"x_{section_id}"
                )
                if new_x != config['x_col']:
                    logger.debug(f"Config change [X]: {config['x_col']} -> {new_x} (Sect {section_id})")
                    config['x_col'] = new_x
                    config['y_col'] = None # Force Y re-selection if X changes
                    needs_plot_update = True

                # --- Y-Axis Selector ---
                y_options = [c for c in numeric_cols if c != config['x_col']]
                if not y_options: st.error("No compatible Y-axis options available."); continue
                current_y = config['y_col']
                # Try to keep current selection if valid, else default
                default_y = None
                if current_y in y_options: default_y = current_y
                else: # Try common defaults
                    common_y_defaults = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', dp.WATT_COL, 'FC']
                    # Find first common default in options
                    for yc in common_y_defaults:
                        if yc in y_options: default_y = yc; break
                    if default_y is None: default_y = y_options[0] # Fallback to first available
                try: y_idx = y_options.index(default_y)
                except ValueError: y_idx = 0 # Should not happen if default_y is from y_options
                new_y = st.selectbox(
                    f"Y-Axis ({section_id}):", options=y_options, index=y_idx, key=f"y_{section_id}"
                )
                if new_y != config['y_col']:
                    logger.debug(f"Config change [Y]: {config['y_col']} -> {new_y} (Sect {section_id})")
                    config['y_col'] = new_y
                    needs_plot_update = True

                # --- Update Smoothed Data for Plotting if Config Changed ---
                # Also update if df_display is missing (e.g., after file load)
                if needs_plot_update or section['df_display'] is None:
                     logger.info(f"Updating smoothed data for plot (Sect {section_id})")
                     try:
                         # Use the already calculated temp_smoothed_df if available and smooth method hasn't changed since calc
                         # This check might be complex, let's re-smooth for simplicity
                         section['df_display'] = dp.apply_smoothing(df_prepared_sec, config['smooth'], dp.TIME_COL_SECONDS)
                         section['plot_fig'] = None # Force plot regeneration as data changed
                         # A rerun might be needed here if plot doesn't auto-update immediately
                         # st.rerun() # Consider adding this if plot updates lag
                     except Exception as e_final_smooth:
                         st.error(f"Failed to apply smoothing for plot: {e_final_smooth}")
                         logger.error(f"Final smoothing failed (Sect {section_id})", exc_info=True)
                         section['df_display'] = None


            else: # If no prepared_df
                 st.caption("Upload a file to configure plot.")

        # --- Column 2: Plot ---
        with col2:
            st.subheader("2. Plot")
            df_display_sec = section.get('df_display')
            config_sec = section.get('plot_config', {})
            x_col_sec = config_sec.get('x_col')
            y_col_sec = config_sec.get('y_col')

            # Check if ready to plot
            if df_display_sec is not None and x_col_sec and y_col_sec:
                # Regenerate plot if figure cache is empty (due to config change or reset)
                if section.get('plot_fig') is None:
                     logger.info(f"Regenerating plot for section {section_id}")
                     section['plot_fig'] = create_plot(
                         df_display_sec, x_col_sec, y_col_sec, config_sec['smooth'],
                         section['metadata'], section['segments'], section_id
                     )

                fig_to_display = section.get('plot_fig')
                if fig_to_display:
                    plot_chart_key = f"chart_{section_id}"
                    event_data = st.plotly_chart(
                        fig_to_display, key=plot_chart_key, # UNIQUE KEY
                        use_container_width=True, on_select="rerun" # Rerun on selection event
                    )

                    # --- Handle Plot Selection Event FOR THIS SECTION ---
                    select_info = event_data.get('select') if event_data else None
                    points_data = select_info.get('points', []) if select_info else None
                    # Check if event data is new *for this section*
                    is_new_event_sec = (points_data is not None and points_data != section.get('last_select_event'))

                    if is_new_event_sec:
                        logger.info(f"Plot select event (Sect {section_id}): {len(points_data)} pts.")
                        section['last_select_event'] = points_data # Store current event data
                        if len(points_data) >= 1:
                            selected_point = points_data[0] # Use first point of selection
                            x_sel, y_sel = selected_point.get('x'), selected_point.get('y')
                            if x_sel is not None and y_sel is not None:
                                target = section['selection_target']
                                mi_sec = section['manual_input'] # Ref to section's manual input
                                if target == 'P1':
                                    mi_sec['x1'] = x_sel; mi_sec['y1'] = y_sel
                                    section['selection_target'] = 'P2'
                                    logger.debug(f"Updated Manual P1 from plot (Sect {section_id})")
                                elif target == 'P2':
                                    mi_sec['x2'] = x_sel; mi_sec['y2'] = y_sel
                                    section['selection_target'] = 'P1'
                                    logger.debug(f"Updated Manual P2 from plot (Sect {section_id})")
                                # Rerun needed to update the number inputs in col3
                                st.rerun()
                            else:
                                logger.warning(f"Selection event point missing coordinate data (Sect {section_id}).")
                        # else: No points in selection (e.g., click off points) - do nothing
                else:
                    # This might happen if plot generation failed
                     st.warning("Plot could not be generated or is pending update.")
            else: # If not ready to plot
                st.caption("Upload data and configure axes/smoothing to display plot.")

        # --- Column 3: Segment Input & Control ---
        with col3:
            st.subheader("3. Segments")
            # Only show if data has been loaded for this section
            if section.get('prepared_df') is not None:
                target_sec = section.get('selection_target', 'P1')
                st.info(f"Next plot click updates: **{target_sec}**")
                mi_sec = section.get('manual_input', {}) # Use .get for safety

                # Use unique keys for all widgets in the loop
                mi_sec['x1'] = st.number_input(f"P1 X ({section_id}):", value=mi_sec.get('x1'), format="%.3f", key=f"num_x1_{section_id}")
                mi_sec['y1'] = st.number_input(f"P1 Y ({section_id}):", value=mi_sec.get('y1'), format="%.3f", key=f"num_y1_{section_id}")
                mi_sec['x2'] = st.number_input(f"P2 X ({section_id}):", value=mi_sec.get('x2'), format="%.3f", key=f"num_x2_{section_id}")
                mi_sec['y2'] = st.number_input(f"P2 Y ({section_id}):", value=mi_sec.get('y2'), format="%.3f", key=f"num_y2_{section_id}")

                # Buttons also need unique keys
                add_button_sec = st.button("Add Segment", key=f"add_{section_id}", use_container_width=True)
                clear_button_sec = st.button("Clear Inputs", key=f"clear_{section_id}", use_container_width=True)
                reset_button_sec = st.button("Reset All Segments", key=f"reset_{section_id}", use_container_width=True)

                # Button Actions - Modify the specific section's state
                if add_button_sec:
                    x1, y1, x2, y2 = mi_sec.get('x1'), mi_sec.get('y1'), mi_sec.get('x2'), mi_sec.get('y2')
                    if None in [x1, y1, x2, y2]:
                        st.error("All coordinates (P1 X/Y, P2 X/Y) must be entered.")
                    else:
                        p1 = (x1, y1); p2 = (x2, y2)
                        # Basic check for distinct points
                        if abs(p1[0] - p2[0]) > 1e-6 or abs(p1[1] - p2[1]) > 1e-6:
                            try:
                                slope = dp.calculate_slope(p1, p2)
                                section['segments'].append({'start': p1, 'end': p2, 'slope': slope})
                                logger.info(f"Added segment {len(section['segments'])} (Sect {section_id}): P1={p1}, P2={p2}, m={slope:.4f}")
                                # Clear inputs and reset target for this section
                                section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                                section['selection_target'] = 'P1'
                                section['plot_fig'] = None # Force plot redraw to show the new segment
                                st.rerun()
                            except Exception as e_slope:
                                st.error(f"Error calculating slope: {e_slope}")
                                logger.error(f"Slope calc error (Sect {section_id})", exc_info=True)
                        else:
                            st.error("P1 and P2 coordinates are too close. Select distinct points.")

                if clear_button_sec:
                    section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                    section['selection_target'] = 'P1'
                    st.rerun()

                if reset_button_sec:
                    section['segments'] = []
                    section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                    section['selection_target'] = 'P1'
                    section['plot_fig'] = None # Force plot redraw
                    logger.info(f"Reset segments (Sect {section_id})")
                    st.rerun()

                # --- Display Segment Table for this section ---
                st.markdown("---")
                st.markdown("**Defined Segments**")
                segments_sec = section.get('segments', [])
                if segments_sec:
                    data_disp_sec = []
                    for i_s, seg_s in enumerate(segments_sec):
                         # Check segment structure before accessing
                         if isinstance(seg_s, dict) and all(k in seg_s for k in ['start', 'end', 'slope']):
                             try:
                                 data_disp_sec.append({
                                    "Seg #": i_s + 1,
                                    "Start X": f"{seg_s['start'][0]:.2f}", "Start Y": f"{seg_s['start'][1]:.2f}",
                                    "End X": f"{seg_s['end'][0]:.2f}", "End Y": f"{seg_s['end'][1]:.2f}",
                                    "Slope (m)": f"{seg_s['slope']:.4f}"
                                 })
                             except (TypeError, IndexError, KeyError) as e_fmt:
                                 logger.warning(f"Error formatting segment {i_s+1} for display (Sect {section_id}): {e_fmt}")
                                 data_disp_sec.append({"Seg #": i_s + 1, "Start X": "Err", "Start Y": "Err", "End X": "Err", "End Y": "Err", "Slope (m)": "Err"})
                         else:
                             logger.warning(f"Skipping invalid segment structure in table (Sect {section_id}): {seg_s}")

                    if data_disp_sec:
                        try:
                            df_segs_sec = pd.DataFrame(data_disp_sec).set_index('Seg #')
                            st.dataframe(df_segs_sec, use_container_width=True)
                        except Exception as e_df:
                            st.error(f"Error displaying segments table: {e_df}")
                            logger.error(f"Segments table DF error (Sect {section_id})", exc_info=True)
                    else:
                        st.caption("No valid segments to display.")
                else: # If segments list is empty
                    st.caption("No segments defined yet for this section.")
            else: # If no prepared_df
                st.caption("Load data in this section to define segments.")

        # --- Section Footer ---
        st.markdown("---")
        # Use args in on_click to pass the section ID to the remove function
        st.button(f"❌ Remove Section {section_id}", key=f"remove_{section_id}", on_click=remove_section, args=(section_id,), type="secondary")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"*{APP_TITLE}*")
# Use markdown for better formatting if needed
st.sidebar.markdown("📍 **Location:** Sassari, Sardinia, Italy")
now_local = datetime.now()
# Attempt to get timezone, fallback
try:
    local_tz_name = now_local.astimezone().tzname()
except:
    local_tz_name = "Local Time"
st.sidebar.caption(f"Timestamp: {now_local.strftime('%Y-%m-%d %H:%M:%S')} ({local_tz_name})")