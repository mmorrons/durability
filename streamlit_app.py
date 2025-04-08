# streamlit_app.py (Multi-Section Version)

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
try:
    import data_processing as dp
except ImportError:
    st.error("Fatal Error: `data_processing.py` not found.")
    st.stop()

# --- Configuration ---
APP_TITLE = "Multi-Analysis Segmenter"
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s-%(levelname)s-%(message)s')

# --- Helper Function for Plotting (Same as before) ---
def create_plot(df_display, x_col, y_col, smoothing_method, metadata, segments_list, section_id):
    """Creates the Plotly figure for display, adding section ID to title."""
    fig = go.Figure()
    dataset_name = metadata.get('display_name', f"Data {section_id}")
    plot_title = f"Section {section_id}: {y_col} vs {x_col} ({smoothing_method}) - {dataset_name}"

    # Basic data validation
    plot_error = False
    err_msg = ""
    if df_display is None or df_display.empty: plot_error = True; err_msg = "No data (apply smoothing?)."
    elif x_col not in df_display.columns: plot_error = True; err_msg = f"X-col '{x_col}' missing."
    elif y_col not in df_display.columns: plot_error = True; err_msg = f"Y-col '{y_col}' missing."
    elif df_display[x_col].isnull().all() or df_display[y_col].isnull().all(): plot_error = True; err_msg = f"All NaN data for axes."

    if plot_error:
        logging.warning(f"Plot Error (Section {section_id}) - {err_msg}")
        fig.add_annotation(text=f"Cannot plot: {err_msg}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f"{plot_title} [ERROR]", xaxis_title=x_col, yaxis_title=y_col)
        return fig

    # Add main scatter trace
    try:
        fig.add_trace(go.Scattergl(
            x=df_display[x_col], y=df_display[y_col], mode='markers',
            marker=dict(color='blue', size=5, opacity=0.7), name=f'Data',
            hovertemplate=f"<b>X ({x_col})</b>: %{{x:.2f}}<br><b>Y ({y_col})</b>: %{{y:.2f}}<extra></extra>"
        ))
    except Exception as e:
         logging.error(f"Error adding scatter trace (Sect {section_id}): {e}")
         fig.add_annotation(text=f"Plot Error:\n{e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Add Segments
    if isinstance(segments_list, list):
        for i, segment in enumerate(segments_list):
             if isinstance(segment, dict) and all(k in segment for k in ['start', 'end', 'slope']):
                 try:
                     p_s, p_e, m = segment['start'], segment['end'], segment['slope']
                     if isinstance(p_s, (list, tuple)) and len(p_s)==2 and isinstance(p_e, (list, tuple)) and len(p_e)==2:
                         fig.add_trace(go.Scatter(
                             x=[p_s[0], p_e[0]], y=[p_s[1], p_e[1]], mode='lines+markers',
                             line=dict(color='red', width=2), marker=dict(color='red', size=8),
                             name=f'Seg{i+1}(m={m:.2f})'))
                         mid_x, mid_y = (p_s[0] + p_e[0]) / 2, (p_s[1] + p_e[1]) / 2
                         fig.add_annotation(x=mid_x, y=mid_y, text=f' m={m:.2f}', showarrow=False, font=dict(color='red', size=10), xshift=5)
                 except Exception as e_seg: logging.warning(f"Could not plot segment {i+1} (Sect {section_id}): {e_seg}")
             else: logging.warning(f"Invalid segment structure index {i} (Sect {section_id})")

    fig.update_layout(
        title=plot_title, xaxis_title=x_col, yaxis_title=y_col,
        hovermode='closest', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode='select'
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
    section_id = f"{int(time.time())}-{len(st.session_state.analysis_sections)}" # Simple unique ID
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
    logging.info(f"Added new section with ID: {section_id}")

def remove_section(section_id_to_remove):
    """Removes a section from the list by its ID."""
    st.session_state.analysis_sections = [
        sec for sec in st.session_state.analysis_sections if sec['id'] != section_id_to_remove
    ]
    logging.info(f"Removed section with ID: {section_id_to_remove}")

# --- Control Button to Add Sections ---
st.button("➕ Add Analysis Section", on_click=add_new_section, key="add_section_btn")

# --- Render Each Analysis Section ---
if not st.session_state.analysis_sections:
    st.warning("Click 'Add Analysis Section' to begin.")

# Use index for slightly more robust iteration if list length changes during interaction
sections_to_render = st.session_state.analysis_sections.copy() # Iterate over a copy

for i, section in enumerate(sections_to_render):
    section_id = section['id']
    # Use an expander or container for each section
    with st.expander(f"Analysis Section {section_id} (File: {section.get('file_info', {}).get('name', 'None')})", expanded=True):

        # Use columns within the expander for layout
        col1, col2, col3 = st.columns([1, 3, 1.5]) # Config | Plot | Input

        # --- Column 1: File Upload & Config ---
        with col1:
            st.subheader("1. Load & Configure")
            # A. File Upload (Unique key per section)
            uploaded_file = st.file_uploader(
                f"Upload Data File",
                type=["xlsx", "xls", "csv"],
                key=f"uploader_{section_id}", # UNIQUE KEY
                accept_multiple_files=False # Only one file per section
            )

            # Process file if uploaded FOR THIS SECTION
            if uploaded_file is not None:
                # Check if it's a *new* file for this section
                current_file_name = section.get('file_info', {}).get('name')
                if current_file_name != uploaded_file.name:
                    logging.info(f"Processing new file '{uploaded_file.name}' for section {section_id}")
                    try:
                        filename = uploaded_file.name
                        metadata, unique_key = dp.parse_filename(filename)
                        if unique_key is None: unique_key = filename

                        file_content_buffer = io.BytesIO(uploaded_file.getvalue())
                        raw_df = dp.load_data(file_content_buffer, filename)
                        if raw_df is not None:
                            prepared_df = dp.prepare_data(raw_df)
                            if prepared_df is not None and not prepared_df.empty:
                                # Update section state
                                section['file_info'] = {'name': filename, 'unique_key': unique_key}
                                section['prepared_df'] = prepared_df
                                section['metadata'] = metadata or {'filename': filename, 'display_name': filename}
                                # Reset dependent state for this section
                                section['plot_config'] = {'smooth': "Raw Data", 'x_col': None, 'y_col': None}
                                section['df_display'] = None
                                section['segments'] = []
                                section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                                section['selection_target'] = 'P1'
                                section['last_select_event'] = None
                                section['plot_fig'] = None
                                st.success(f"Loaded: {filename}")
                            else: st.error("Preparation failed."); section['prepared_df'] = None
                        else: st.error("Loading failed."); section['prepared_df'] = None
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logging.error(f"File processing error (Sect {section_id}): {e}", exc_info=True)
                        section['prepared_df'] = None
                    # No need to rerun explicitly here, Streamlit handles widget state change

            # B. Configuration (Only if data is loaded for this section)
            if section['prepared_df'] is not None:
                st.markdown("---")
                df_prepared_sec = section['prepared_df']
                config = section['plot_config'] # Reference to section's config dict
                needs_plot_update = False # Flag to redraw plot if config changes

                # Smoothing
                smoothing_options = ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"]
                current_smooth = config['smooth']
                smooth_idx = smoothing_options.index(current_smooth) if current_smooth in smoothing_options else 0
                new_smooth = st.selectbox(
                    "Smoothing:", options=smoothing_options, index=smooth_idx, key=f"smooth_{section_id}" # UNIQUE KEY
                )
                if new_smooth != config['smooth']:
                    config['smooth'] = new_smooth
                    needs_plot_update = True

                # Apply smoothing TEMPORARILY to get columns
                try:
                    df_for_cols = dp.apply_smoothing(df_prepared_sec, config['smooth'], dp.TIME_COL_SECONDS)
                    numeric_cols = df_for_cols.select_dtypes(include=np.number).columns.tolist() if df_for_cols is not None else []
                    if not numeric_cols: st.warning("No numeric columns after smoothing."); numeric_cols = df_prepared_sec.select_dtypes(include=np.number).columns.tolist() # fallback
                except Exception as e_smooth_cols:
                    st.error(f"Smoothing error: {e_smooth_cols}"); numeric_cols = df_prepared_sec.select_dtypes(include=np.number).columns.tolist()

                if not numeric_cols: st.error("No numeric columns found in data."); continue # Skip rest of config if no cols

                # X-Axis
                current_x = config['x_col']
                default_x = current_x if current_x in numeric_cols else (dp.TIME_COL_SECONDS if dp.TIME_COL_SECONDS in numeric_cols else numeric_cols[0])
                try: x_idx = numeric_cols.index(default_x)
                except ValueError: x_idx = 0
                new_x = st.selectbox(
                    "X-Axis:", options=numeric_cols, index=x_idx, key=f"x_{section_id}" # UNIQUE KEY
                )
                if new_x != config['x_col']:
                    config['x_col'] = new_x
                    # Reset Y if X changed, as options depend on X
                    config['y_col'] = None
                    needs_plot_update = True

                # Y-Axis
                y_options = [c for c in numeric_cols if c != config['x_col']]
                if not y_options: st.error("No compatible Y-axis options."); continue
                current_y = config['y_col']
                default_y = None
                if current_y in y_options: default_y = current_y
                else: common_y = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', dp.WATT_COL, 'FC']; [default_y := yc for yc in y_options if yc in common_y]; default_y = default_y if default_y else y_options[0]
                try: y_idx = y_options.index(default_y)
                except ValueError: y_idx = 0
                new_y = st.selectbox(
                    "Y-Axis:", options=y_options, index=y_idx, key=f"y_{section_id}" # UNIQUE KEY
                )
                if new_y != config['y_col']:
                    config['y_col'] = new_y
                    needs_plot_update = True

                # Apply smoothing permanently for plotting if needed
                if needs_plot_update or section['df_display'] is None:
                     try:
                         section['df_display'] = dp.apply_smoothing(df_prepared_sec, config['smooth'], dp.TIME_COL_SECONDS)
                         section['plot_fig'] = None # Force plot regeneration
                         logging.info(f"Updated display data for section {section_id}")
                     except Exception as e_final_smooth:
                         st.error(f"Failed to apply smoothing: {e_final_smooth}")
                         section['df_display'] = None


            else:
                 st.caption("Upload a file to configure plot.")

        # --- Column 2: Plot ---
        with col2:
            st.subheader("2. Plot")
            df_display_sec = section.get('df_display')
            config_sec = section.get('plot_config')
            x_col_sec = config_sec.get('x_col')
            y_col_sec = config_sec.get('y_col')

            if df_display_sec is not None and x_col_sec and y_col_sec:
                # Regenerate plot if figure cache is empty
                if section['plot_fig'] is None:
                     section['plot_fig'] = create_plot(
                         df_display_sec, x_col_sec, y_col_sec, config_sec['smooth'],
                         section['metadata'], section['segments'], section_id
                     )

                fig_to_display = section['plot_fig']
                if fig_to_display:
                    # Use unique key for the chart itself
                    plot_chart_key = f"chart_{section_id}"
                    event_data = st.plotly_chart(
                        fig_to_display, key=plot_chart_key, # UNIQUE KEY
                        use_container_width=True, on_select="rerun"
                    )

                    # Handle plot selection event FOR THIS SECTION
                    select_info = event_data.get('select') if event_data else None
                    points_data = select_info.get('points', []) if select_info else None
                    # Check if event data is new *for this section*
                    is_new_event_sec = (points_data is not None and points_data != section['last_select_event'])

                    if is_new_event_sec:
                        logging.info(f"Plot select event (Sect {section_id}): {len(points_data)} pts.")
                        section['last_select_event'] = points_data # Store event data
                        if len(points_data) >= 1:
                            selected_point = points_data[0]
                            x_sel, y_sel = selected_point.get('x'), selected_point.get('y')
                            if x_sel is not None and y_sel is not None:
                                target = section['selection_target']
                                mi_sec = section['manual_input'] # Ref to section's manual input
                                if target == 'P1':
                                    mi_sec['x1'] = x_sel; mi_sec['y1'] = y_sel
                                    section['selection_target'] = 'P2'
                                elif target == 'P2':
                                    mi_sec['x2'] = x_sel; mi_sec['y2'] = y_sel
                                    section['selection_target'] = 'P1'
                                # Rerun needed to update the number inputs in col3
                                st.rerun()
                else:
                    st.warning("Plot could not be generated.")
            else:
                st.caption("Configure axes and smoothing after uploading data.")

        # --- Column 3: Segment Input & Control ---
        with col3:
            st.subheader("3. Segments")
            if section['prepared_df'] is not None: # Only show if data loaded
                target_sec = section['selection_target']
                st.info(f"Next plot click updates: **{target_sec}**")
                mi_sec = section['manual_input']
                mi_sec['x1'] = st.number_input(f"P1 X:", value=mi_sec['x1'], format="%.3f", key=f"num_x1_{section_id}") # UNIQUE KEY
                mi_sec['y1'] = st.number_input(f"P1 Y:", value=mi_sec['y1'], format="%.3f", key=f"num_y1_{section_id}") # UNIQUE KEY
                mi_sec['x2'] = st.number_input(f"P2 X:", value=mi_sec['x2'], format="%.3f", key=f"num_x2_{section_id}") # UNIQUE KEY
                mi_sec['y2'] = st.number_input(f"P2 Y:", value=mi_sec['y2'], format="%.3f", key=f"num_y2_{section_id}") # UNIQUE KEY

                add_button_sec = st.button("Add Segment", key=f"add_{section_id}", use_container_width=True) # UNIQUE KEY
                clear_button_sec = st.button("Clear Inputs", key=f"clear_{section_id}", use_container_width=True) # UNIQUE KEY
                reset_button_sec = st.button("Reset All Segments", key=f"reset_{section_id}", use_container_width=True) # UNIQUE KEY

                if add_button_sec:
                    x1, y1, x2, y2 = mi_sec['x1'], mi_sec['y1'], mi_sec['x2'], mi_sec['y2']
                    if None in [x1, y1, x2, y2]: st.error("All coordinates needed.")
                    else:
                        p1 = (x1, y1); p2 = (x2, y2)
                        if abs(p1[0] - p2[0]) > 1e-6 or abs(p1[1] - p2[1]) > 1e-6:
                            slope = dp.calculate_slope(p1, p2)
                            section['segments'].append({'start': p1, 'end': p2, 'slope': slope})
                            logging.info(f"Added seg {len(section['segments'])} (Sect {section_id})")
                            # Clear inputs and reset target for this section
                            section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                            section['selection_target'] = 'P1'
                            section['plot_fig'] = None # Force redraw
                            st.rerun()
                        else: st.error("P1/P2 too close.")

                if clear_button_sec:
                    section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                    section['selection_target'] = 'P1'
                    st.rerun()

                if reset_button_sec:
                    section['segments'] = []
                    section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                    section['selection_target'] = 'P1'
                    section['plot_fig'] = None # Force redraw
                    logging.info(f"Reset segments (Sect {section_id})")
                    st.rerun()

                # Display Segment Table for this section
                st.markdown("---")
                st.markdown("**Defined Segments**")
                segments_sec = section['segments']
                if segments_sec:
                    data_disp_sec = []
                    for i_s, seg_s in enumerate(segments_sec):
                        try: data_disp_sec.append({"Seg #": i_s + 1, "Start X": f"{seg_s['start'][0]:.2f}", "Start Y": f"{seg_s['start'][1]:.2f}", "End X": f"{seg_s['end'][0]:.2f}", "End Y": f"{seg_s['end'][1]:.2f}", "Slope (m)": f"{seg_s['slope']:.4f}"})
                        except: data_disp_sec.append({"Seg #": i_s + 1, "Start X": "Err", "Start Y": "Err", "End X": "Err", "End Y": "Err", "Slope (m)": "Err"})
                    df_segs_sec = pd.DataFrame(data_disp_sec).set_index('Seg #')
                    st.dataframe(df_segs_sec, use_container_width=True)
                else: st.caption("No segments defined.")
            else:
                st.caption("Load data to define segments.")

        # --- Section Footer ---
        st.markdown("---")
        # Use args in on_click to pass the section ID
        st.button("❌ Remove Section", key=f"remove_{section_id}", on_click=remove_section, args=(section_id,), type="secondary")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"*{APP_TITLE}*")
st.sidebar.info(f"📍 Sassari, Sardinia, Italy")
now_local = datetime.now()
timezone_hint = "CEST"
st.sidebar.caption(f"Timestamp: {now_local.strftime('%Y-%m-%d %H:%M:%S')} {timezone_hint}")