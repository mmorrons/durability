# streamlit_app.py (Multi-Section Version - Full Updated Code v4 - Zoom Fix)

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
    st.error("Fatal Error: `data_processing.py` not found. Please ensure it's in the same directory as this script.")
    st.stop()

# --- Configuration ---
APP_TITLE = "Multi-Analysis Segmenter (v4)"
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function for Plotting (Updated dragmode) ---
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
    elif df_display[x_col].isnull().all(): plot_error = True; err_msg = f"All X ('{x_col}') data is invalid/NaN."
    elif df_display[y_col].isnull().all(): plot_error = True; err_msg = f"All Y ('{y_col}') data is invalid/NaN."

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
            hovertemplate=f"<b>X ({x_col})</b>: %{{x:.2f}}<br><b>Y ({y_col})</b>: %{{y:.2f}}<extra></extra>"
        ))
    except Exception as e:
         logger.error(f"Error adding scatter trace (Sect {section_id}): {e}", exc_info=True)
         fig.add_annotation(text=f"Plot Error:\n{e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Add Segments
    if isinstance(segments_list, list):
        for i, segment in enumerate(segments_list):
             if isinstance(segment, dict) and all(k in segment for k in ['start', 'end', 'slope']):
                 try:
                     p_s, p_e, m = segment['start'], segment['end'], segment['slope']
                     if isinstance(p_s,(list,tuple)) and len(p_s)==2 and all(isinstance(n,(int,float)) for n in p_s) and \
                        isinstance(p_e,(list,tuple)) and len(p_e)==2 and all(isinstance(n,(int,float)) for n in p_e):
                         fig.add_trace(go.Scatter(x=[p_s[0], p_e[0]], y=[p_s[1], p_e[1]], mode='lines+markers', line=dict(color='red', width=2), marker=dict(color='red', size=8), name=f'Seg{i+1}(m={m:.2f})'))
                         mid_x, mid_y = (p_s[0] + p_e[0]) / 2, (p_s[1] + p_e[1]) / 2
                         fig.add_annotation(x=mid_x, y=mid_y, text=f' m={m:.2f}', showarrow=False, font=dict(color='red', size=10), xshift=5)
                     else: logger.warning(f"Invalid point data in segment {i+1} (Sect {section_id})")
                 except Exception as e_seg: logger.warning(f"Could not plot segment {i+1} (Sect {section_id}): {e_seg}", exc_info=True)
             else: logger.warning(f"Invalid segment structure index {i} (Sect {section_id})")

    # <<< CHANGE: Updated dragmode from 'select' to 'zoom' >>>
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_col, yaxis_title=y_col,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode='zoom' # Set default drag to zoom
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
    c. **Define segments by clicking points** on the plot (fills P1/P2 inputs) or typing coordinates manually.
    d. Click "Add Segment" within the section.
    e. Use **drag-to-zoom** on the plot area or axes. Use mode bar for pan/other zoom options.
    f. Use "Clear Inputs" or "Reset All Segments" within the section as needed.
    g. Click "Remove Section" to delete an analysis area.
""") # Added note about zooming

# --- Session State Initialization for Multi-Section ---
if 'analysis_sections' not in st.session_state:
    st.session_state.analysis_sections = [] # List to hold each section's state

def add_new_section():
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
    # Rerun needed to update UI after removal
    st.rerun()


# --- Control Button to Add Sections ---
st.button("➕ Add Analysis Section", on_click=add_new_section, key="add_section_btn")

# --- Render Each Analysis Section ---
if not st.session_state.analysis_sections:
    st.warning("Click 'Add Analysis Section' to begin.")

# Iterate directly over the list. Streamlit's execution model handles changes.
for section in st.session_state.analysis_sections:
    section_id = section['id']

    # Safely get the file name for the title
    file_info = section.get('file_info')
    file_name_for_title = "None"
    if isinstance(file_info, dict): file_name_for_title = file_info.get('name', "Unknown")

    expander_title = f"Analysis Section {section_id} (File: {file_name_for_title})"
    with st.expander(expander_title, expanded=True):

        col1, col2, col3 = st.columns([1, 3, 1.5]) # Config | Plot | Input

        # --- Column 1: File Upload & Config ---
        with col1:
            st.subheader("1. Load & Configure")
            uploaded_file = st.file_uploader(f"Upload Data File ({section_id})", type=["xlsx", "xls", "csv"], key=f"uploader_{section_id}", accept_multiple_files=False)

            if uploaded_file is not None:
                # Safely get current file name before comparing
                current_file_name = None
                file_info_current = section.get('file_info')
                if isinstance(file_info_current, dict): current_file_name = file_info_current.get('name')

                if current_file_name != uploaded_file.name:
                    logger.info(f"Processing new file '{uploaded_file.name}' for section {section_id}")
                    # Reset section state
                    section['prepared_df'] = None; section['metadata'] = {}; section['plot_config'] = {'smooth': "Raw Data", 'x_col': None, 'y_col': None}; section['df_display'] = None; section['segments'] = []; section['manual_input'] = {'x1': None, 'y1': None, 'x2': None, 'y2': None}; section['selection_target'] = 'P1'; section['last_select_event'] = None; section['plot_fig'] = None
                    try:
                        filename = uploaded_file.name
                        metadata, unique_key = dp.parse_filename(filename)
                        if unique_key is None: unique_key = filename
                        file_content_buffer = io.BytesIO(uploaded_file.getvalue())
                        raw_df = dp.load_data(file_content_buffer, filename)
                        if raw_df is not None:
                            prepared_df = dp.prepare_data(raw_df)
                            if prepared_df is not None and not prepared_df.empty:
                                section['file_info'] = {'name': filename, 'unique_key': unique_key}
                                section['prepared_df'] = prepared_df
                                section['metadata'] = metadata or {'filename': filename, 'display_name': filename}
                                st.success(f"Loaded: {filename}")
                                st.rerun() # Rerun needed
                            else: st.error(f"Data prep failed: {filename}."); section['prepared_df'] = None
                        else: st.error(f"File load failed: {filename}."); section['prepared_df'] = None
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        logger.error(f"File processing error (Sect {section_id}): {e}", exc_info=True)
                        section['prepared_df'] = None
                        section['file_info'] = {'name': uploaded_file.name, 'unique_key': uploaded_file.name}

            # B. Configuration
            if section.get('prepared_df') is not None:
                st.markdown("---")
                df_prepared_sec = section['prepared_df']
                config = section['plot_config']
                needs_plot_update = False

                # Smoothing
                smoothing_options = ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"]
                current_smooth = config['smooth']
                smooth_idx = smoothing_options.index(current_smooth) if current_smooth in smoothing_options else 0
                new_smooth = st.selectbox(f"Smoothing ({section_id}):", options=smoothing_options, index=smooth_idx, key=f"smooth_{section_id}")
                if new_smooth != config['smooth']: config['smooth'] = new_smooth; needs_plot_update = True

                # Get numeric cols
                temp_smoothed_df = None; numeric_cols = []
                try:
                    # Use selected smooth method for column selection
                    temp_smoothed_df = dp.apply_smoothing(df_prepared_sec, config['smooth'], dp.TIME_COL_SECONDS)
                    if temp_smoothed_df is not None and not temp_smoothed_df.empty: numeric_cols = temp_smoothed_df.select_dtypes(include=np.number).columns.tolist()
                    if not numeric_cols: logger.warning(f"No numeric cols after smooth (Sect {section_id})."); numeric_cols = df_prepared_sec.select_dtypes(include=np.number).columns.tolist()
                except Exception as e_smooth_cols: st.error(f"Smooth error: {e_smooth_cols}"); logger.error(f"Temp smooth failed (Sect {section_id})", exc_info=True); numeric_cols = df_prepared_sec.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols: st.error("No numeric columns found."); continue

                # X-Axis
                current_x = config['x_col']
                default_x = current_x if current_x in numeric_cols else (dp.TIME_COL_SECONDS if dp.TIME_COL_SECONDS in numeric_cols else numeric_cols[0])
                try: x_idx = numeric_cols.index(default_x)
                except ValueError: x_idx = 0
                new_x = st.selectbox(f"X-Axis ({section_id}):", options=numeric_cols, index=x_idx, key=f"x_{section_id}")
                if new_x != config['x_col']: config['x_col'] = new_x; config['y_col'] = None; needs_plot_update = True

                # Y-Axis
                y_options = [c for c in numeric_cols if c != config['x_col']];
                if not y_options: st.error("No compatible Y-axis."); continue
                current_y = config['y_col']
                default_y = None
                if current_y in y_options: default_y = current_y
                else: common_y = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', dp.WATT_COL, 'FC']; [default_y := yc for yc in y_options if yc in common_y]; default_y = default_y if default_y else y_options[0]
                try: y_idx = y_options.index(default_y)
                except ValueError: y_idx = 0
                new_y = st.selectbox(f"Y-Axis ({section_id}):", options=y_options, index=y_idx, key=f"y_{section_id}")
                if new_y != config['y_col']: config['y_col'] = new_y; needs_plot_update = True

                # Update smoothed data if needed
                # Also update if df_display is missing, or if relevant config changed
                if needs_plot_update or section.get('df_display') is None:
                     logger.info(f"Updating smoothed data for plot (Sect {section_id}) - Reason: needs_update={needs_plot_update}, df_display_missing={section.get('df_display') is None}")
                     try:
                         # Always re-calculate smoothing when config changes
                         section['df_display'] = dp.apply_smoothing(df_prepared_sec, config['smooth'], dp.TIME_COL_SECONDS)
                         section['plot_fig'] = None # Force plot regeneration
                     except Exception as e_final_smooth: st.error(f"Smooth fail: {e_final_smooth}"); logger.error(f"Final smooth fail (Sect {section_id})", exc_info=True); section['df_display'] = None
            else: st.caption("Upload a file to configure plot.")

        # --- Column 2: Plot ---
        with col2:
            st.subheader("2. Plot")
            df_display_sec = section.get('df_display')
            config_sec = section.get('plot_config', {})
            x_col_sec = config_sec.get('x_col')
            y_col_sec = config_sec.get('y_col')

            if df_display_sec is not None and x_col_sec and y_col_sec:
                if section.get('plot_fig') is None: # Regenerate if needed
                     logger.info(f"Regenerating plot for section {section_id}")
                     section['plot_fig'] = create_plot(df_display_sec, x_col_sec, y_col_sec, config_sec['smooth'], section['metadata'], section['segments'], section_id)

                fig_to_display = section.get('plot_fig')
                if fig_to_display:
                    plot_chart_key = f"chart_{section_id}"
                    # Ensure on_select is set to trigger reruns for click interactions
                    event_data = st.plotly_chart(fig_to_display, key=plot_chart_key, use_container_width=True, on_select="rerun")

                    # Handle plot selection event FOR THIS SECTION
                    select_info = event_data.get('select') if event_data else None
                    points_data = select_info.get('points', []) if select_info else None
                    is_new_event_sec = (points_data is not None and points_data != section.get('last_select_event'))

                    if is_new_event_sec:
                        section['last_select_event'] = points_data # Store event data
                        # Process clicks even if dragmode is 'zoom'
                        if len(points_data) >= 1:
                            logger.info(f"Plot click/select event (Sect {section_id}): {len(points_data)} pts.")
                            selected_point = points_data[0] # Use first point of selection
                            x_sel, y_sel = selected_point.get('x'), selected_point.get('y')
                            if x_sel is not None and y_sel is not None:
                                target = section['selection_target']; mi_sec = section['manual_input']
                                if target == 'P1': mi_sec['x1'] = x_sel; mi_sec['y1'] = y_sel; section['selection_target'] = 'P2'
                                elif target == 'P2': mi_sec['x2'] = x_sel; mi_sec['y2'] = y_sel; section['selection_target'] = 'P1'
                                logger.debug(f"Updated Manual {target} from plot click (Sect {section_id})")
                                st.rerun() # Rerun needed to update number inputs
                            else: logger.warning(f"Plot click point missing coords (Sect {section_id}).")
                        # else: Selection cleared or no points, ignore
                else: st.warning("Plot could not be generated.")
            else: st.caption("Configure axes and smoothing after upload.")

        # --- Column 3: Segment Input & Control ---
        with col3:
            st.subheader("3. Segments")
            if section.get('prepared_df') is not None:
                target_sec = section.get('selection_target', 'P1')
                st.info(f"Next plot click updates: **{target_sec}**")
                mi_sec = section.get('manual_input', {})
                mi_sec['x1'] = st.number_input(f"P1 X ({section_id}):", value=mi_sec.get('x1'), format="%.3f", key=f"num_x1_{section_id}")
                mi_sec['y1'] = st.number_input(f"P1 Y ({section_id}):", value=mi_sec.get('y1'), format="%.3f", key=f"num_y1_{section_id}")
                mi_sec['x2'] = st.number_input(f"P2 X ({section_id}):", value=mi_sec.get('x2'), format="%.3f", key=f"num_x2_{section_id}")
                mi_sec['y2'] = st.number_input(f"P2 Y ({section_id}):", value=mi_sec.get('y2'), format="%.3f", key=f"num_y2_{section_id}")
                add_button_sec = st.button("Add Segment", key=f"add_{section_id}", use_container_width=True)
                clear_button_sec = st.button("Clear Inputs", key=f"clear_{section_id}", use_container_width=True)
                reset_button_sec = st.button("Reset All Segments", key=f"reset_{section_id}", use_container_width=True)

                if add_button_sec:
                    x1,y1,x2,y2 = mi_sec.get('x1'),mi_sec.get('y1'),mi_sec.get('x2'),mi_sec.get('y2')
                    if None in [x1,y1,x2,y2]: st.error("All coordinates needed.")
                    else:
                        p1 = (x1,y1); p2 = (x2,y2)
                        if abs(p1[0] - p2[0]) > 1e-6 or abs(p1[1] - p2[1]) > 1e-6:
                            try:
                                slope = dp.calculate_slope(p1, p2)
                                section['segments'].append({'start': p1, 'end': p2, 'slope': slope})
                                logger.info(f"Added seg {len(section['segments'])} (Sect {section_id})")
                                section['manual_input'] = {'x1':None,'y1':None,'x2':None,'y2':None}; section['selection_target'] = 'P1'
                                section['plot_fig'] = None # Force redraw
                                st.rerun()
                            except Exception as e_slope: st.error(f"Slope calc error: {e_slope}"); logger.error(f"Slope error (Sect {section_id})", exc_info=True)
                        else: st.error("P1/P2 too close.")
                if clear_button_sec: section['manual_input'] = {'x1':None,'y1':None,'x2':None,'y2':None}; section['selection_target']='P1'; st.rerun()
                if reset_button_sec: section['segments'] = []; section['manual_input'] = {'x1':None,'y1':None,'x2':None,'y2':None}; section['selection_target']='P1'; section['plot_fig']=None; logger.info(f"Reset segments (Sect {section_id})"); st.rerun()

                st.markdown("---"); st.markdown("**Defined Segments**")
                segments_sec = section.get('segments', [])
                if segments_sec:
                    data_disp_sec = []
                    for i_s, seg_s in enumerate(segments_sec):
                        if isinstance(seg_s, dict) and all(k in seg_s for k in ['start','end','slope']):
                             try: data_disp_sec.append({"Seg #": i_s + 1, "Start X": f"{seg_s['start'][0]:.2f}", "Start Y": f"{seg_s['start'][1]:.2f}", "End X": f"{seg_s['end'][0]:.2f}", "End Y": f"{seg_s['end'][1]:.2f}", "Slope (m)": f"{seg_s['slope']:.4f}"})
                             except Exception as e_fmt: logger.warning(f"Err fmt seg {i_s+1} (Sect {section_id}): {e_fmt}"); data_disp_sec.append({"Seg #":i_s+1, "Start X":"Err", "Start Y":"Err", "End X":"Err", "End Y":"Err", "Slope (m)":"Err"})
                        else: logger.warning(f"Skipping invalid seg struct in table (Sect {section_id}): {seg_s}")
                    if data_disp_sec:
                        try: df_segs_sec = pd.DataFrame(data_disp_sec).set_index('Seg #'); st.dataframe(df_segs_sec, use_container_width=True)
                        except Exception as e_df: st.error(f"Err display segments: {e_df}"); logger.error(f"Segments table DF err (Sect {section_id})", exc_info=True)
                    else: st.caption("No valid segments.")
                else: st.caption("No segments defined.")
            else: st.caption("Load data to define segments.")

        # --- Section Footer ---
        st.markdown("---")
        st.button(f"❌ Remove Section {section_id}", key=f"remove_{section_id}", on_click=remove_section, args=(section_id,), type="secondary")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"*{APP_TITLE}*")
st.sidebar.markdown("📍 **Location:** Sassari, Sardinia, Italy")
now_local = datetime.now()
try: local_tz_name = now_local.astimezone().tzname()
except: local_tz_name = "Local Time"
st.sidebar.caption(f"Timestamp: {now_local.strftime('%Y-%m-%d %H:%M:%S')} ({local_tz_name})")