# streamlit_app.py (Multi-Section Version - v1.0 Cleaned - Click-to-Display)

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
    # Ensure data_processing.py (Caching Disabled - Sec MA Index Approach Refined version) is present
    import data_processing as dp
except ImportError:
    st.error("Fatal Error: `data_processing.py` not found. Please ensure it's in the same directory.")
    st.stop()

# --- Configuration ---
APP_TITLE = "Multi-Analysis Segmenter (v1.0)"
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon="📊" # Set page icon
)


# --- Helper Function for Plotting (Unchanged) ---
def create_plot(df_display, x_col, y_col, smoothing_method, metadata, segments_list, section_id):
    """Creates the Plotly figure for display, adding section ID to title."""
    fig = go.Figure()
    dataset_name = metadata.get('display_name', f"Data {section_id}")
    plot_title = f"Section {section_id}: {y_col} vs {x_col} ({smoothing_method})<br>{dataset_name}"

    plot_error = False; err_msg = ""
    if df_display is None or df_display.empty: plot_error = True; err_msg = "No data."
    elif not x_col or not y_col: plot_error = True; err_msg = "X/Y axis not selected."
    elif x_col not in df_display.columns: plot_error = True; err_msg = f"X-col '{x_col}' missing."
    elif y_col not in df_display.columns: plot_error = True; err_msg = f"Y-col '{y_col}' missing."
    elif df_display[x_col].isnull().all(): plot_error = True; err_msg = f"All X ('{x_col}') data NaN."
    elif df_display[y_col].isnull().all(): plot_error = True; err_msg = f"All Y ('{y_col}') data NaN."

    if plot_error:
        logger.warning(f"Plot Error (Section {section_id}) - {err_msg}")
        fig.add_annotation(text=f"Cannot plot: {err_msg}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f"Section {section_id}: Plot Error", xaxis_title=x_col or "X", yaxis_title=y_col or "Y")
        return fig

    try:
        fig.add_trace(go.Scattergl(
            x=df_display[x_col], y=df_display[y_col], mode='markers',
            marker=dict(color='blue', size=5, opacity=0.7), name=f'Data',
            hovertemplate=f"<b>X ({x_col})</b>: %{{x:.2f}}<br><b>Y ({y_col})</b>: %{{y:.2f}}<extra></extra>"
        ))
    except Exception as e:
         logger.error(f"Error adding scatter trace (Sect {section_id}): {e}", exc_info=True)
         fig.add_annotation(text=f"Plot Error:\n{e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

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

    fig.update_layout(title=plot_title, xaxis_title=x_col, yaxis_title=y_col, hovermode='closest', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), dragmode='zoom')
    return fig

# --- Streamlit App Initialization ---
st.title(APP_TITLE)
st.info("""
**Instructions:**
1. Click "Add Analysis Section" to create a new independent analysis area.
2. In each section:
    a. Upload **one** `.xlsx` or `.csv` file.
    b. Configure Smoothing, X-Axis, and Y-Axis. The plot updates automatically.
    c. **Click a point** on the plot to see its coordinates displayed below.
    d. Manually type the desired coordinates into the P1 and P2 fields.
    e. Click "Add Segment" within the section.
    f. Use drag-to-zoom on the plot. Use mode bar for pan/other options.
    g. Use "Clear Inputs" or "Reset All Segments".
    h. Click "Remove Section" to delete an analysis area.
""") # Updated instructions

# --- Session State Initialization ---
if 'analysis_sections' not in st.session_state:
    st.session_state.analysis_sections = []

def add_new_section():
    section_id = f"S{int(time.time()*100)}_{len(st.session_state.analysis_sections)}"
    st.session_state.analysis_sections.append({
        'id': section_id, 'file_info': None, 'prepared_df': None, 'metadata': {},
        'plot_config': {'smooth': "Raw Data", 'x_col': None, 'y_col': None},
        'df_display': None, 'segments': [],
        'manual_input': {'x1': None, 'y1': None, 'x2': None, 'y2': None},
        'last_clicked_point': None, # Store {x: val, y: val} or None
        'plot_fig': None
        # Removed: selection_target, last_select_event
    })
    logger.info(f"Added new section with ID: {section_id}")

def remove_section(section_id_to_remove):
    initial_len = len(st.session_state.analysis_sections)
    st.session_state.analysis_sections = [sec for sec in st.session_state.analysis_sections if sec['id'] != section_id_to_remove]
    if len(st.session_state.analysis_sections) < initial_len: logger.info(f"Removed section {section_id_to_remove}")
    else: logger.warning(f"Attempted remove section {section_id_to_remove}, not found.")
    st.rerun()

# --- Control Button to Add Sections ---
st.button("➕ Add Analysis Section", on_click=add_new_section, key="add_section_btn")

# --- Render Each Analysis Section ---
if not st.session_state.analysis_sections:
    st.warning("Click 'Add Analysis Section' to begin.")

for section in st.session_state.analysis_sections:
    section_id = section['id']

    # Construct expander title safely
    metadata = section.get('metadata', {}); config = section.get('plot_config', {})
    name = metadata.get('name', ''); surname = metadata.get('surname', 'UnknownSubject')
    y_var = config.get('y_col', 'Y'); x_var = config.get('x_col', 'X'); smooth_type = config.get('smooth', 'Raw Data')
    subject_name = f"{name} {surname}".strip()
    if not subject_name: subject_name = metadata.get('display_name', f"Section {section_id}")
    expander_title = f"{subject_name}: {y_var} vs {x_var} ({smooth_type})"

    with st.expander(expander_title, expanded=True):
        col1, col2, col3 = st.columns([1, 3, 1.5]) # Config | Plot | Input

        # --- Column 1: File Upload & Config ---
        with col1:
            st.subheader("1. Load & Configure")
            uploaded_file = st.file_uploader(f"Upload Data File ({section_id})", type=["xlsx", "xls", "csv"], key=f"uploader_{section_id}", accept_multiple_files=False)
            if uploaded_file is not None:
                current_file_name = None; file_info_current = section.get('file_info')
                if isinstance(file_info_current, dict): current_file_name = file_info_current.get('name')
                if current_file_name != uploaded_file.name:
                    logger.info(f"Processing new file '{uploaded_file.name}' for section {section_id}")
                    # Reset section state
                    section['prepared_df']=None; section['metadata']={}; section['plot_config']={'smooth': "Raw Data",'x_col':None,'y_col':None}; section['df_display']=None; section['segments']=[]; section['manual_input']={'x1':None,'y1':None,'x2':None,'y2':None}; section['last_clicked_point']=None; section['plot_fig']=None # Reset new state too
                    try:
                        filename = uploaded_file.name; metadata_parsed, unique_key = dp.parse_filename(filename)
                        if unique_key is None: unique_key = filename
                        file_content_buffer = io.BytesIO(uploaded_file.getvalue()); raw_df = dp.load_data(file_content_buffer, filename)
                        if raw_df is not None:
                            prepared_df = dp.prepare_data(raw_df)
                            if prepared_df is not None and not prepared_df.empty:
                                section['file_info']={'name':filename,'unique_key':unique_key}; section['prepared_df']=prepared_df; section['metadata']=metadata_parsed or {'filename':filename,'display_name':filename}
                                st.success(f"Loaded: {filename}"); st.rerun()
                            else: st.error(f"Data prep failed: {filename}."); section['prepared_df'] = None
                        else: st.error(f"File load failed: {filename}."); section['prepared_df'] = None
                    except Exception as e: st.error(f"Error processing {uploaded_file.name}: {e}"); logger.error(f"File processing error (Sect {section_id}): {e}", exc_info=True); section['prepared_df'] = None; section['file_info'] = {'name': uploaded_file.name, 'unique_key': uploaded_file.name}

            if section.get('prepared_df') is not None:
                st.markdown("---"); df_prepared_sec = section['prepared_df']; config = section['plot_config']; needs_config_rerun = False
                smoothing_options = ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"]
                current_smooth = config['smooth']; smooth_idx = smoothing_options.index(current_smooth) if current_smooth in smoothing_options else 0
                new_smooth = st.selectbox(f"Smoothing ({section_id}):", options=smoothing_options, index=smooth_idx, key=f"smooth_{section_id}")
                if new_smooth != config['smooth']: config['smooth'] = new_smooth; section['plot_fig'] = None; needs_config_rerun = True

                numeric_cols = [];
                try:
                    df_display_for_cols = section.get('df_display')
                    if df_display_for_cols is None or needs_config_rerun or section.get('plot_fig') is None: # Update df_display if needed
                         df_display_for_cols = dp.apply_smoothing(df_prepared_sec, config['smooth'], dp.TIME_COL_SECONDS)
                         section['df_display'] = df_display_for_cols
                    if df_display_for_cols is not None and not df_display_for_cols.empty: numeric_cols = df_display_for_cols.select_dtypes(include=np.number).columns.tolist()
                    if not numeric_cols: logger.warning(f"No numeric cols (Sect {section_id})."); numeric_cols = df_prepared_sec.select_dtypes(include=np.number).columns.tolist()
                except Exception as e_smooth_cols: st.error(f"Smooth error: {e_smooth_cols}"); logger.error(f"Col select smooth fail (Sect {section_id})", exc_info=True); numeric_cols = df_prepared_sec.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols: st.error("No numeric columns."); continue

                current_x = config.get('x_col'); default_x = current_x if current_x in numeric_cols else (dp.TIME_COL_SECONDS if dp.TIME_COL_SECONDS in numeric_cols else numeric_cols[0])
                try: x_idx = numeric_cols.index(default_x)
                except ValueError: x_idx = 0
                new_x = st.selectbox(f"X-Axis ({section_id}):", options=numeric_cols, index=x_idx, key=f"x_{section_id}")
                if new_x != config['x_col']: config['x_col'] = new_x; config['y_col'] = None; section['plot_fig'] = None; needs_config_rerun = True

                y_options = [c for c in numeric_cols if c != config['x_col']];
                if not y_options: st.error("No compatible Y-axis."); continue
                current_y = config.get('y_col'); default_y = None
                if current_y in y_options: default_y = current_y
                else: common_y=['V\'O2/kg','V\'O2','V\'CO2','V\'E',dp.WATT_COL,'FC']; [default_y := yc for yc in y_options if yc in common_y]; default_y = default_y if default_y else y_options[0]
                try: y_idx = y_options.index(default_y)
                except ValueError: y_idx = 0
                new_y = st.selectbox(f"Y-Axis ({section_id}):", options=y_options, index=y_idx, key=f"y_{section_id}")
                if new_y != config['y_col']: config['y_col'] = new_y; section['plot_fig'] = None; needs_config_rerun = True

                if needs_config_rerun:
                    if section.get('plot_fig') is None:
                        try: section['df_display'] = dp.apply_smoothing(df_prepared_sec, config['smooth'], dp.TIME_COL_SECONDS)
                        except Exception as e_smooth_rerun: st.error(f"Smooth error: {e_smooth_rerun}"); section['df_display'] = None
                    st.rerun()
            else: st.caption("Upload file to configure plot.")

        # --- Column 2: Plot ---
        with col2:
            st.subheader("2. Plot")
            df_display_sec = section.get('df_display'); config_sec = section.get('plot_config', {}); x_col_sec = config_sec.get('x_col'); y_col_sec = config_sec.get('y_col'); segments_sec = section.get('segments', []); metadata_sec = section.get('metadata', {})
            if df_display_sec is not None and x_col_sec and y_col_sec:
                if section.get('plot_fig') is None:
                     logger.info(f"Regenerating plot for section {section_id}")
                     section['plot_fig'] = create_plot(df_display_sec, x_col_sec, y_col_sec, config_sec.get('smooth','Raw Data'), metadata_sec, segments_sec, section_id)
                fig_to_display = section.get('plot_fig')
                if fig_to_display:
                    plot_chart_key = f"chart_{section_id}"
                    st.plotly_chart(fig_to_display, key=plot_chart_key, use_container_width=True, on_click="rerun") # Keep on_click

                    # --- Handle Plot CLICK Event FOR THIS SECTION (Simplified) ---
                    click_event_data = st.session_state.get(plot_chart_key) # Get click data from state

                    if click_event_data: # If a click happened
                        logger.info(f"Plot click event detected (Sect {section_id}): {click_event_data}")
                        clicked_point_data = None
                        if isinstance(click_event_data, list) and len(click_event_data) > 0: clicked_point_data = click_event_data[0]
                        elif isinstance(click_event_data, dict):
                            if 'points' in click_event_data and isinstance(click_event_data['points'], list) and len(click_event_data['points']) > 0: clicked_point_data = click_event_data['points'][0]
                            else: clicked_point_data = click_event_data # Assume dict is the point

                        if isinstance(clicked_point_data, dict):
                            x_sel = clicked_point_data.get('x'); y_sel = clicked_point_data.get('y')
                            if x_sel is not None and y_sel is not None:
                                # <<< CHANGE: Store clicked coords, don't update inputs >>>
                                section['last_clicked_point'] = {'x': x_sel, 'y': y_sel}
                                logger.debug(f"Stored last click (Sect {section_id}): X={x_sel:.2f}, Y={y_sel:.2f}")
                            else:
                                logger.warning(f"Click point missing coords (Sect {section_id}).")
                                section['last_clicked_point'] = None # Clear if data invalid
                        else:
                            logger.warning(f"Could not extract point data dict from click (Sect {section_id})")
                            section['last_clicked_point'] = None # Clear if structure invalid

                        # Clear the event from state immediately after processing
                        st.session_state[plot_chart_key] = None
                        st.rerun() # Rerun needed to display the last_clicked_point value
                else: st.warning("Plot could not be generated.")
            else: st.caption("Configure axes and smoothing after upload.")

        # --- Column 3: Segment Input & Control ---
        with col3:
            st.subheader("3. Segments")
            if section.get('prepared_df') is not None:
                # <<< CHANGE: Display last clicked point >>>
                last_click = section.get('last_clicked_point')
                if last_click:
                    st.caption(f"**Last Click:** `X = {last_click['x']:.3f}`, `Y = {last_click['y']:.3f}`")
                else:
                    st.caption("Click a point on the plot to see coordinates here.")
                st.markdown("---") # Separator
                # <<< END CHANGE >>>

                mi_sec = section.get('manual_input', {}); # Get manual input dict
                # Number inputs for manual entry (still linked to state)
                mi_sec['x1'] = st.number_input(f"P1 X ({section_id}):", value=mi_sec.get('x1'), format="%.3f", key=f"num_x1_{section_id}")
                mi_sec['y1'] = st.number_input(f"P1 Y ({section_id}):", value=mi_sec.get('y1'), format="%.3f", key=f"num_y1_{section_id}")
                mi_sec['x2'] = st.number_input(f"P2 X ({section_id}):", value=mi_sec.get('x2'), format="%.3f", key=f"num_x2_{section_id}")
                mi_sec['y2'] = st.number_input(f"P2 Y ({section_id}):", value=mi_sec.get('y2'), format="%.3f", key=f"num_y2_{section_id}")

                # Buttons remain the same, operating on manual_input state
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
                                slope = dp.calculate_slope(p1, p2); section['segments'].append({'start': p1, 'end': p2, 'slope': slope})
                                logger.info(f"Added seg {len(section['segments'])} (Sect {section_id})");
                                # Clear inputs after adding
                                section['manual_input'] = {'x1':None,'y1':None,'x2':None,'y2':None};
                                section['plot_fig'] = None; # Force redraw to show segment
                                st.rerun()
                            except Exception as e_slope: st.error(f"Slope calc error: {e_slope}"); logger.error(f"Slope error (Sect {section_id})", exc_info=True)
                        else: st.error("P1/P2 too close.")
                if clear_button_sec:
                    section['manual_input'] = {'x1':None,'y1':None,'x2':None,'y2':None};
                    st.rerun() # Rerun to clear fields
                if reset_button_sec:
                    section['segments'] = []; section['manual_input'] = {'x1':None,'y1':None,'x2':None,'y2':None}; section['last_clicked_point']=None; section['plot_fig']=None;
                    logger.info(f"Reset segments (Sect {section_id})");
                    st.rerun() # Rerun to clear fields and plot

                st.markdown("---"); st.markdown("**Defined Segments**")
                segments_sec = section.get('segments', [])
                if segments_sec:
                    data_disp_sec = []
                    for i_s, seg_s in enumerate(segments_sec):
                        if isinstance(seg_s, dict) and all(k in seg_s for k in ['start','end','slope']):
                             try: data_disp_sec.append({"Seg #": i_s + 1, "Start X": f"{seg_s['start'][0]:.2f}", "Start Y": f"{seg_s['start'][1]:.2f}", "End X": f"{seg_s['end'][0]:.2f}", "End Y": f"{seg_s['end'][1]:.2f}", "Slope (m)": f"{seg_s['slope']:.4f}"})
                             except Exception as e_fmt: logger.warning(f"Err fmt seg {i_s+1} (Sect {section_id}): {e_fmt}"); data_disp_sec.append({"Seg #":i_s+1, "Start X":"Err", "Start Y":"Err", "End X":"Err", "End Y":"Err", "Slope (m)":"Err"})
                        else: logger.warning(f"Skipping invalid seg struct (Sect {section_id})")
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