# streamlit_app.py (Modificato con Pulsante Applica)

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

# --- Helper Function for Plotting (invariata) ---
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
1. Load file(s).
2. Configure Dataset, Smoothing, and Axes (Right Panel).
3. **Click 'Apply Changes'** to update the plot with the new configuration.
4. **To add segments:**
    Type P1(X,Y) and P2(X,Y) coords directly (Middle Panel), then click 'Add Manual Segment'.
5. Segments show on plot and details below. Use Reset buttons as needed.
""")

# --- Session State Initialization ---
# Data storage
if 'processed_data' not in st.session_state: st.session_state.processed_data = {}
# Global Selections (quelli che verranno applicati)
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
# <<< NUOVO: Stato per gestire l'aggiornamento con il pulsante >>>
if 'last_figure' not in st.session_state: st.session_state.last_figure = None # Memorizza l'ultimo grafico *applicato*
if 'last_applied_config' not in st.session_state: st.session_state.last_applied_config = {} # Memorizza la config dell'ultimo grafico *applicato*
if 'files_processed_flag' not in st.session_state: st.session_state.files_processed_flag = False


# --- File Upload & Processing ---
uploaded_files = st.file_uploader("Upload Data File(s)", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="file_uploader_main")

# Rileva se sono stati caricati *nuovi* file rispetto all'ultimo controllo
# (Questo richiede una logica un po' più sofisticata per tracciare i file già visti,
# ma per ora modifichiamo la logica di base per *aggiungere* invece di *resettare*)

# Verifica se ci sono file caricati *in questa esecuzione*
if uploaded_files:
    # Flag per indicare se sono stati processati nuovi file in questo ciclo
    new_files_processed_this_run = False
    error_files_this_run = []
    files_to_process = []

    # Inizializza processed_data se non esiste
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}

    # Identifica i file che non sono già stati processati
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        metadata, unique_key = dp.parse_filename(filename)
        if unique_key is None: unique_key = filename

        # Processa solo se la chiave non è già presente o se vuoi permettere l'aggiornamento
        # Qui assumiamo di NON riprocessare file con la stessa chiave
        if unique_key not in st.session_state.processed_data:
            files_to_process.append((uploaded_file, filename, unique_key))
        # else:
        #     logging.info(f"Skipping already processed file: {filename} (Key: {unique_key})")


    if files_to_process:
        with st.spinner(f"Processing {len(files_to_process)} new file(s)..."):
            for uploaded_file, filename, unique_key in files_to_process:
                file_content_buffer = io.BytesIO(uploaded_file.getvalue())
                logging.info(f"Processing new file: {filename}")
                raw_df = dp.load_data(file_content_buffer, filename)
                if raw_df is not None:
                    prepared_df = dp.prepare_data(raw_df)
                    if prepared_df is not None and not prepared_df.empty:
                        # --- MODIFICA CHIAVE: Aggiungi al dizionario esistente ---
                        metadata_parsed, _ = dp.parse_filename(filename) # Ricalcola metadata qui se serve aggiornarlo
                        st.session_state.processed_data[unique_key] = {
                            'metadata': metadata_parsed or {'filename': filename, 'display_name': filename},
                            'prepared_df': prepared_df
                        }
                        # ---------------------------------------------------------
                        new_files_processed_this_run = True
                    else: error_files_this_run.append(filename)
                else: error_files_this_run.append(filename)

        if new_files_processed_this_run:
             st.success(f"Processed {len(files_to_process) - len(error_files_this_run)} new file(s).")
        if error_files_this_run:
             st.error(f"Failed to process: {', '.join(error_files_this_run)}")
        # Non serve più st.session_state.files_processed_flag se gestiamo così
        st.rerun() # Rerun per aggiornare la UI dopo l'aggiunta
# --- Main Area ---
if not st.session_state.processed_data:
     st.warning("Upload data file(s).")
else:
    # --- Layout ---
    plot_col, manual_col, config_col = st.columns([2, 1.5, 1]) # Adjust ratios as needed

    # --- Configuration Column ---
    with config_col:
        st.subheader("Configuration")

        # Dataset selection
        dataset_options = {k: v['metadata'].get('display_name', k) for k, v in st.session_state.processed_data.items()}
        sorted_keys = sorted(dataset_options, key=dataset_options.get); display_names = [dataset_options[key] for key in sorted_keys]
        # Leggi la chiave corrente dallo stato
        current_key_on_load = st.session_state.current_selected_key
        current_display_name = dataset_options.get(current_key_on_load, None)
        try: current_idx = display_names.index(current_display_name) if current_display_name and display_names else 0
        except ValueError: current_idx = 0

        # Usa un widget selectbox per selezionare il NOME visualizzato
        selected_display_name = st.selectbox(
            "Dataset:", options=display_names, key="dataset_selector_display", index=current_idx,
            help="Changing the dataset will reset segments and require applying changes again."
            )
        # Trova la CHIAVE corrispondente al nome selezionato
        selected_key = next((key for key, name in dataset_options.items() if name == selected_display_name), None)

        # <<< MODIFICATO: Logica di cambio dataset >>>
        # Controlla se la chiave selezionata è cambiata rispetto a quella memorizzata
        if selected_key != st.session_state.current_selected_key:
            logging.info(f"Dataset selection changed to: {selected_key}. Resetting states.")
            st.session_state.current_selected_key = selected_key
            # --- Reset states on dataset change ---
            st.session_state.segments = []
            st.session_state.manual_x1, st.session_state.manual_y1 = None, None
            st.session_state.manual_x2, st.session_state.manual_y2 = None, None
            st.session_state.selection_target = 'P1'
            st.session_state.last_select_event_data = None
            st.session_state.last_figure = None # Forza rigenerazione al prossimo Apply
            st.session_state.last_applied_config = {} # Resetta la config applicata
            st.session_state.current_x_col = None # Resetta anche gli assi e smoothing
            st.session_state.current_y_col = None
            st.session_state.current_smoothing = "Raw Data"
            # --- End Reset ---
            st.rerun() # Rerun per applicare il reset

        # Assicurati che la chiave sia valida dopo il potenziale cambio
        if not selected_key or selected_key not in st.session_state.processed_data: st.error("Dataset error or not selected."); st.stop()

        # Ottieni i dati per la chiave CORRENTE (già aggiornata se necessario)
        current_metadata = st.session_state.processed_data[selected_key]['metadata']; df_prepared = st.session_state.processed_data[selected_key]['prepared_df']

        # Smoothing selection (aggiorna lo stato direttamente)
        smoothing_options = ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"]
        current_smoothing_idx = smoothing_options.index(st.session_state.current_smoothing) if st.session_state.current_smoothing in smoothing_options else 0
        st.session_state.current_smoothing = st.selectbox(
                "Smoothing:", options=smoothing_options,
                index=current_smoothing_idx,
                key="smoothing_selector"
            )

        # Applicare smoothing TEMPORANEAMENTE per ottenere le colonne disponibili per i selettori degli assi
        # NOTA: Questo smoothing verrà riapplicato al momento della generazione del plot se si clicca "Applica"
        try:
             df_for_columns = dp.apply_smoothing(df_prepared, st.session_state.current_smoothing, dp.TIME_COL_SECONDS)
             if df_for_columns is None or df_for_columns.empty: raise ValueError("Smoothing returned no data")
             numeric_cols = df_for_columns.select_dtypes(include=np.number).columns.tolist()
             if not numeric_cols: raise ValueError("No numeric columns after smoothing")
        except Exception as e:
             st.error(f"Error during temporary smoothing for axis selection: {e}")
             numeric_cols = df_prepared.select_dtypes(include=np.number).columns.tolist() # Fallback to prepared cols
             if not numeric_cols: st.stop()

        # Axis selection (aggiorna lo stato direttamente)
        current_x = st.session_state.get('current_x_col')
        default_x = current_x if current_x in numeric_cols else (dp.TIME_COL_SECONDS if dp.TIME_COL_SECONDS in numeric_cols else numeric_cols[0])
        try: default_x_idx = numeric_cols.index(default_x)
        except ValueError: default_x_idx = 0
        st.session_state.current_x_col = st.selectbox(
                "X-Axis:", numeric_cols,
                index=default_x_idx,
                key=f"x_select"
            )

        current_y = st.session_state.get('current_y_col')
        y_options = [c for c in numeric_cols if c != st.session_state.current_x_col]
        if not y_options: st.error("Only one numeric column available for Y-axis."); st.stop()
        default_y = None
        if current_y in y_options: default_y = current_y
        else:
            common_y = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', dp.WATT_COL, 'FC'];
            for yc in y_options:
                if yc in common_y: default_y = yc; break
            if default_y is None: default_y = y_options[0]
        try: default_y_idx = y_options.index(default_y)
        except ValueError: default_y_idx = 0
        st.session_state.current_y_col = st.selectbox(
            "Y-Axis:", y_options,
            index=default_y_idx,
            key=f"y_select"
            )

        # <<< NUOVO: Pulsante Applica Modifiche >>>
        st.markdown("---") # Separatore visuale
        apply_changes_button = st.button("Applica Modifiche", key="apply_config", type="primary", use_container_width=True)


    # --- Manual Input Column (invariata) ---
    with manual_col:
        st.subheader("Manual Segment")
        st.caption("Use plot selection or type coords below.") # Modificato caption leggermente

        # Display area for selection target
        st.markdown(f"**Next Plot Selection will update:** `{st.session_state.selection_target}`")

        # Use session state values for number inputs
        st.session_state.manual_x1 = st.number_input("P1 X:", value=st.session_state.manual_x1, format="%.3f", key="disp_x1_manual")
        st.session_state.manual_y1 = st.number_input("P1 Y:", value=st.session_state.manual_y1, format="%.3f", key="disp_y1_manual")
        st.session_state.manual_x2 = st.number_input("P2 X:", value=st.session_state.manual_x2, format="%.3f", key="disp_x2_manual")
        st.session_state.manual_y2 = st.number_input("P2 Y:", value=st.session_state.manual_y2, format="%.3f", key="disp_y2_manual")

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
                    logging.info(f"Manual Seg {segment_count} added. m={slope:.4f}")
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

        # <<< NUOVO: Logica per decidere se aggiornare il grafico >>>
        # Raccogli la configurazione CORRENTE dai selettori
        x_col_selected = st.session_state.get('current_x_col')
        y_col_selected = st.session_state.get('current_y_col')
        key_selected = st.session_state.get('current_selected_key')
        smoothing_selected = st.session_state.get('current_smoothing')

        current_plot_config = {
            'key': key_selected,
            'x': x_col_selected,
            'y': y_col_selected,
            'smooth': smoothing_selected
        }

        # Controlla se la configurazione attuale è diversa dall'ultima applicata
        config_has_changed = (current_plot_config != st.session_state.get('last_applied_config'))

        fig_to_display = None
        regenerate_plot = False

        if apply_changes_button:
            logging.info("Apply Changes button clicked. Regenerating plot.")
            regenerate_plot = True
        elif st.session_state.last_figure is None and key_selected:
            logging.info("No previous plot found. Triggering initial plot generation on next Apply.")
            # Non rigenerare ora, ma informa l'utente che deve applicare
            st.info("Configure plot options and click 'Apply Changes' to generate the initial plot.")
        elif config_has_changed:
            logging.info("Configuration changed, but Apply not clicked. Showing previous plot.")
            st.warning("Configuration changed. Click 'Apply Changes' (Right Panel) to update the plot.")
            fig_to_display = st.session_state.last_figure # Usa l'ultimo grafico valido
        else:
            # Nessun cambiamento e il bottone non è stato premuto, usa l'ultimo grafico valido
             fig_to_display = st.session_state.last_figure

        # --- Generazione Effettiva del Grafico (se necessario) ---
        if regenerate_plot:
            if x_col_selected and y_col_selected and key_selected:
                 try:
                     df_prepared_plot = st.session_state.processed_data[key_selected]['prepared_df']
                     metadata_plot = st.session_state.processed_data[key_selected]['metadata']
                     segments_plot = st.session_state.segments # Usa i segmenti correnti
                     # Applica lo smoothing selezionato
                     df_display = dp.apply_smoothing(df_prepared_plot, smoothing_selected, dp.TIME_COL_SECONDS)
                     # Crea il grafico
                     fig = create_plot(df_display, x_col_selected, y_col_selected, smoothing_selected,
                                       metadata_plot, segments_plot)
                     # Memorizza il nuovo grafico e la sua configurazione
                     st.session_state.last_figure = fig
                     st.session_state.last_applied_config = current_plot_config
                     fig_to_display = fig
                     logging.info(f"Plot regenerated for {key_selected}, {x_col_selected} vs {y_col_selected}, {smoothing_selected}")
                 except Exception as e_plot:
                     st.error(f"Error generating plot: {e_plot}")
                     logging.error("Error during plot generation", exc_info=True)
                     st.session_state.last_figure = go.Figure() # Salva un grafico vuoto in caso di errore
                     st.session_state.last_applied_config = {}
                     fig_to_display = st.session_state.last_figure
            else:
                 st.error("Cannot generate plot: Missing X/Y selection or dataset.")
                 fig_to_display = go.Figure()
                 st.session_state.last_figure = fig_to_display
                 st.session_state.last_applied_config = {} # Reset config if plot failed

        # --- Visualizzazione del Grafico e Gestione Eventi Selezione ---
        if fig_to_display is not None:
            # Usa una chiave stabile per preservare lo stato di zoom/pan tra i rerun
            # MA deve cambiare se la *struttura dati* sottostante cambia (es. cambio dataset)
            chart_key = f"main_chart_{key_selected}"
            event_data = st.plotly_chart(fig_to_display, key=chart_key, use_container_width=True, on_select="rerun")

            # --- Process Selection Event for Click-to-Transfer (invariato) ---
            select_info = event_data.get('select') if event_data else None
            current_event_data = select_info.get('points', []) if select_info else None
            is_new_event = (current_event_data is not None and current_event_data != st.session_state.last_select_event_data)

            if is_new_event:
                logging.info(f"Selection event: {len(current_event_data)} pts.")
                if len(current_event_data) >= 1:
                     selected_point = current_event_data[0]
                     x_sel, y_sel = selected_point.get('x'), selected_point.get('y')
                     if x_sel is not None and y_sel is not None:
                         logging.debug(f"Plot selected: ({x_sel:.2f}, {y_sel:.2f})") # Debug log
                         target = st.session_state.selection_target
                         if target == 'P1':
                             st.session_state.manual_x1 = x_sel; st.session_state.manual_y1 = y_sel
                             st.session_state.selection_target = 'P2'
                             logging.info("Updated P1 from selection.")
                         elif target == 'P2':
                             st.session_state.manual_x2 = x_sel; st.session_state.manual_y2 = y_sel
                             st.session_state.selection_target = 'P1'
                             logging.info("Updated P2 from selection.")
                         st.session_state.last_select_event_data = current_event_data
                         st.rerun() # Rerun to update the number_input widgets
                     else: logging.warning("Selection event missing coordinate data."); st.session_state.last_select_event_data = current_event_data; st.rerun()
                else: logging.info("Selection event had no points."); st.session_state.last_select_event_data = None
            elif not is_new_event and st.session_state.last_select_event_data is not None:
                 st.session_state.last_select_event_data = None # Clear if event hasn't changed
        else:
            st.caption("Plot will be displayed here after applying changes.")


    # --- Display Segment Information (Below Columns) ---
    st.markdown("---")
    st.subheader("Defined Segment Details")
    segments = st.session_state.get('segments', [])
    if segments and isinstance(segments, list):
        data_to_display = []
        for i, seg in enumerate(segments):
            if isinstance(seg, dict) and all(k in seg for k in ['start', 'end', 'slope']):
                try: # Aggiunto try-except per robustezza formattazione
                     data_to_display.append({
                        "Seg #": i + 1,
                        "Start X": f"{seg['start'][0]:.2f}", "Start Y": f"{seg['start'][1]:.2f}",
                        "End X": f"{seg['end'][0]:.2f}", "End Y": f"{seg['end'][1]:.2f}",
                        "Slope (m)": f"{seg['slope']:.4f}" })
                except (TypeError, IndexError, KeyError) as e_seg_disp:
                     logging.warning(f"Error formatting segment {i+1} for display: {e_seg_disp}")
                     data_to_display.append({"Seg #": i + 1, "Start X": "Error", "Start Y": "Error", "End X": "Error", "End Y": "Error", "Slope (m)": "Error" })
        try: # Aggiunto try-except per creazione DataFrame
             df_display_segs = pd.DataFrame(data_to_display)
             if not df_display_segs.empty:
                 st.dataframe(df_display_segs.set_index('Seg #'), use_container_width=True)
             else: st.caption("No valid segments to display.")
        except Exception as e_df:
             st.error(f"Error creating segments table: {e_df}")
             logging.error("Error creating segments DataFrame", exc_info=True)
    else: st.caption("No segments defined yet.")

# --- Footer (invariato) ---
st.sidebar.markdown("---"); st.sidebar.markdown(f"*{APP_TITLE}*"); st.sidebar.info(f"📍 Sassari, Sardinia, Italy")
now_local = datetime.now(); timezone_hint = "CEST"; st.sidebar.caption(f"Generated: {now_local.strftime('%Y-%m-%d %H:%M:%S')} {timezone_hint}")