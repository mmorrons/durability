# app.py
import dash
from dash import dcc, html, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import os
import logging

import data_processing as dp # Assuming data_processing.py is in the same folder

# --- Configuration & Logging ---
APP_TITLE = "📊 Dash Segment Analyzer"
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s-%(levelname)s-%(message)s')

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = APP_TITLE

# --- Reusable Functions for Layout ---
def create_dropdown(id, options=None, placeholder="Select...", value=None, **kwargs):
    if options is None: options = []
    return dcc.Dropdown(id=id, options=options, placeholder=placeholder, value=value, clearable=False, **kwargs)

# --- App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(APP_TITLE), width=12)),
    dbc.Row([
        dbc.Col([ dcc.Upload(id='upload-data', children=html.Div(['Drag & Drop or ', html.A('Select Files')]),
                    style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0px'},
                    multiple=True) ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Dataset:"), create_dropdown(id='dataset-dropdown', placeholder="Load files first...", disabled=True), html.Br(),
            dbc.Label("Smoothing Method:"), create_dropdown(id='smoothing-dropdown', options=[{"label": m, "value": m} for m in ["Raw Data", "10 Breaths MA", "15 Breaths MA", "20 Breaths MA", "30 Breaths MA", "5 Sec MA", "10 Sec MA", "15 Sec MA", "20 Sec MA", "30 Sec MA"]], value="Raw Data"), html.Br(),
            dbc.Label("X-Axis Variable:"), create_dropdown(id='xaxis-dropdown', placeholder="Select dataset first...", disabled=True), html.Br(),
            dbc.Label("Y-Axis Variable:"), create_dropdown(id='yaxis-dropdown', placeholder="Select X-axis first...", disabled=True),
        ], md=3),
        dbc.Col([dcc.Graph(id='main-plot', config={'displayModeBar': True})], md=9),
    ]),
    dbc.Row([
         dbc.Col([ dbc.Button("Clear Current Click (P1)", id="clear-p1-button", color="warning", className="me-2", size="sm", disabled=True),
              dbc.Button("Reset All Segments", id="reset-segments-button", color="danger", size="sm", disabled=True),
              html.P(id='status-message', children="Load files to begin.", style={'marginTop': '10px'}) ], width=12)
    ]),
    dcc.Store(id='processed-data-store', storage_type='session'),
    dcc.Store(id='current-analysis-state', storage_type='session', data={'p1': None, 'key': None})
], fluid=True)

# --- Callbacks ---
@app.callback(
    Output('processed-data-store', 'data'), Output('dataset-dropdown', 'options'), Output('dataset-dropdown', 'value'), Output('dataset-dropdown', 'disabled'), Output('status-message', 'children', allow_duplicate=True),
    Input('upload-data', 'contents'), State('upload-data', 'filename'), State('processed-data-store', 'data'),
    prevent_initial_call=True
)
def process_uploaded_files(list_of_contents, list_of_names, existing_data):
    if list_of_contents is None: return no_update
    processed_data = existing_data if existing_data else {}
    new_files_count, error_files, skipped_files = 0, [], []
    for content, name in zip(list_of_contents, list_of_names):
        metadata, unique_key = dp.parse_filename(name)
        if unique_key is None: unique_key = name
        if unique_key in processed_data: skipped_files.append(name); logging.info(f"Skip: {unique_key}"); continue
        try:
            content_type, content_string = content.split(','); decoded = base64.b64decode(content_string)
            file_like = io.BytesIO(decoded)
            # --- >>> UPDATED CALL to load_data <<< ---
            raw_df = dp.load_data(file_like, name) # Pass content stream AND filename
            # --- >>> END UPDATE <<< ---
            if raw_df is not None: # Check if loading succeeded
                prep_df = dp.prepare_data(raw_df)
                if prep_df is not None and not prep_df.empty:
                    processed_data[unique_key] = {'metadata': metadata if metadata else {'filename': name, 'display_name': name}, 'prepared_json': prep_df.to_json(orient='split', date_format='iso'), 'segments': []}
                    new_files_count += 1; logging.info(f"Processed key: {unique_key}")
                else: error_files.append(name); logging.error(f"Prep failed {name}")
            else: error_files.append(name); logging.error(f"Load failed {name}")
        except Exception as e: logging.error(f"Error processing file {name}: {e}", exc_info=True); error_files.append(name)
    display_options = sorted([{'label': d['metadata'].get('display_name', k), 'value': k} for k, d in processed_data.items()], key=lambda x: x['label'])
    dropdown_disabled = not bool(display_options)
    current_selection = display_options[0]['value'] if display_options else None
    status = f"Loaded {new_files_count}. " if new_files_count else ""
    if skipped_files: status += f"Skipped {len(skipped_files)}. "
    if error_files: status += f"Failed {len(error_files)}. "
    status += "Select dataset." if display_options else "Load valid files."
    return processed_data, display_options, current_selection, dropdown_disabled, status

@app.callback(
    Output('xaxis-dropdown', 'options'), Output('xaxis-dropdown', 'value'), Output('xaxis-dropdown', 'disabled'),
    Output('yaxis-dropdown', 'options'), Output('yaxis-dropdown', 'value'), Output('yaxis-dropdown', 'disabled'),
    Output('status-message', 'children', allow_duplicate=True),
    Output('current-analysis-state', 'data', allow_duplicate=True),
    Input('dataset-dropdown', 'value'), State('processed-data-store', 'data'),
    prevent_initial_call=True
)
def update_axis_options(selected_key, stored_data):
    if not selected_key or not stored_data or selected_key not in stored_data: return [], None, True, [], None, True, "Select valid dataset.", {'p1': None, 'key': None}
    try:
        dataset = stored_data[selected_key]; df_prepared = pd.read_json(dataset['prepared_json'], orient='split')
        if df_prepared.empty: raise ValueError("DataFrame empty")
        numeric_cols = df_prepared.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols: return [], None, True, [], None, True, f"No numeric cols.", {'p1': None, 'key': selected_key}
        axis_options = [{'label': col, 'value': col} for col in numeric_cols]
        default_x = dp.TIME_COL_SECONDS if dp.TIME_COL_SECONDS in numeric_cols else numeric_cols[0]
        y_options = [{'label': col, 'value': col} for col in numeric_cols if col != default_x]
        default_y = None
        if y_options:
             common_y = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', dp.WATT_COL, 'FC'] # Use dp.WATT_COL
             for opt in y_options:
                  if opt['value'] in common_y: default_y = opt['value']; break
             if default_y is None: default_y = y_options[0]['value']
        new_analysis_state = {'p1': None, 'key': selected_key}
        status = f"Selected {dataset['metadata'].get('display_name', selected_key)}. Choose axes."
        return axis_options, default_x, False, y_options, default_y, not bool(y_options), status, new_analysis_state
    except Exception as e: logging.error(f"Error updating axes {selected_key}: {e}"); return [], None, True, [], None, True, f"Error loading {selected_key}.", {'p1': None, 'key': None}

@app.callback(
    Output('yaxis-dropdown', 'options', allow_duplicate=True), Output('yaxis-dropdown', 'value', allow_duplicate=True), Output('yaxis-dropdown', 'disabled', allow_duplicate=True),
    Input('xaxis-dropdown', 'value'), State('xaxis-dropdown', 'options'),
    prevent_initial_call=True
)
def update_yaxis_options_on_x_change(selected_x, all_numeric_options):
    if not selected_x or not all_numeric_options: return [], None, True
    all_numeric_cols = [opt['value'] for opt in all_numeric_options]
    y_options_list = [col for col in all_numeric_cols if col != selected_x]
    y_options = [{'label': col, 'value': col} for col in y_options_list]
    default_y = None
    if y_options:
        common_y = ['V\'O2/kg', 'V\'O2', 'V\'CO2', 'V\'E', dp.WATT_COL, 'FC'] # Use dp.WATT_COL
        for opt in y_options:
            if opt['value'] in common_y: default_y = opt['value']; break
        if default_y is None: default_y = y_options[0]['value']
    return y_options, default_y, not bool(y_options)

@app.callback(
    Output('main-plot', 'figure'), Output('status-message', 'children'), Output('current-analysis-state', 'data'),
    Output('processed-data-store', 'data', allow_duplicate=True), Output('clear-p1-button', 'disabled'), Output('reset-segments-button', 'disabled'),
    Input('dataset-dropdown', 'value'), Input('smoothing-dropdown', 'value'), Input('xaxis-dropdown', 'value'), Input('yaxis-dropdown', 'value'),
    Input('main-plot', 'clickData'), Input('clear-p1-button', 'n_clicks'), Input('reset-segments-button', 'n_clicks'),
    State('processed-data-store', 'data'), State('current-analysis-state', 'data'),
    prevent_initial_call=True
)
def update_figure_and_state(selected_key, smoothing, x_col, y_col, click_data, clear_p1_clicks, reset_segs_clicks, stored_data, current_state):
    trigger_id = ctx.triggered_id if ctx.triggered_id else 'initial'; logging.debug(f"Callback triggered by: {trigger_id}")
    if not selected_key or not stored_data or selected_key not in stored_data: return go.Figure(), "Select dataset.", {'p1': None, 'key': None}, no_update, True, True

    dataset = stored_data[selected_key]; df_prepared = pd.read_json(dataset['prepared_json'], orient='split')
    metadata = dataset['metadata']; segments = dataset.get('segments', [])
    p1 = current_state.get('p1') if current_state and current_state.get('key') == selected_key else None
    status = None

    if trigger_id == 'reset-segments-button':
        logging.info(f"Resetting segments for {selected_key}"); p1 = None; segments = []; stored_data[selected_key]['segments'] = segments
        status = "Segments reset."
    elif trigger_id == 'clear-p1-button':
        if p1: logging.info(f"Clearing P1 for {selected_key}"); p1 = None; status = "P1 cleared."
        else: return no_update, "No P1 to clear.", no_update, no_update, True, not bool(segments)
    elif trigger_id == 'main-plot' and click_data:
        point = click_data['points'][0]; x_clicked, y_clicked = point['x'], point['y']; logging.info(f"Click:({x_clicked:.2f}, {y_clicked:.2f})")
        if p1 is None: p1 = (x_clicked, y_clicked); status = "P1 selected. Click P2."
        else:
            p2 = (x_clicked, y_clicked)
            if abs(p1[0] - p2[0]) > 1e-6 or abs(p1[1] - p2[1]) > 1e-6:
                 slope = dp.calculate_slope(p1, p2); new_segment = {'start': p1, 'end': p2, 'slope': slope}
                 segments.append(new_segment); stored_data[selected_key]['segments'] = segments
                 seg_count = len(segments); print(f"Seg {seg_count} added. m={slope:.4f}"); status = f"Seg {seg_count} added. Select P1."
                 p1 = None
            else: status = "P2 same as P1. P1 cleared."; p1 = None

    df_display = dp.apply_smoothing(df_prepared, smoothing, dp.TIME_COL_SECONDS)
    if df_display is None or df_display.empty: return go.Figure(), f"No data after smoothing '{smoothing}'.", {'p1': p1, 'key': selected_key}, stored_data, not bool(p1), not bool(segments)
    if not x_col or not y_col or x_col not in df_display.columns or y_col not in df_display.columns: return go.Figure(), "Select X & Y cols.", {'p1': p1, 'key': selected_key}, stored_data, not bool(p1), not bool(segments)

    fig = go.Figure(); title = f"{y_col} vs {x_col} ({smoothing}) - {metadata.get('display_name', selected_key)}"
    if not (df_display[x_col].isnull().all() or df_display[y_col].isnull().all()):
         fig.add_trace(go.Scattergl(x=df_display[x_col], y=df_display[y_col], mode='markers', marker=dict(color='blue', size=5, opacity=0.7), name=f'Data'))
    else: fig.add_annotation(text="No valid data.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if isinstance(segments, list):
        for i, segment in enumerate(segments):
            if segment and 'start' in segment and 'end' in segment and 'slope' in segment:
                p_s, p_e, m = segment['start'], segment['end'], segment['slope']; fig.add_trace(go.Scatter(x=[p_s[0], p_e[0]], y=[p_s[1], p_e[1]], mode='lines+markers', line=dict(color='red', width=2), marker=dict(color='red', size=8), name=f'Seg{i+1}(m={m:.2f})'))
                mid_x, mid_y = (p_s[0] + p_e[0]) / 2, (p_s[1] + p_e[1]) / 2; fig.add_annotation(x=mid_x, y=mid_y, text=f' m={m:.2f}', showarrow=False, font=dict(color='red', size=10), xshift=5)
    if p1: fig.add_trace(go.Scatter(x=[p1[0]], y=[p1[1]], mode='markers', marker=dict(color='orange', size=10, symbol='x'), name='P1'))
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, hovermode='closest', title=title, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    if status is None: status = "Plot updated. Click points." if not p1 else "P1 selected. Click P2."
    new_analysis_state = {'p1': p1, 'key': selected_key}; clear_p1_disabled = not bool(p1); reset_segs_disabled = not bool(segments)
    return fig, status, new_analysis_state, stored_data, clear_p1_disabled, reset_segs_disabled

# --- Run the Server ---
if __name__ == '__main__':
    logging.info("Starting Dash server... Access at http://127.0.0.1:8050/")
    app.run(debug=True) # Use app.run (corrected method)