# data_processing.py
import pandas as pd
import numpy as np
import re
from datetime import timedelta # No need for full datetime here, just timedelta
import logging
import os
# Import Streamlit elements used within this module
# If this module were reused outside Streamlit, these would need adjustment
try:
    from streamlit import cache_data, error as st_error
except ImportError:
    logging.warning("Streamlit not found. cache_data decorator and st.error will not function.")
    # Define dummy decorator and function if Streamlit is not available
    def cache_data(func=None, **kwargs):
        if func: return func
        else: return lambda f: f
    def st_error(msg):
        logging.error(f"Streamlit unavailable: {msg}")


# --- Configuration Constants ---
RAW_TIME_COL = 't'
TIME_COL_SECONDS = 'time_seconds'
MARKER_COL = 'Marker'
EXPECTED_HEADER_ROW_INDEX = 142 # 0-based index for Excel row 143
WATT_COL = "W" # Derived Watt column name

# --- Logging Setup ---
# Basic logging, Streamlit/main app might configure root logger further
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

# --- Core Data Functions ---

def parse_filename(filename):
    """Parses the specific filename structure to extract metadata."""
    try:
        base_name = os.path.splitext(os.path.basename(filename))[0]; parts = base_name.split('_')
        if len(parts) < 4: logging.warning(f"Fn '{filename}' fmt issue (<4 parts)."); return None, filename
        first_part = parts[0]; match = re.match(r'^([A-Z]+)(\d+)$', first_part)
        if not match: logging.warning(f"Fn '{filename}': First part '{first_part}' invalid fmt."); return None, filename
        prefix, id_str = match.group(1), match.group(2); surname, test_code = parts[1], parts[-1]; name = " ".join(parts[2:-1])
        try: subject_id = int(id_str)
        except ValueError: logging.error(f"Fn '{filename}': Invalid ID '{id_str}'."); return None, filename
        test_type_map = {"C1": "Cycling Fresh", "C2": "Cycling Fatigued", "T1": "Treadmill Fresh", "T2": "Treadmill Fatigued"}
        test_description = test_type_map.get(test_code, f"Unknown ({test_code})")
        if not name: logging.warning(f"Fn '{filename}': Parsed name empty.")
        unique_key = f"{prefix}{subject_id}_{surname}_{name}_{test_code}"
        display_name = f"{name} {surname} (ID: {subject_id}) - {test_description}"
        metadata = {"filename": filename, "unique_key": unique_key, "display_name": display_name, "subject_id": subject_id, "surname": surname, "name": name, "test_code": test_code, "test_description": test_description, "prefix": prefix }
        # logging.info(f"Parsed metadata for {filename}") # Keep logging minimal here
        return metadata, unique_key
    except Exception as e: logging.error(f"Error parsing filename '{filename}': {e}", exc_info=True); return None, filename

def time_str_to_seconds(time_str):
    """Converts H:MM:SS,ms or MM:SS,ms string to total seconds."""
    if pd.isna(time_str) or not isinstance(time_str, str): return None
    try:
        time_str = time_str.strip(); parts = time_str.split(','); time_part = parts[0]
        milliseconds = 0
        if len(parts) > 1 and parts[1].isdigit(): milliseconds = int(parts[1])
        elif len(parts) > 1: return None # Non-numeric ms part
        time_obj = pd.to_datetime(time_part, format='%H:%M:%S', errors='coerce')
        if pd.isna(time_obj): time_obj = pd.to_datetime(time_part, format='%M:%S', errors='coerce')
        if pd.isna(time_obj): return None # Cannot parse time part
        # Use time properties directly for timedelta calculation
        hour = time_obj.hour if hasattr(time_obj, 'hour') else 0
        minute = time_obj.minute if hasattr(time_obj, 'minute') else 0
        second = time_obj.second if hasattr(time_obj, 'second') else 0
        return timedelta(hours=hour, minutes=minute, seconds=second, milliseconds=milliseconds).total_seconds()
    except Exception as e: logging.warning(f"Error parsing time string '{time_str}': {e}"); return None

# Streamlit caching decorator applied here
@cache_data(show_spinner="Loading data: {_filename}...")
def load_data(_file_content, _filename): # Renamed for clarity with caching
    """Loads data from file content (BytesIO), using filename to determine type."""
    logging.info(f"Loading data from memory stream for: {_filename}")
    try:
        df = None
        if _filename.endswith('.csv'):
            try: df = pd.read_csv(_file_content)
            except pd.errors.ParserError: _file_content.seek(0); df = pd.read_csv(_file_content, sep=';')
            except Exception as e_csv: logging.error(f"Error reading CSV '{_filename}' from stream: {e_csv}"); return None
        elif _filename.endswith(('.xls', '.xlsx')):
            _file_content.seek(0) # Ensure stream is at the beginning for ExcelFile
            try:
                engine = 'openpyxl' if _filename.endswith('.xlsx') else None
                xls = pd.ExcelFile(_file_content, engine=engine)
                sheet_to_read = xls.sheet_names[0]
                logging.info(f"Attempting read '{_filename}' sheet '{sheet_to_read}' row {EXPECTED_HEADER_ROW_INDEX + 1} from stream")
                df = pd.read_excel(xls, sheet_name=sheet_to_read, header=EXPECTED_HEADER_ROW_INDEX)
            except IndexError: logging.error(f"Header row {EXPECTED_HEADER_ROW_INDEX + 1} not found in '{_filename}'."); return None
            except ValueError as e_val: logging.error(f"Invalid header parameter reading '{_filename}': {e_val}"); return None
            except Exception as e_excel: logging.error(f"Error reading Excel '{_filename}' from stream: {e_excel}"); return None
        else: logging.error(f"Unsupported file type: {_filename}"); return None

        if df is None or df.empty: logging.warning(f"Loaded file '{_filename}' is None or empty."); return None
        logging.info(f"Loaded '{_filename}'. Shape: {df.shape}")
        df.attrs['filename'] = _filename # Store filename attribute
        return df
    except Exception as e: logging.error(f"Critical error loading from stream for '{_filename}': {e}"); return None

# Streamlit caching decorator applied here
@cache_data(show_spinner="Preparing data: {_df_raw.attrs.get('filename', 'N/A')}...")
def prepare_data(_df_raw):
    """Prepares raw data: time conversion, unit row handling, Watt derivation, START/STOP filter, numeric conversion."""
    if _df_raw is None or not isinstance(_df_raw, pd.DataFrame) or _df_raw.empty: return None
    df = _df_raw.copy(); filename = df.attrs.get('filename', 'N/A'); logging.info(f"[{filename}] Starting data preparation...")
    marker_col_present = MARKER_COL in df.columns
    if not marker_col_present: logging.warning(f"[{filename}] Marker col '{MARKER_COL}' not found.")

    # --- Time Conversion & Unit Row Removal ---
    rows_dropped = 0
    if RAW_TIME_COL in df.columns:
        logging.info(f"[{filename}] Converting time '{RAW_TIME_COL}' -> '{TIME_COL_SECONDS}'")
        df[TIME_COL_SECONDS] = df[RAW_TIME_COL].apply(time_str_to_seconds)
        null_time_count = df[TIME_COL_SECONDS].isnull().sum()
        if null_time_count > 0:
            logging.warning(f"[{filename}] Found {null_time_count} invalid times. Removing...")
            original_rows = len(df); df.dropna(subset=[TIME_COL_SECONDS], inplace=True); rows_dropped = original_rows - len(df)
            if rows_dropped > 0: logging.info(f"[{filename}] Removed {rows_dropped} rows.")
        df.sort_values(by=TIME_COL_SECONDS, inplace=True, na_position='last'); df.reset_index(drop=True, inplace=True)
    else: logging.warning(f"[{filename}] Col '{RAW_TIME_COL}' not found. Using index."); df[TIME_COL_SECONDS] = df.index
    if df.empty: logging.error(f"[{filename}] Empty after time cleaning."); return None

    # --- Derive Watt Column ---
    if marker_col_present:
        logging.info(f"[{filename}] Deriving Watt column '{WATT_COL}'.")
        try:
            watt_regex = re.compile(r'^(\d+(\.\d+)?)\s*W?$')
            watt_markers = df[MARKER_COL].apply(lambda m: float(match.group(1)) if pd.notna(m) and (match := watt_regex.match(str(m).strip().upper())) else np.nan)
            df[WATT_COL] = watt_markers.ffill()
            start_indices = df[df[MARKER_COL].astype(str).str.upper().str.strip() == 'START'].index
            start_idx = start_indices[0] if not start_indices.empty else 0
            if start_idx > 0: df.loc[0:start_idx-1, WATT_COL] = df.loc[0:start_idx-1, WATT_COL].fillna(0) # Use simple range slicing after reset_index
            if start_idx in df.index and pd.isna(df.loc[start_idx, WATT_COL]): df.loc[start_idx, WATT_COL] = 0
            df[WATT_COL].fillna(0, inplace=True); df[WATT_COL] = pd.to_numeric(df[WATT_COL], errors='coerce'); df[WATT_COL].fillna(0, inplace=True)
            logging.info(f"[{filename}] Derived '{WATT_COL}' column created.")
        except Exception as e_watt: logging.error(f"[{filename}] Failed Watt derivation: {e_watt}"); df[WATT_COL] = 0
    else: logging.warning(f"[{filename}] No '{MARKER_COL}', creating '{WATT_COL}'=0."); df[WATT_COL] = 0

    # --- Filter by START/STOP ---
    df_filtered = df # Default to current df
    if marker_col_present:
        logging.info(f"[{filename}] Filtering by START/STOP.")
        start_indices_filter = df[df[MARKER_COL].astype(str).str.upper().str.strip() == 'START'].index
        if not start_indices_filter.empty:
            start_idx_filter = start_indices_filter[0]; logging.info(f"[{filename}] Found START at index {start_idx_filter}.")
            df_after_start = df.loc[start_idx_filter:] # Slice from START index onwards
            stop_indices_filter = df_after_start[df_after_start[MARKER_COL].astype(str).str.upper().str.strip() == 'STOP'].index
            if not stop_indices_filter.empty:
                stop_idx_filter = stop_indices_filter[0]; logging.info(f"[{filename}] Found STOP at index {stop_idx_filter}.")
                # Slice using index positions based on the current df's index
                start_pos = df.index.get_loc(start_idx_filter)
                stop_pos = df.index.get_loc(stop_idx_filter)
                df_filtered = df.iloc[start_pos : stop_pos].copy() # Exclusive of stop position
                logging.info(f"[{filename}] Extracted {len(df_filtered)} rows.")
            else: logging.warning(f"[{filename}] No STOP after START. Using data from START."); df_filtered = df.loc[start_idx_filter:].copy()
        else: logging.warning(f"[{filename}] No START marker found. Using all rows.")
    else: logging.warning(f"[{filename}] No '{MARKER_COL}', skipping START/STOP filter.")
    if df_filtered.empty: logging.error(f"[{filename}] Empty after START/STOP filter."); return None

    # --- Time Normalization ---
    if TIME_COL_SECONDS in df_filtered.columns and not df_filtered.empty:
        try:
            start_time_norm = df_filtered[TIME_COL_SECONDS].iloc[0]
            if pd.notna(start_time_norm):
                logging.info(f"[{filename}] Normalizing time from {start_time_norm:.2f}s.")
                df_filtered[TIME_COL_SECONDS] = df_filtered[TIME_COL_SECONDS] - start_time_norm
                df_filtered.loc[df_filtered[TIME_COL_SECONDS] < 0, TIME_COL_SECONDS] = 0
            else: logging.warning(f"[{filename}] Cannot normalize time.")
        except Exception as e_norm: logging.error(f"[{filename}] Time normalization error: {e_norm}")

    # --- Final Numeric Conversion Check ---
    logging.info(f"[{filename}] Final numeric conversion check ({len(df_filtered)} rows)...")
    cols_converted = []; cols_failed = []
    for col in df_filtered.columns:
        if col in [TIME_COL_SECONDS, RAW_TIME_COL, WATT_COL] or pd.api.types.is_numeric_dtype(df_filtered[col]): continue
        original_dtype = df_filtered[col].dtype
        try:
            temp_col = df_filtered[col]; converted_col = None
            if original_dtype == 'object':
                 try: converted_col = pd.to_numeric(temp_col, errors='raise')
                 except (ValueError, TypeError):
                      temp_col = temp_col.astype(str).str.replace(',', '.', regex=False)
                      converted_col = pd.to_numeric(temp_col, errors='raise')
            else: converted_col = pd.to_numeric(temp_col, errors='raise')
            df_filtered[col] = converted_col
            if df_filtered[col].dtype != original_dtype: cols_converted.append(col)
        except (ValueError, TypeError): cols_failed.append(col); logging.debug(f"[{filename}] Final num convert failed '{col}'.")
    if cols_converted: logging.info(f"[{filename}] Final conversion: {cols_converted}")
    if cols_failed: logging.warning(f"[{filename}] Final conversion failed: {cols_failed}")

    # --- Final Cleanup and Return ---
    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered.attrs['filename'] = filename
    logging.info(f"[{filename}] Preparation finished. Shape: {df_filtered.shape}")
    return df_filtered

# Streamlit caching decorator applied here
@cache_data(show_spinner="Applying smoothing: {method}...")
def apply_smoothing(df_prepared, method, time_col_sec):
    """Applies selected smoothing to the prepared data DataFrame. Includes 'Sec' fix."""
    if df_prepared is None or df_prepared.empty: return df_prepared
    if not isinstance(df_prepared, pd.DataFrame): return None
    logging.info(f"Applying smoothing method: {method}"); filename = df_prepared.attrs.get('filename', 'N/A')
    if method == "Raw Data": return df_prepared.copy()

    # Check time column using constant
    if "Sec" in method: # Use "Sec"
        if time_col_sec not in df_prepared.columns: logging.error(f"[{filename}] Time col '{time_col_sec}' missing."); st_error(f"[{filename}] Time column missing."); return df_prepared.copy()
        if not pd.api.types.is_numeric_dtype(df_prepared[time_col_sec]): logging.error(f"[{filename}] Time col '{time_col_sec}' not numeric."); st_error(f"[{filename}] Time column not numeric."); return df_prepared.copy()

    numeric_cols = df_prepared.select_dtypes(include=np.number).columns.tolist()
    if time_col_sec in numeric_cols: numeric_cols.remove(time_col_sec)
    cols_to_exclude = ['subject_id', 'ID', 'index']
    numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
    if not numeric_cols: logging.warning(f"[{filename}] No numeric cols to smooth."); return df_prepared.copy()

    cols_to_keep_raw = df_prepared.select_dtypes(exclude=np.number).columns.tolist()
    if time_col_sec in df_prepared.columns and time_col_sec not in cols_to_keep_raw: cols_to_keep_raw.append(time_col_sec)
    if MARKER_COL in df_prepared.columns and MARKER_COL not in cols_to_keep_raw: cols_to_keep_raw.append(MARKER_COL)
    cols_to_keep_raw.extend([col for col in cols_to_exclude if col in df_prepared.columns])
    cols_to_keep_raw = sorted(list(set(c for c in cols_to_keep_raw if c in df_prepared.columns)))

    df_smoothed_numeric = pd.DataFrame(index=df_prepared.index)
    try:
        if "Breath" in method:
            match = re.search(r'(\d+)\s*Breath', method);
            if not match: raise ValueError(f"Cannot parse breath window: {method}")
            window_size = int(match.group(1))
            smoothed_data = df_prepared[numeric_cols].rolling(window=window_size, min_periods=1).mean()
            df_smoothed_numeric[numeric_cols] = smoothed_data[numeric_cols]
        elif "Sec" in method: # Corrected check
            match = re.search(r'(\d+)\s*Sec', method); # Corrected regex
            if not match: raise ValueError(f"Cannot parse second window: {method}")
            time_window_str = f"{int(match.group(1))}s"
            logging.debug(f"[{filename}] Applying {time_window_str} time rolling on '{time_col_sec}'.")
            df_temp = df_prepared[[time_col_sec] + numeric_cols].dropna(subset=[time_col_sec]).copy()
            if df_temp.empty: logging.warning(f"[{filename}] No valid time data."); return df_prepared.copy()
            try: timedelta_col_name = f"_{time_col_sec}_td_"; df_temp[timedelta_col_name] = pd.to_timedelta(df_temp[time_col_sec], unit='s'); df_temp = df_temp.sort_values(by=timedelta_col_name)
            except Exception as e: raise ValueError(f"Time col '{time_col_sec}' error: {e}")
            rolling_obj = df_temp.rolling(window=time_window_str, on=timedelta_col_name, min_periods=1, closed='right')
            smoothed_data = rolling_obj[numeric_cols].mean()
            df_smoothed_numeric = pd.DataFrame(np.nan, index=df_prepared.index, columns=numeric_cols)
            df_smoothed_numeric.loc[df_temp.index, numeric_cols] = smoothed_data[numeric_cols].values
        else: raise ValueError(f"Unknown smoothing method: {method}")

        raw_cols_df = df_prepared[cols_to_keep_raw]; df_final = pd.concat([raw_cols_df, df_smoothed_numeric[numeric_cols]], axis=1)
        original_order = df_prepared.columns.tolist(); final_cols_order = [c for c in original_order if c in df_final] + [c for c in df_final if c not in original_order]; df_final = df_final.reindex(columns=final_cols_order)
        logging.info(f"[{filename}] Smoothing '{method}' applied. Shape: {df_final.shape}"); df_final.attrs['filename'] = filename; return df_final
    except Exception as e: logging.error(f"[{filename}] Error smoothing '{method}': {e}"); st_error(f"[{filename}] Failed smoothing '{method}'."); return df_prepared.copy()


def calculate_slope(p1, p2):
    """Calculates slope between two points (tuples), handles vertical lines."""
    if p1 is None or p2 is None: return 0
    delta_x = p2[0] - p1[0]; delta_y = p2[1] - p1[1]
    if abs(delta_x) < 1e-9: return np.inf if delta_y > 0 else -np.inf if delta_y < 0 else 0
    return delta_y / delta_x