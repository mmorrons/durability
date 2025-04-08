# data_processing.py (Caching Disabled Version)

import pandas as pd
import numpy as np
import re
from datetime import timedelta
import logging
import os

# --- Configuration Constants ---
RAW_TIME_COL = 't'
TIME_COL_SECONDS = 'time_seconds'
MARKER_COL = 'Marker'
EXPECTED_HEADER_ROW_INDEX = 142 # 0-based index for Excel row 143
WATT_COL = "W" # Derived Watt column name

# --- Logging Setup ---
# Basic logging, main app might configure root logger further
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a specific logger

# --- Core Data Functions ---

def parse_filename(filename):
    """Parses the specific filename structure to extract metadata."""
    logger.debug(f"Parsing filename: {filename}")
    try:
        base_name = os.path.splitext(os.path.basename(filename))[0]; parts = base_name.split('_')
        if len(parts) < 4: logger.warning(f"Fn '{filename}' fmt issue (<4 parts)."); return None, filename
        first_part = parts[0]; match = re.match(r'^([A-Z]+)(\d+)$', first_part)
        if not match: logger.warning(f"Fn '{filename}': First part '{first_part}' invalid fmt."); return None, filename
        prefix, id_str = match.group(1), match.group(2); surname, test_code = parts[1], parts[-1]; name = " ".join(parts[2:-1])
        try: subject_id = int(id_str)
        except ValueError: logger.error(f"Fn '{filename}': Invalid ID '{id_str}'."); return None, filename
        test_type_map = {"C1": "Cycling Fresh", "C2": "Cycling Fatigued", "T1": "Treadmill Fresh", "T2": "Treadmill Fatigued"}
        test_description = test_type_map.get(test_code, f"Unknown ({test_code})")
        if not name: logger.warning(f"Fn '{filename}': Parsed name empty.")
        unique_key = f"{prefix}{subject_id}_{surname}_{name}_{test_code}"
        display_name = f"{name} {surname} (ID: {subject_id}) - {test_description}"
        metadata = {"filename": filename, "unique_key": unique_key, "display_name": display_name, "subject_id": subject_id, "surname": surname, "name": name, "test_code": test_code, "test_description": test_description, "prefix": prefix }
        logger.info(f"Parsed metadata for {filename}: {display_name}")
        return metadata, unique_key
    except Exception as e: logger.error(f"Error parsing filename '{filename}': {e}", exc_info=True); return None, filename

def time_str_to_seconds(time_str):
    """Converts H:MM:SS,ms or MM:SS,ms string to total seconds."""
    if pd.isna(time_str) or not isinstance(time_str, str): return None
    try:
        time_str_cleaned = time_str.strip(); parts = time_str_cleaned.split(','); time_part = parts[0]
        milliseconds = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        time_obj = pd.to_datetime(time_part, format='%H:%M:%S', errors='coerce')
        if pd.isna(time_obj): time_obj = pd.to_datetime(time_part, format='%M:%S', errors='coerce')
        if pd.isna(time_obj): logger.warning(f"Could not parse time part: '{time_part}'"); return None
        hour = time_obj.hour if hasattr(time_obj, 'hour') else 0
        minute = time_obj.minute if hasattr(time_obj, 'minute') else 0
        second = time_obj.second if hasattr(time_obj, 'second') else 0
        return timedelta(hours=hour, minutes=minute, seconds=second, milliseconds=milliseconds).total_seconds()
    except Exception as e: logger.warning(f"Error parsing time string '{time_str}': {e}"); return None

# --- NO @st.cache_data ---
def load_data(file_content_or_path, filename):
    """Loads data from file content (BytesIO) or path, using filename to determine type."""
    logger.info(f"Loading data for: {filename} (Cache DISABLED)")
    df = None
    is_stream = hasattr(file_content_or_path, 'seek') # Check if it's a stream (like BytesIO)

    try:
        if filename.lower().endswith('.csv'):
            try:
                # Ensure stream is at the beginning if it's a stream
                if is_stream: file_content_or_path.seek(0)
                df = pd.read_csv(file_content_or_path)
            except pd.errors.ParserError:
                logger.warning(f"CSV parsing failed for {filename}, trying separator=';'")
                if is_stream: file_content_or_path.seek(0)
                df = pd.read_csv(file_content_or_path, sep=';')
            except Exception as e_csv:
                logger.error(f"Error reading CSV '{filename}': {e_csv}", exc_info=True); return None

        elif filename.lower().endswith(('.xls', '.xlsx')):
            engine = 'openpyxl' if filename.lower().endswith('.xlsx') else None
            # Ensure stream is at the beginning for ExcelFile if it's a stream
            if is_stream: file_content_or_path.seek(0)
            try:
                xls = pd.ExcelFile(file_content_or_path, engine=engine)
                if not xls.sheet_names:
                    logger.error(f"No sheets found in Excel file: {filename}"); return None
                sheet_to_read = xls.sheet_names[0]
                logger.info(f"Reading '{filename}' sheet '{sheet_to_read}' header row {EXPECTED_HEADER_ROW_INDEX + 1}")
                df = pd.read_excel(xls, sheet_name=sheet_to_read, header=EXPECTED_HEADER_ROW_INDEX)
            except IndexError:
                logger.error(f"Header row {EXPECTED_HEADER_ROW_INDEX + 1} not found in '{filename}'. Check file format/structure."); return None
            except ValueError as e_val:
                logger.error(f"Invalid header parameter reading '{filename}': {e_val}"); return None
            except Exception as e_excel:
                logger.error(f"Error reading Excel '{filename}': {e_excel}", exc_info=True); return None
        else:
            logger.error(f"Unsupported file type: {filename}"); return None

        if df is None or df.empty:
            logger.warning(f"Loaded file '{filename}' resulted in an empty DataFrame."); return None

        logger.info(f"Successfully loaded '{filename}'. Shape: {df.shape}")
        df.attrs['filename'] = filename # Store filename attribute
        return df

    except Exception as e:
        logger.error(f"Critical error loading data for '{filename}': {e}", exc_info=True); return None


# --- NO @st.cache_data ---
def prepare_data(df_raw):
    """Prepares raw data: time conversion, Watt derivation, START/STOP filter, numeric conversion."""
    filename = df_raw.attrs.get('filename', 'N/A')
    logger.info(f"[{filename}] Starting data preparation (Cache DISABLED)")
    if df_raw is None or not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        logger.error(f"[{filename}] Invalid input DataFrame to prepare_data."); return None

    df = df_raw.copy() # Work on a copy
    initial_rows = len(df)

    marker_col_present = MARKER_COL in df.columns
    if not marker_col_present: logger.warning(f"[{filename}] Marker col '{MARKER_COL}' not found.")

    # --- Time Conversion ---
    rows_dropped_time = 0
    if RAW_TIME_COL in df.columns:
        logger.info(f"[{filename}] Converting time col '{RAW_TIME_COL}' -> '{TIME_COL_SECONDS}'")
        df[TIME_COL_SECONDS] = df[RAW_TIME_COL].apply(time_str_to_seconds)
        invalid_time_count = df[TIME_COL_SECONDS].isnull().sum()
        if invalid_time_count > 0:
            logger.warning(f"[{filename}] Found {invalid_time_count} invalid times. Removing corresponding rows.")
            df.dropna(subset=[TIME_COL_SECONDS], inplace=True)
            rows_dropped_time = initial_rows - len(df)
        # Ensure time column is numeric after conversion
        df[TIME_COL_SECONDS] = pd.to_numeric(df[TIME_COL_SECONDS], errors='coerce')
        # Sort by numeric time
        df.sort_values(by=TIME_COL_SECONDS, inplace=True, na_position='last')
    else:
        logger.warning(f"[{filename}] Raw time col '{RAW_TIME_COL}' not found. Cannot convert time."); return None # Critical if time needed

    if df.empty: logger.error(f"[{filename}] DataFrame empty after time cleaning."); return None

    # --- Derive Watt Column ---
    if marker_col_present:
        logger.info(f"[{filename}] Deriving Watt column '{WATT_COL}'.")
        try:
            # Match numbers possibly followed by 'W', handling floats
            watt_regex = re.compile(r'^\s*(\d+(\.\d+)?)\s*W?\s*$', re.IGNORECASE)
            # Apply regex and extract the numeric part, convert to float
            watt_values = df[MARKER_COL].astype(str).str.extract(watt_regex, expand=False)[0]
            df[WATT_COL] = pd.to_numeric(watt_values, errors='coerce')
            # Forward fill the extracted Watt values
            df[WATT_COL].ffill(inplace=True)
            # Handle values before START
            start_indices = df.index[df[MARKER_COL].astype(str).str.upper().str.strip() == 'START']
            first_start_idx = start_indices[0] if not start_indices.empty else -1
            if first_start_idx != -1:
                 # Fill NaN with 0 before the first START index
                 df.loc[:first_start_idx, WATT_COL] = df.loc[:first_start_idx, WATT_COL].fillna(0)
            # Fill remaining NaNs (e.g., at the very beginning or if no START) with 0
            df[WATT_COL].fillna(0, inplace=True)
            logger.info(f"[{filename}] Derived '{WATT_COL}'.")
        except Exception as e_watt:
            logger.error(f"[{filename}] Failed Watt derivation: {e_watt}", exc_info=True)
            df[WATT_COL] = 0 # Fallback to 0
    else:
        logger.warning(f"[{filename}] No '{MARKER_COL}', creating '{WATT_COL}'=0."); df[WATT_COL] = 0

    # --- Filter by START/STOP ---
    df_filtered = df # Default to current df if no markers or filter fails
    if marker_col_present:
        logger.info(f"[{filename}] Filtering by START/STOP markers.")
        start_indices_filter = df.index[df[MARKER_COL].astype(str).str.upper().str.strip() == 'START']
        stop_indices_filter = df.index[df[MARKER_COL].astype(str).str.upper().str.strip() == 'STOP']

        start_idx = start_indices_filter[0] if not start_indices_filter.empty else -1
        stop_idx = stop_indices_filter[0] if not stop_indices_filter.empty else -1

        if start_idx != -1:
            logger.info(f"[{filename}] Found START at index {start_idx}.")
            if stop_idx != -1 and stop_idx > start_idx:
                 logger.info(f"[{filename}] Found STOP at index {stop_idx}.")
                 df_filtered = df.loc[start_idx:stop_idx].copy() # Inclusive of start and stop index
                 logger.info(f"[{filename}] Filtered between START and STOP, {len(df_filtered)} rows remain.")
            else:
                 logger.warning(f"[{filename}] No valid STOP marker found after START. Using data from START onwards.")
                 df_filtered = df.loc[start_idx:].copy()
        else:
            logger.warning(f"[{filename}] No START marker found. Using all rows after time cleaning.")
            df_filtered = df.copy() # Use the time-cleaned df
    else:
        logger.warning(f"[{filename}] No '{MARKER_COL}', skipping START/STOP filter.")
        df_filtered = df.copy() # Use the time-cleaned df

    if df_filtered.empty:
        logger.error(f"[{filename}] DataFrame empty after START/STOP filtering stage."); return None

    # --- Time Normalization (relative to filtered data) ---
    if TIME_COL_SECONDS in df_filtered.columns and not df_filtered.empty:
        first_valid_time_index = df_filtered[TIME_COL_SECONDS].first_valid_index()
        if first_valid_time_index is not None:
            start_time_norm = df_filtered.loc[first_valid_time_index, TIME_COL_SECONDS]
            if pd.notna(start_time_norm):
                logger.info(f"[{filename}] Normalizing time column, starting from {start_time_norm:.2f}s.")
                df_filtered[TIME_COL_SECONDS] = df_filtered[TIME_COL_SECONDS] - start_time_norm
                # Ensure no negative times due to precision issues
                df_filtered.loc[df_filtered[TIME_COL_SECONDS] < 0, TIME_COL_SECONDS] = 0
            else: logger.warning(f"[{filename}] First valid time is NaN, cannot normalize time.")
        else: logger.warning(f"[{filename}] No valid time found in filtered data, cannot normalize time.")

    # --- Final Numeric Conversion Check ---
    logger.info(f"[{filename}] Checking/Converting columns to numeric...")
    cols_converted = []; cols_failed = []
    for col in df_filtered.columns:
        # Skip already numeric types, time/marker/watt cols
        if pd.api.types.is_numeric_dtype(df_filtered[col]): continue
        if col in [RAW_TIME_COL, MARKER_COL, WATT_COL, TIME_COL_SECONDS]: continue

        original_dtype = str(df_filtered[col].dtype)
        # Try direct conversion first
        converted_col = pd.to_numeric(df_filtered[col], errors='coerce')

        # If direct conversion results in all NaNs, maybe it had commas as decimal separators?
        if converted_col.isnull().all() and not df_filtered[col].isnull().all():
             logger.debug(f"[{filename}] Direct numeric conversion failed for '{col}', trying comma replacement...")
             try:
                 # Ensure it's string type before replacing
                 if pd.api.types.is_object_dtype(df_filtered[col]):
                      converted_col = pd.to_numeric(df_filtered[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
                 else: # If not object, comma replace likely won't help
                      pass
             except Exception as e_replace:
                 logger.warning(f"[{filename}] Error during comma replacement for '{col}': {e_replace}")

        # Check if conversion was successful (at least some non-NaN values)
        if not converted_col.isnull().all():
            df_filtered[col] = converted_col
            if str(df_filtered[col].dtype) != original_dtype:
                 cols_converted.append(col)
                 logger.debug(f"[{filename}] Converted column '{col}' to numeric.")
        else:
            cols_failed.append(col)
            logger.debug(f"[{filename}] Failed to convert column '{col}' to numeric, keeping as {original_dtype}.")

    if cols_converted: logger.info(f"[{filename}] Columns converted to numeric: {cols_converted}")
    if cols_failed: logger.warning(f"[{filename}] Columns failed numeric conversion: {cols_failed}")

    # --- Final Cleanup and Return ---
    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered.attrs['filename'] = filename # Keep filename attribute
    final_rows = len(df_filtered)
    rows_dropped_total = initial_rows - final_rows
    logger.info(f"[{filename}] Data preparation finished. Final shape: {df_filtered.shape}. Total rows dropped: {rows_dropped_total}")
    return df_filtered

# --- NO @st.cache_data ---
def apply_smoothing(df_prepared, method, time_col_sec):
    """Applies selected smoothing to the prepared data DataFrame."""
    filename = df_prepared.attrs.get('filename', 'N/A')
    logger.info(f"[{filename}] Applying smoothing: {method} (Cache DISABLED)")

    if df_prepared is None or df_prepared.empty:
        logger.warning(f"[{filename}] Input DataFrame for smoothing is empty or None.")
        return df_prepared # Return empty/None as is
    if not isinstance(df_prepared, pd.DataFrame):
        logger.error(f"[{filename}] Invalid input type for smoothing: {type(df_prepared)}")
        return None

    if method == "Raw Data":
        logger.debug(f"[{filename}] Smoothing method is 'Raw Data', returning copy.")
        return df_prepared.copy()

    # Identify numeric columns suitable for smoothing
    numeric_cols = df_prepared.select_dtypes(include=np.number).columns.tolist()
    # Exclude time column and potentially other non-signal columns like ID if they exist
    cols_to_exclude = [time_col_sec, 'subject_id', 'ID', 'index'] # Add known non-signal numeric cols
    cols_to_smooth = [col for col in numeric_cols if col not in cols_to_exclude]

    if not cols_to_smooth:
        logger.warning(f"[{filename}] No numeric columns found to apply smoothing '{method}'.")
        return df_prepared.copy()

    # Keep non-numeric and excluded columns as they are
    cols_to_keep = df_prepared.columns.difference(cols_to_smooth).tolist()
    df_smoothed = df_prepared[cols_to_keep].copy()

    logger.debug(f"[{filename}] Columns to smooth: {cols_to_smooth}")
    logger.debug(f"[{filename}] Columns to keep raw: {cols_to_keep}")

    try:
        if "Breath" in method:
            match = re.search(r'(\d+)\s*Breath', method)
            if not match: raise ValueError(f"Cannot parse breath window: {method}")
            window_size = int(match.group(1))
            if window_size <= 0: raise ValueError("Breath window size must be positive.")
            logger.debug(f"[{filename}] Applying {window_size}-breath rolling mean.")
            smoothed_data = df_prepared[cols_to_smooth].rolling(window=window_size, min_periods=1).mean()
            df_smoothed[cols_to_smooth] = smoothed_data

        elif "Sec" in method:
            if time_col_sec not in df_prepared.columns:
                raise ValueError(f"Time column '{time_col_sec}' needed for time-based smoothing is missing.")
            if not pd.api.types.is_numeric_dtype(df_prepared[time_col_sec]):
                raise ValueError(f"Time column '{time_col_sec}' must be numeric for time-based smoothing.")

            match = re.search(r'(\d+)\s*Sec', method)
            if not match: raise ValueError(f"Cannot parse second window: {method}")
            seconds = int(match.group(1))
            if seconds <= 0: raise ValueError("Time window must be positive.")
            time_window_str = f"{seconds}s"
            logger.debug(f"[{filename}] Applying {time_window_str} time rolling mean based on '{time_col_sec}'.")

            # Ensure data is sorted by time for time-based rolling
            df_temp_time = df_prepared[[time_col_sec] + cols_to_smooth].sort_values(by=time_col_sec).copy()
            # Create a TimedeltaIndex for rolling
            df_temp_time.index = pd.to_timedelta(df_temp_time[time_col_sec], unit='s')

            smoothed_data = df_temp_time[cols_to_smooth].rolling(window=time_window_str, min_periods=1).mean()
            # Merge back using the original DataFrame's index
            df_smoothed[cols_to_smooth] = smoothed_data.reindex(df_prepared.index) # Reindex needed after sorting

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        # Restore original column order if possible
        try:
            df_smoothed = df_smoothed[df_prepared.columns]
        except KeyError:
            logger.warning(f"[{filename}] Could not restore original column order after smoothing.")
            pass # Keep the current order

        df_smoothed.attrs['filename'] = filename # Preserve attribute
        logger.info(f"[{filename}] Smoothing '{method}' applied successfully.")
        return df_smoothed

    except Exception as e:
        logger.error(f"[{filename}] Error applying smoothing '{method}': {e}", exc_info=True)
        # Return the original prepared data if smoothing fails
        return df_prepared.copy()


def calculate_slope(p1, p2):
    """Calculates slope between two points (tuples), handles vertical lines."""
    if p1 is None or p2 is None: logger.warning("Slope calculation received None point."); return 0
    if not (isinstance(p1, (tuple, list)) and len(p1) == 2 and isinstance(p2, (tuple, list)) and len(p2) == 2):
         logger.warning(f"Invalid point format for slope calculation: p1={p1}, p2={p2}"); return 0

    try:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
    except (ValueError, TypeError) as e:
         logger.warning(f"Non-numeric point coordinates for slope: p1={p1}, p2={p2}, Error: {e}"); return 0

    delta_x = x2 - x1
    delta_y = y2 - y1

    if abs(delta_x) < 1e-9: # Threshold to consider vertical
        if abs(delta_y) < 1e-9: return 0 # Points are identical
        return np.inf if delta_y > 0 else -np.inf # Vertical line
    return delta_y / delta_x