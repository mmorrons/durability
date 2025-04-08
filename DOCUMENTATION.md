# Code Documentation - Streamlit Physiology Segment Analyzer

**Version:** 1.0 (As of April 8, 2025)

This document provides an overview of the codebase structure and the functions used in the application.

## Project Structure

The application is primarily composed of two Python files:

1.  **`data_processing.py`:** Contains the core backend logic for handling data. It includes functions for parsing filenames, loading data from files, cleaning and preparing the data (time conversion, unit row removal, Watt derivation, START/STOP filtering), applying smoothing algorithms, and calculating slopes. It is designed to be mostly independent of the user interface.
2.  **`streamlit_app.py`:** Contains the user interface logic built using the Streamlit framework. It handles file uploads, user selections (dataset, smoothing, plot axes), orchestrates calls to functions in `data_processing.py`, manages application state using `st.session_state`, displays interactive Plotly charts using `st.plotly_chart`, processes plot selection events to assist manual input, handles manual coordinate input, and displays results.

## `data_processing.py` Details

### Constants

* `RAW_TIME_COL` (str): Name of the raw time column in input files (e.g., 't').
* `TIME_COL_SECONDS` (str): Name of the column created to store time in total seconds (e.g., 'time_seconds').
* `MARKER_COL` (str): Name of the column containing markers like "START", "STOP", and Watt values (e.g., 'Marker').
* `EXPECTED_HEADER_ROW_INDEX` (int): The 0-based index corresponding to the header row in the input Excel files (e.g., 142 for row 143).
* `WATT_COL` (str): Name of the derived Watt column to be created (e.g., 'W').

### Functions

* **`parse_filename(filename)`**
    * **Input:** `filename` (str) - The base filename.
    * **Output:** `(metadata, unique_key)` (tuple) - `metadata` is a dictionary containing parsed info (subject ID, name, test code, etc.) or `None` on failure. `unique_key` is a generated key string or the original filename on failure.
    * **Purpose:** Extracts structured information based on the `PREFIXID_Surname_Name_TestCode` naming convention.

* **`time_str_to_seconds(time_str)`**
    * **Input:** `time_str` (str | None) - Time string in "H:MM:SS,ms" or "MM:SS,ms" format.
    * **Output:** `float | None` - Time converted to total seconds, or `None` if parsing fails.
    * **Purpose:** Converts standard time formats from the data file into a numeric representation. Handles missing hours.

* **`load_data(file_content, filename)`**
    * **Input:** `file_content` (BytesIO stream), `filename` (str).
    * **Output:** `pandas.DataFrame | None` - Loaded DataFrame or `None` on failure.
    * **Purpose:** Reads data from an in-memory file stream (`BytesIO`). Handles both CSV and Excel formats. For Excel, it assumes the header is at `EXPECTED_HEADER_ROW_INDEX`. Stores `filename` in `df.attrs`.

* **`prepare_data(_df_raw)`**
    * **Input:** `_df_raw` (pandas.DataFrame) - Raw DataFrame from `load_data`.
    * **Output:** `pandas.DataFrame | None` - Processed and filtered DataFrame, or `None` if processing fails critically.
    * **Purpose:** Orchestrates the main data cleaning and preparation pipeline:
        1.  Converts time column using `time_str_to_seconds`.
        2.  Removes rows with invalid time formats (often the unit row). Sorts by time.
        3.  Derives the `WATT_COL` ('W') by parsing numeric values (like "90W") from `MARKER_COL` and forward-filling. Handles setting Watt=0 before the 'START' marker.
        4.  Filters the DataFrame to include only rows between the first 'START' marker and the first subsequent 'STOP' marker (if found). Handles missing START or STOP.
        5.  Normalizes the `TIME_COL_SECONDS` column within the filtered data to start from 0.
        6.  Attempts to convert potentially numeric columns (that might be loaded as objects) into actual numeric types.
        7.  Resets the DataFrame index.

* **`apply_smoothing(df_prepared, method, time_col_sec)`**
    * **Input:** `df_prepared` (pandas.DataFrame), `method` (str - e.g., "Raw Data", "15 Breaths MA", "10 Sec MA"), `time_col_sec` (str - name of the numeric time column).
    * **Output:** `pandas.DataFrame` - Smoothed DataFrame, or the original `df_prepared` if method is "Raw Data" or smoothing fails.
    * **Purpose:** Applies either a rolling window average based on row count ("Breath") or a time-based rolling window average ("Sec") using pandas `.rolling()` methods. Includes the fix to correctly parse "Sec MA" options. Relies on Streamlit's `@st.cache_data` for caching results.

* **`calculate_slope(p1, p2)`**
    * **Input:** `p1`, `p2` (tuple) - Each is an `(x, y)` tuple representing a point.
    * **Output:** `float` - Calculated slope (m). Handles vertical lines (returns +/- `np.inf`).
    * **Purpose:** Simple calculation of the slope between two points.

## `streamlit_app.py` Details

### State Management (`st.session_state`)

* `processed_data`: Dictionary storing the main loaded and prepared data for each dataset (key: `unique_key`, value: `{'metadata': ..., 'prepared_df': ...}`).
* `plot_configs`: Dictionary storing the selected X/Y axes for each plot display area (key: `plot_id`, value: `{'x_col': ..., 'y_col': ...}`).
* `plot_analysis_states`: Dictionary storing the segment analysis state for each plot display area (key: `plot_id`, value: `{'segments': [...]}`). The `p1` state for plot selection is now implicitly handled via the manual input fields.
* `current_selected_key`: Stores the unique key of the currently active dataset.
* `current_smoothing`: Stores the currently selected smoothing method string.
* `current_x_col`, `current_y_col`: Stores the axes selected for the (currently single) main plot area configuration. *Note: This was simplified in the 3-column layout version*. Re-check implementation - config is now stored per-plot in `plot_configs`.
* `manual_x1`, `manual_y1`, `manual_x2`, `manual_y2`: Store the values currently entered in the manual coordinate input fields.
* `selection_target`: Tracks whether the next plot selection should populate 'P1' or 'P2' fields.
* `last_select_event_data`: Stores the data from the last processed plot selection event to prevent duplicate processing on reruns.
* `files_processed_flag`: A simple flag to avoid reprocessing files on every Streamlit rerun if the uploaded files haven't changed.

### UI and Workflow

1.  **File Upload:** `st.file_uploader` allows multiple file uploads.
2.  **Processing:** When new files are uploaded, they are processed using functions from `data_processing.py`, and the results are stored in `st.session_state.processed_data`.
3.  **Layout:** Uses `st.columns` to create a 3-column layout (Plot | Manual Input | Configuration).
4.  **Configuration (Right Column):**
    * Dataset selection (`st.selectbox`) updates `current_selected_key` and resets analysis state.
    * Smoothing selection (`st.selectbox`) updates `current_smoothing`.
    * Plot axis selection (`st.selectbox` for X and Y for each plot configured) updates `plot_configs`. Data for dropdowns comes from the smoothed `df_display`.
5.  **Manual Input (Middle Column):**
    * Displays target for next plot selection (`selection_target`).
    * Shows `st.number_input` fields bound to `manual_x1` etc. session state keys.
    * "Add Manual Segment" button reads values from state, validates, calculates slope (`dp.calculate_slope`), adds segment to `st.session_state.segments`, clears manual fields (optional), resets target, and triggers rerun.
    * "Clear Manual Inputs" button clears the number input state.
    * "Reset All Segments" button clears the `segments` list state.
6.  **Plot Display (Left Column):**
    * The `create_plot` helper function generates a Plotly `go.Figure`.
    * `st.plotly_chart` displays the figure and is configured to return selection events (`on_select="rerun"` - *Note: This part was adapted, confirm current implementation details*).
7.  **Plot Selection->Manual Transfer Logic (After `st.plotly_chart`):**
    * Checks the return value (`event_data`) of `st.plotly_chart`.
    * If a new selection event with points is detected:
        * Extracts coordinates `(x_sel, y_sel)` from the first point.
        * Updates the corresponding `manual_x1/y1` or `manual_x2/y2` values in `st.session_state` based on the `selection_target`.
        * Updates the `selection_target` state ('P1' -> 'P2' -> 'P1').
        * Stores the event data in `last_select_event_data` to prevent reprocessing.
        * Triggers `st.rerun()` to update the displayed values in the `st.number_input` fields.
8.  **Segment Table:** A `st.dataframe` at the bottom displays the details of all segments currently stored in `st.session_state.segments`.

## Running the App

Ensure dependencies are installed (`requirements.txt`). Run using:
```bash
streamlit run streamlit_app.py