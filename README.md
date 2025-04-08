# Streamlit Physiology Segment Analyzer

**Version:** 1.0 (As of April 8, 2025)
**Author:** [mmorrons]

## Description

This is a web application built with Streamlit for visualizing and analyzing physiological data from exercise tests (e.g., CPET). It allows users to upload raw data files (specifically formatted `.xlsx` or `.csv`), process them, apply smoothing, display the data in interactive plots, and define linear segments on the plots to calculate slopes (e.g., for V-Slope analysis).

The primary goal is to provide an interactive tool for identifying key points or phases within exercise data by analyzing the slope between user-defined points on various graphs.

## Features

* **Multi-File Upload:** Handles uploading multiple `.xlsx` or `.csv` files simultaneously.
* **Metadata Parsing:** Extracts subject ID, name, test type, etc., from filenames following the `PREFIXID_Surname_Name_TestCode` convention (e.g., `DUR001_Cappai_Antonello_C1.xlsx`).
* **Data Processing Pipeline (`data_processing.py`):**
    * Loads data assuming a fixed header row (Row 143 / index 142) for Excel files.
    * Handles and removes the typical "unit" row below the header.
    * Converts time strings (H:MM:SS,ms or MM:SS,ms) to total seconds.
    * Derives a `W` (Watt) column from numerical markers (e.g., "90W") found in the 'Marker' column, using forward-fill.
    * Filters data between the first "START" and subsequent "STOP" markers found in the 'Marker' column.
    * Normalizes the time axis to start from 0s after the "START" marker.
    * Attempts robust conversion of relevant columns to numeric types.
* **Data Smoothing:** Applies moving average smoothing based on breath counts (e.g., "15 Breaths MA") or time duration (e.g., "10 Sec MA").
* **Multi-Plot Display:** Shows multiple plots (currently configured for 2) in the main area.
* **Configurable Plots:** Users can select the dataset, smoothing method, and X/Y variables for each plot via controls in the right-hand panel.
* **Interactive Segment Definition:**
    * **Primary Method (Plot Selection Assist):** Use Plotly's Box Select or Lasso Select tool on a plot to draw a small shape around a single data point. The coordinates of that point are automatically transferred to the "Manual Segment Input" fields (P1 first, then P2).
    * **Secondary Method (Manual Input):** Directly type the X and Y coordinates for the start (P1) and end (P2) points of a segment into the input fields.
    * Click "Add Manual Segment" to calculate the slope and add the segment to the selected target plot and the results table.
* **Segment Visualization:** Defined segments are drawn as red lines on the corresponding plot.
* **Results Display:** A table summarizes the defined segments for all plots, showing start/end coordinates and calculated slopes.
* **Reset Options:** Buttons allow clearing manual inputs or resetting all defined segments for a specific plot.

## File Structure

* **`streamlit_app.py`:** The main Streamlit application script. Handles the UI layout, user interactions, state management, and calls functions from `data_processing`.
* **`data_processing.py`:** A module containing the core logic for file parsing, data loading, cleaning, processing (Watt derivation, filtering), smoothing, and slope calculation. Designed to be relatively independent of the Streamlit UI.
* **`requirements.txt`:** Lists the necessary Python libraries and their versions for deployment.
