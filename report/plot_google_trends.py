import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path

# --- Configuration ---
# Directory to scan for CSV files
TRENDS_SUBFOLDER = "trends"
# Date to filter data from. Set to a string like "YYYY-MM-DD" or None to disable.
FILTER_START_DATE = "2020-01-01"

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid') # A good base for a 'research-y' look

# Attempt to set CMU Serif font
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'CMU Serif'
    print("Attempting to set font to CMU Serif. Ensure it's installed on your system.")
except Exception as e:
    print(f"Could not set CMU Serif font, ensure it is installed. Falling back to default serif. Error: {e}")


line_colors = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#F0E442', '#56B4E9'] # Colorblind-friendly palette
font_size_axis_labels = 11
font_size_title = 13
font_size_ticks = 9
font_size_legend = 8.5 # Slightly adjusted for internal legend
line_width = 1.0
figure_dpi = 120

# --- Dynamic CSV File Discovery ---
csv_files = {}
trends_path = Path(TRENDS_SUBFOLDER)

print(f"--- Scanning for CSV files in '{trends_path.resolve()}' ---")

if not trends_path.is_dir():
    print(f"Error: Subfolder '{TRENDS_SUBFOLDER}' not found.")
    print(f"Please create a subfolder named '{TRENDS_SUBFOLDER}' in the same directory as this script,")
    print("and place your CSV files inside it.")
else:
    for file_path in trends_path.glob('*.csv'):
        label = file_path.stem
        csv_files[label] = str(file_path)

if not csv_files:
    if trends_path.is_dir():
        print(f"No CSV files found in the '{TRENDS_SUBFOLDER}' subfolder.")
else:
    print(f"Found {len(csv_files)} CSV files to process:")
    for label, path_str in csv_files.items():
        print(f"  - Label: '{label}', File: '{path_str}'")

# --- Data Loading and Processing ---
data_to_plot = {}
print("\n--- Starting Data Processing ---")

for label, file_path_str in csv_files.items():
    file_path = Path(file_path_str)
    print(f"Processing: {file_path} for label: '{label}'")
    try:
        skiprows = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if not lines:
                print(f"Warning: File {file_path} is empty. Skipping.")
                continue

            for i, line in enumerate(lines):
                line_lower_stripped = line.strip().lower()
                if line_lower_stripped.startswith(("week,", "day,", "month,")):
                    is_actual_data_header = True
                    if i > 0 and "category:" in lines[i-1].strip().lower():
                         is_actual_data_header = True
                    elif i > 1 and "category:" in lines[i-2].strip().lower():
                         is_actual_data_header = True
                    if is_actual_data_header:
                        skiprows = i
                        break
                if any(phrase in line_lower_stripped for phrase in ["category: all categories", "interesse im zeitverlauf", "search interest", "geomap data"]):
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if lines[j].strip().lower().startswith(("week,", "day,", "month,")):
                            skiprows = j
                            break
                    else:
                        skiprows = i + 1
                    break
                if i >= 10:
                    print(f"Warning: Could not reliably auto-detect data header for {file_path} within the first 10 lines. Defaulting to skiprows=0.")
                    skiprows = 0
                    break
            if not lines: skiprows = 0
        except Exception as e_skip:
            print(f"Warning: Error during skiprows detection for {file_path}: {e_skip}. Defaulting to skiprows=0.")
            skiprows = 0

        print(f"Attempting to read {file_path} with skiprows={skiprows}")
        df = pd.read_csv(file_path, skiprows=skiprows)

        if df.empty and skiprows < 5 :
            print(f"Warning: DataFrame is empty after reading with skiprows={skiprows}. Trying with skiprows=2 for {file_path} as a fallback for Google Trends format.")
            try:
                df_fallback = pd.read_csv(file_path, skiprows=2)
                if not df_fallback.empty:
                    df = df_fallback
                    skiprows = 2
                    print("Fallback to skiprows=2 successful.")
                else:
                    print("Fallback to skiprows=2 also resulted in an empty DataFrame.")
            except Exception as e_fallback:
                print(f"Error during fallback read with skiprows=2 for {file_path}: {e_fallback}")

        if df.empty:
            print(f"Error: DataFrame is empty after attempting to read {file_path}. Check CSV content and skiprows. Skipping file.")
            continue

        date_col = None
        for potential_col_name in ['Week', 'Day', 'Month', 'Date', 'week', 'day', 'month', 'date']:
            if potential_col_name in df.columns:
                date_col = potential_col_name
                break
        
        if not date_col:
            if len(df.columns) > 0:
                potential_date_col = df.columns[0]
                try:
                    pd.to_datetime(df[potential_date_col].head(), errors='raise')
                    date_col = potential_date_col
                    print(f"Info: Assuming first column '{date_col}' is the date column for {file_path}.")
                except Exception:
                     print(f"Error: Could not find a suitable date column or parse first column as date in {file_path}.")
                     continue
            else:
                print(f"Error: No columns found in {file_path}.")
                continue
        
        data_col_name_in_csv = None
        for col in df.columns:
            if col.lower() != date_col.lower():
                temp_series_str = df[col].astype(str)
                if temp_series_str.str.contains('<1').any():
                    temp_series_numeric = pd.to_numeric(temp_series_str.str.replace('<1', '0.5', regex=False), errors='coerce')
                else:
                    temp_series_numeric = pd.to_numeric(df[col], errors='coerce')
                if temp_series_numeric.notna().any():
                    data_col_name_in_csv = col
                    break
        
        if not data_col_name_in_csv:
            if len(df.columns) > 1:
                potential_data_cols = [c for c in df.columns if c.lower() != date_col.lower()]
                if potential_data_cols:
                    data_col_name_in_csv = potential_data_cols[0]
                    print(f"Warning: Could not definitively identify a numeric data column for {file_path}. Assuming first non-date column: '{data_col_name_in_csv}'.")
                else:
                    print(f"Error: Only one column found or no other columns available in {file_path}.")
                    continue
            else:
                print(f"Error: Not enough columns in {file_path} to identify data column after '{date_col}'.")
                continue

        print(f"Identified date column: '{date_col}', data column: '{data_col_name_in_csv}' for {file_path}")

        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e_date:
            print(f"Error converting date column '{date_col}' to datetime for {file_path}: {e_date}. Skipping file.")
            continue

        if df[data_col_name_in_csv].dtype == 'object' or pd.api.types.is_string_dtype(df[data_col_name_in_csv]):
            df[data_col_name_in_csv] = df[data_col_name_in_csv].astype(str).str.replace('<1', '0.5', regex=False)
        
        df[data_col_name_in_csv] = pd.to_numeric(df[data_col_name_in_csv], errors='coerce')
        df = df.dropna(subset=[date_col, data_col_name_in_csv])

        if df.empty:
            print(f"Warning: No valid data rows found in {file_path} for column '{data_col_name_in_csv}' before date filtering. Skipping.")
            continue
            
        # Rename columns before potential filtering
        processed_df = df[[date_col, data_col_name_in_csv]].copy()
        processed_df.rename(columns={data_col_name_in_csv: 'Value', date_col: 'Date'}, inplace=True)

        # Apply date filtering if configured
        if FILTER_START_DATE:
            try:
                filter_date_obj = pd.to_datetime(FILTER_START_DATE)
                processed_df = processed_df[processed_df['Date'] >= filter_date_obj]
                if processed_df.empty:
                    print(f"Warning: No data remains for label '{label}' after filtering from date {FILTER_START_DATE}. Skipping this series.")
                    continue # Skip to next file if this one became empty after filtering
                else:
                    print(f"Applied date filter: showing data from {FILTER_START_DATE} for label '{label}'.")
            except ValueError:
                print(f"Warning: Invalid FILTER_START_DATE format: '{FILTER_START_DATE}'. Disabling date filter for this run. Please use YYYY-MM-DD.")
                # Potentially disable filter for all subsequent files too, or just this one. For now, just prints warning.
            except Exception as e_filter:
                print(f"Warning: Error during date filtering for label '{label}': {e_filter}. Proceeding without filter for this series.")
        
        if processed_df.empty: # Double check after potential filtering
             print(f"Warning: Data for label '{label}' became empty after processing/filtering. Skipping.")
             continue

        data_to_plot[label] = processed_df
        print(f"Successfully processed and stored data from {file_path} for label '{label}'.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Skipping.")
    except pd.errors.EmptyDataError:
        print(f"Error: File {file_path} is empty or has no data after skipping rows. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}. Skipping.")

# --- Plotting ---
if not data_to_plot:
    print("\nNo data successfully processed to plot. Exiting.")
else:
    print("\n--- Starting Plotting ---")
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=figure_dpi)
    
    color_idx = 0
    plotted_series_count = 0
    for label, df_plot in data_to_plot.items():
        if 'Date' not in df_plot.columns or 'Value' not in df_plot.columns:
            print(f"Warning: Standardized 'Date' or 'Value' column not found for label '{label}'. Skipping this series.")
            continue
        if df_plot.empty:
            print(f"Warning: Data for label '{label}' is empty before plotting. Skipping this series.")
            continue

        ax.plot(df_plot['Date'], df_plot['Value'],
                label=label,
                color=line_colors[color_idx % len(line_colors)],
                linewidth=line_width)
        color_idx += 1
        plotted_series_count +=1
        print(f"Plotting series: {label}")

    if plotted_series_count == 0:
        print("No series were actually plotted. Check data processing steps and warnings.")
        ax.set_title("No Data to Display", fontsize=font_size_title, fontweight='bold', pad=15)
    else:
        ax.set_title("Search interest over time, Google Trends", fontsize=font_size_title, fontweight='bold', pad=15)
    
    ax.set_xlabel("Date", fontsize=font_size_axis_labels, labelpad=10)
    ax.set_ylabel("Relative Search Interest", fontsize=font_size_axis_labels, labelpad=10)

    ax.tick_params(axis='x', labelsize=font_size_ticks)
    ax.tick_params(axis='y', labelsize=font_size_ticks)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=30, ha='right')

    if plotted_series_count > 0 : # Show legend if any series were plotted
        # Legend inside the plot at top-left
        legend = ax.legend(fontsize=font_size_legend, frameon=True, loc='upper left')
        if legend: 
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_alpha(0.8) # Make legend slightly transparent if overlapping data
    
    if plotted_series_count == 1 and list(data_to_plot.keys()): # Check if keys exist before accessing
        single_label = list(data_to_plot.keys())[0]
        ax.set_title(f"Time Series: {single_label}", fontsize=font_size_title, fontweight='bold', pad=15)

    plt.tight_layout() # Apply tight_layout after all elements are added

    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.6)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    # Save the plot (optional, uncomment to use)
    output_filename_base = "trends"
    if plotted_series_count > 0:
       plt.savefig(f"{output_filename_base}.png", dpi=figure_dpi + 50, bbox_inches='tight')
       plt.savefig(f"{output_filename_base}.pdf", bbox_inches='tight')
       print(f"Plot saved as {output_filename_base}.png and {output_filename_base}.pdf")

    # plt.show()

print("\n--- Script Finished ---")
if not csv_files and trends_path.is_dir():
     print(f"No CSV files were found in the '{TRENDS_SUBFOLDER}' subfolder.")
elif not trends_path.is_dir() and not csv_files :
     print(f"The '{TRENDS_SUBFOLDER}' subfolder does not exist. Please create it and add your CSV files.")

if data_to_plot or (not csv_files and trends_path.is_dir()) or (not trends_path.is_dir()):
    print("\nInstructions for use:")
    print(f"1. Ensure this script is in a directory.")
    print(f"2. Create a subfolder named '{TRENDS_SUBFOLDER}' in that same directory.")
    print(f"3. Place all your CSV files (e.g., from Google Trends) into the '{TRENDS_SUBFOLDER}' subfolder.")
    print(f"4. The script will automatically use the CSV filenames (without '.csv') as labels in the plot.")
    print(f"5. To filter data by date, modify the 'FILTER_START_DATE' variable at the top of the script.")
    print(f"   Set it to a date string like 'YYYY-MM-DD' or to 'None' to disable filtering (current: '{FILTER_START_DATE}').")
    print(f"6. Run the script. It will process all CSVs in '{TRENDS_SUBFOLDER}' and generate a combined plot.")
    print(f"7. For 'CMU Serif' font, ensure it is installed on your system.")
    print(f"8. Review console output for any warnings or errors if data doesn't look as expected.")

