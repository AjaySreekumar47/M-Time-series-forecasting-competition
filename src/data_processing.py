"""
Data processing utilities for M1 Competition time series.
This module contains functions for loading and preprocessing M1 Competition data.
"""

import pandas as pd
import numpy as np


def load_m1_data(file_path):
    """
    Load M1 competition data from Excel file
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing M1 competition data
        
    Returns:
    --------
    tuple
        (data_df, seas_df) - DataFrames containing main data and seasonality indicators
    """
    try:
        # Load main data sheet
        data_df = pd.read_excel(file_path, sheet_name=0)  # First sheet
        
        # Load seasonality indicators
        seas_df = pd.read_excel(file_path, sheet_name=1)  # Second sheet
        
        return data_df, seas_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def find_matching_series(data_df, target_series_id):
    """
    Find a series that matches the target ID, handling trailing spaces
    
    Parameters:
    -----------
    data_df : DataFrame
        DataFrame containing series data
    target_series_id : str
        ID of the series to find
        
    Returns:
    --------
    str or None
        The actual series ID if found, None otherwise
    """
    # Try exact match first
    if target_series_id in data_df['Series'].values:
        return target_series_id
    
    # Try stripped match
    stripped_id = target_series_id.strip()
    for series in data_df['Series']:
        if series.strip() == stripped_id:
            return series
    
    return None  # No match found


def prepare_time_series(data_df, series_id):
    """
    Prepare a single time series for analysis
    
    Parameters:
    -----------
    data_df : DataFrame
        DataFrame containing series data
    series_id : str
        ID of the series to prepare
        
    Returns:
    --------
    tuple
        (ts_df, metadata) - DataFrame containing time series and dictionary with metadata
    """
    # Handle trailing spaces in series IDs
    # First try direct match
    series_row = data_df[data_df['Series'] == series_id]
    
    # If not found, try with trimming spaces
    if len(series_row) == 0:
        # Look for series with spaces trimmed
        series_row = data_df[data_df['Series'].str.strip() == series_id.strip()]
        
        if len(series_row) == 0:
            # Print all series IDs for debugging
            print(f"Could not find series '{series_id}' in dataset")
            print("Available series (first 10):", data_df['Series'].head(10).tolist())
            raise ValueError(f"Series {series_id} not found in the dataset")
    
    series_row = series_row.iloc[0]
    
    # Get metadata
    start_year = series_row['Starting date']
    n_obs = int(series_row['N Obs'])
    category = series_row['Category']
    
    # Extract the time series values
    # Create a list of numeric column names
    numeric_cols = [col for col in data_df.columns if isinstance(col, int) or 
                   (isinstance(col, str) and col.isdigit())]
    
    # If no numeric columns found, use position-based access
    if not numeric_cols:
        # Find columns that might contain the time series data
        # Start after the last metadata column (typically 'Category')
        potential_data_cols = list(range(7, 7+n_obs))
        values = []
        for i in potential_data_cols:
            if i < len(series_row):
                values.append(series_row.iloc[i])
            else:
                break
    else:
        # Use only the first n_obs numeric columns
        numeric_cols = sorted(numeric_cols)[:n_obs]
        values = series_row[numeric_cols].values
    
    # Ensure values are numeric and handle any potential strings or NaN values
    clean_values = []
    for val in values:
        try:
            clean_val = float(val)
            if not np.isnan(clean_val):
                clean_values.append(clean_val)
        except (ValueError, TypeError):
            print(f"Skipping non-numeric value: {val}")
            
    # Create a time series with proper dates
    years = pd.date_range(start=f"{start_year}-01-01", periods=len(clean_values), freq='YS')
    ts_df = pd.DataFrame({'date': years, 'value': clean_values})
    ts_df.set_index('date', inplace=True)
    
    # Convert to numeric to ensure no object dtype
    ts_df['value'] = pd.to_numeric(ts_df['value'], errors='coerce')
    
    # Drop any remaining NaN values
    ts_df = ts_df.dropna()
    
    return ts_df, {'category': category, 'start_year': start_year, 'n_obs': len(clean_values)}


def train_test_split(ts_df, test_size=0.2):
    """
    Split time series into training and test sets
    
    Parameters:
    -----------
    ts_df : DataFrame
        DataFrame containing time series data
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (train, test) - DataFrames containing training and test data
    """
    n = len(ts_df)
    train_size = int(n * (1 - test_size))
    
    train = ts_df.iloc[:train_size]
    test = ts_df.iloc[train_size:]
    
    return train, test


def check_seasonality(seas_df, series_id):
    """
    Check if a series has seasonality based on seasonality indicators
    
    Parameters:
    -----------
    seas_df : DataFrame
        DataFrame containing seasonality indicators
    series_id : str
        ID of the series to check
        
    Returns:
    --------
    tuple
        (has_seasonality, period) - Boolean indicating seasonality and the seasonal period
    """
    # Handle trailing spaces in series IDs
    series_row = seas_df[seas_df['Series'] == series_id]
    
    if len(series_row) == 0:
        # Try with trimming spaces
        series_row = seas_df[seas_df['Series'].str.strip() == series_id.strip()]
        if len(series_row) == 0:
            return False, 1
    
    series_row = series_row.iloc[0]
    
    # Get the numeric columns from the seasonality sheet
    num_cols = [col for col in seas_df.columns if isinstance(col, int) or 
               (isinstance(col, str) and col.isdigit())]
    
    if not num_cols:
        # Assume columns start after 'Seasonality' if no numeric columns found
        seas_col_idx = list(seas_df.columns).index('Seasonality') + 1
        indicators = series_row.iloc[seas_col_idx:seas_col_idx+12].values
    else:
        indicators = series_row[sorted(num_cols)[:12]].values
    
    # Convert to numeric
    indicators = pd.to_numeric(indicators, errors='coerce').fillna(0).astype(int)
    
    # Check if it has seasonality
    if np.sum(indicators) > 1:  # More than one period has indicator value
        period = len(indicators)
        return True, period
    else:
        return False, 1
