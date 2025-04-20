"""
Core forecasting functionality for M1 Competition time series.
This module contains the main functions for analyzing and forecasting time series data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Function to load the data from Excel file
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
        
        # Print column names for debugging
        print("Data columns:", data_df.columns.tolist())
        print("First few rows of data:")
        print(data_df.head())
        
        return data_df, seas_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Helper function to find matching series
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

# Function to format data for time series analysis
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
            
    # Print info about the extracted values
    print(f"Extracted {len(clean_values)} numeric values out of {n_obs} expected observations")
    
    # Create a time series with proper dates
    years = pd.date_range(start=f"{start_year}-01-01", periods=len(clean_values), freq='YS')
    ts_df = pd.DataFrame({'date': years, 'value': clean_values})
    ts_df.set_index('date', inplace=True)
    
    # Convert to numeric to ensure no object dtype
    ts_df['value'] = pd.to_numeric(ts_df['value'], errors='coerce')
    
    # Drop any remaining NaN values
    ts_df = ts_df.dropna()
    
    print(f"Final time series shape: {ts_df.shape}, dtype: {ts_df['value'].dtype}")
    
    return ts_df, {'category': category, 'start_year': start_year, 'n_obs': len(clean_values)}

# Function for exploratory analysis
def explore_time_series(ts_df, series_id, metadata):
    """
    Perform exploratory analysis on a time series
    
    Parameters:
    -----------
    ts_df : DataFrame
        DataFrame containing time series data
    series_id : str
        ID of the series being analyzed
    metadata : dict
        Dictionary with series metadata
        
    Returns:
    --------
    Series
        Descriptive statistics of the time series
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot the time series
    axes[0].plot(ts_df.index, ts_df['value'], marker='o')
    axes[0].set_title(f"Time Series: {series_id} ({metadata['category']})")
    axes[0].set_ylabel('Value')
    
    # Calculate appropriate lag for ACF/PACF based on dataset size
    # For PACF, max lag should be less than 50% of sample size
    # For ACF, we can use more lags
    max_acf_lags = min(20, len(ts_df)-1)
    max_pacf_lags = min(10, int(len(ts_df)*0.4))  # Ensure it's less than 50%
    
    print(f"Using {max_acf_lags} lags for ACF and {max_pacf_lags} lags for PACF")
    
    # Plot the ACF
    try:
        plot_acf(ts_df['value'], ax=axes[1], lags=max_acf_lags)
        axes[1].set_title('Autocorrelation Function (ACF)')
    except Exception as e:
        print(f"Error plotting ACF: {e}")
        axes[1].text(0.5, 0.5, "Error plotting ACF", 
                     horizontalalignment='center', verticalalignment='center')
    
    # Plot the PACF
    try:
        plot_pacf(ts_df['value'], ax=axes[2], lags=max_pacf_lags)
        axes[2].set_title('Partial Autocorrelation Function (PACF)')
    except Exception as e:
        print(f"Error plotting PACF: {e}")
        axes[2].text(0.5, 0.5, "Error plotting PACF", 
                     horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate basic statistics
    stats = ts_df['value'].describe()
    print(f"Statistics for {series_id}:")
    print(stats)
    
    # Check for stationarity
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(ts_df['value'])
        print('\nAugmented Dickey-Fuller Test:')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        
        # Stationarity interpretation
        if result[1] <= 0.05:
            print("The time series is stationary (reject null hypothesis)")
        else:
            print("The time series is non-stationary (fail to reject null hypothesis)")
    except Exception as e:
        print(f"Error in stationarity test: {e}")
    
    return stats

# Split time series into training and test sets
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

# Naive Forecast (last value)
def naive_forecast(train, test):
    """
    Generate naive forecasts (using last observed value)
    
    Parameters:
    -----------
    train : DataFrame
        Training data
    test : DataFrame
        Test data
        
    Returns:
    --------
    DataFrame
        Forecasts for the test period
    """
    last_value = train['value'].iloc[-1]
    naive_predictions = pd.DataFrame(
        index=test.index,
        data={'forecast': [last_value] * len(test)}
    )
    return naive_predictions

# Exponential Smoothing
def exp_smoothing_forecast(train, test, seasonal=False):
    """
    Generate forecasts using exponential smoothing
    
    Parameters:
    -----------
    train : DataFrame
        Training data
    test : DataFrame
        Test data
    seasonal : bool
        Whether to include seasonal component
        
    Returns:
    --------
    tuple
        (predictions, model) - DataFrame with forecasts and fitted model
    """
    # Ensure data is numeric
    train_values = pd.to_numeric(train['value'], errors='coerce').dropna()
    
    if len(train_values) < 3:
        raise ValueError(f"Not enough observations for Exponential Smoothing. Need at least 3, got {len(train_values)}")
    
    if seasonal:
        model = ExponentialSmoothing(
            train_values,
            trend='add',
            seasonal='add',
            seasonal_periods=1  # Set to appropriate value for seasonal data
        )
    else:
        model = ExponentialSmoothing(
            train_values,
            trend='add',
            seasonal=None
        )
    
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=len(test))
    es_predictions = pd.DataFrame(
        index=test.index,
        data={'forecast': forecast}
    )
    
    return es_predictions, model_fit

# ARIMA Model
def arima_forecast(train, test, order=(1,1,1)):
    """
    Generate forecasts using ARIMA model
    
    Parameters:
    -----------
    train : DataFrame
        Training data
    test : DataFrame
        Test data
    order : tuple
        ARIMA order (p,d,q)
        
    Returns:
    --------
    tuple
        (predictions, model) - DataFrame with forecasts and fitted model
    """
    # Ensure data is numeric
    train_values = pd.to_numeric(train['value'], errors='coerce').dropna().values
    
    # Check if we have enough data
    if len(train_values) < sum(order) + 1:
        raise ValueError(f"Not enough observations for ARIMA{order}. Need at least {sum(order)+1}, got {len(train_values)}")
        
    model = ARIMA(train_values, order=order)
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=len(test))
    arima_predictions = pd.DataFrame(
        index=test.index,
        data={'forecast': forecast}
    )
    
    return arima_predictions, model_fit

# Evaluate forecasts
def evaluate_forecasts(test, predictions_dict):
    """
    Evaluate forecasting models using various metrics
    
    Parameters:
    -----------
    test : DataFrame
        Actual values
    predictions_dict : dict
        Dictionary of forecast DataFrames
        
    Returns:
    --------
    DataFrame
        Evaluation metrics for each model
    """
    results = {}
    
    for model_name, predictions in predictions_dict.items():
        try:
            # Ensure both test and predictions are numeric
            test_values = pd.to_numeric(test['value'], errors='coerce').dropna()
            pred_values = pd.to_numeric(predictions['forecast'], errors='coerce').dropna()
            
            # Align the indices
            aligned_data = pd.concat([test_values, pred_values], axis=1)
            aligned_data.columns = ['actual', 'predicted']
            aligned_data = aligned_data.dropna()
            
            # Skip if we don't have enough aligned data points
            if len(aligned_data) < 1:
                print(f"Warning: Not enough aligned data points for {model_name}")
                results[model_name] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'MAPE': np.nan
                }
                continue
            
            # Calculate metrics
            mse = mean_squared_error(aligned_data['actual'], aligned_data['predicted'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(aligned_data['actual'], aligned_data['predicted'])
            
            # Handle division by zero in MAPE calculation
            if (aligned_data['actual'] == 0).any():
                print(f"Warning: Zero values in test data, MAPE may be undefined for {model_name}")
                # Use a small epsilon to avoid division by zero
                epsilon = 1e-10
                mape = np.mean(np.abs((aligned_data['actual'] - aligned_data['predicted']) / 
                                      (np.abs(aligned_data['actual']) + epsilon))) * 100
            else:
                mape = mean_absolute_percentage_error(aligned_data['actual'], aligned_data['predicted']) * 100
            
            results[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
        
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results[model_name] = {
                'MSE': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan,
                'Error': str(e)
            }
    
    # Convert to DataFrame for easy comparison
    results_df = pd.DataFrame(results).T
    
    return results_df

# Plot forecasts against actual values
def plot_forecasts(train, test, predictions_dict):
    """
    Plot actual values vs forecasts for each model
    
    Parameters:
    -----------
    train : DataFrame
        Training data
    test : DataFrame
        Test data
    predictions_dict : dict
        Dictionary of forecast DataFrames
    """
    plt.figure(figsize=(14, 8))
    
    # Ensure data is numeric
    train_values = pd.to_numeric(train['value'], errors='coerce')
    test_values = pd.to_numeric(test['value'], errors='coerce')
    
    # Plot training data
    plt.plot(train.index, train_values, label='Training Data', color='black', marker='o')
    
    # Plot test data
    plt.plot(test.index, test_values, label='Actual Values', color='blue', linestyle='--', marker='o')
    
    # Plot predictions for each model
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        pred_values = pd.to_numeric(predictions['forecast'], errors='coerce')
        plt.plot(predictions.index, pred_values, label=f'{model_name} Forecast', color=color, marker='s')
    
    plt.title('Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Add a text annotation for the first and last data points to aid in debugging
    for i, (idx, val) in enumerate(test_values.items()):
        if i == 0 or i == len(test_values) - 1:
            plt.annotate(f'{val:.1f}', (idx, val), textcoords="offset points", 
                         xytext=(0,10), ha='center')
    
    plt.show()

# Main analysis function for a single series
def analyze_series(file_path, series_id, test_size=0.2):
    """
    Perform complete analysis on a specific series
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing M1 competition data
    series_id : str
        ID of the series to analyze
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (ts_df, train, test, predictions_dict, results) - All components of the analysis
    """
    print(f"Analyzing series: {series_id}")
    print("=" * 50)
    
    try:
        # Load data
        data_df, seas_df = load_m1_data(file_path)
        
        # Check if series exists (using both exact and stripped comparison)
        if series_id not in data_df['Series'].values and series_id.strip() not in data_df['Series'].str.strip().values:
            available_series = data_df['Series'].tolist()
            print(f"Series {series_id} not found in dataset.")
            print(f"Available series (first 10): {[s.strip() for s in available_series[:10]]}")
            return None, None, None, None, None
        
        # Prepare time series
        ts_df, metadata = prepare_time_series(data_df, series_id)
        print(f"Series metadata: {metadata}")
        
        # Check if we have enough data
        if len(ts_df) < 5:  # Need at least 5 points for meaningful analysis
            print(f"Not enough data points in series {series_id} for analysis")
            return ts_df, None, None, None, None
        
        # Explore time series
        stats = explore_time_series(ts_df, series_id, metadata)
        
        # Split into train and test sets
        train, test = train_test_split(ts_df, test_size=test_size)
        print(f"Training set size: {len(train)}")
        print(f"Test set size: {len(test)}")
        
        if len(test) < 1:
            print("Test set is empty after split. Adjusting test_size...")
            train, test = train_test_split(ts_df, test_size=1/len(ts_df))
            print(f"New training set size: {len(train)}")
            print(f"New test set size: {len(test)}")
        
        # Generate forecasts
        naive_predictions = naive_forecast(train, test)
        
        try:
            es_predictions, es_model = exp_smoothing_forecast(train, test)
        except Exception as e:
            print(f"Error in exponential smoothing: {e}")
            # Create dummy predictions with NaN
            es_predictions = pd.DataFrame(
                index=test.index,
                data={'forecast': [np.nan] * len(test)}
            )
        
        # Determine ARIMA order
        try:
            # Check if differencing is needed based on ADF test
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(train['value'])
            d = 1 if result[1] > 0.05 else 0
            arima_order = (1, d, 1)  # Simple default
            
            arima_predictions, arima_model = arima_forecast(train, test, order=arima_order)
        except Exception as e:
            print(f"Error in ARIMA modeling: {e}")
            # Create dummy predictions with NaN
            arima_predictions = pd.DataFrame(
                index=test.index,
                data={'forecast': [np.nan] * len(test)}
            )
        
        # Combine all predictions
        predictions_dict = {
            'Naive': naive_predictions
        }
        
        # Only add models that worked
        if not es_predictions['forecast'].isna().all():
            predictions_dict['Exponential Smoothing'] = es_predictions
            
        if not arima_predictions['forecast'].isna().all():
            predictions_dict['ARIMA'] = arima_predictions
        
        # Evaluate forecasts
        results = evaluate_forecasts(test, predictions_dict)
        print("\nForecast Evaluation:")
        print(results)
        
        # Plot forecasts
        plot_forecasts(train, test, predictions_dict)
        
        return ts_df, train, test, predictions_dict, results
    
    except Exception as e:
        print(f"Error in analyze_series: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

# Analyze multiple series
def analyze_multiple_series(file_path, series_ids, test_size=0.2):
    """
    Analyze multiple series and compare results
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing M1 competition data
    series_ids : list
        List of series IDs to analyze
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (all_results, mape_df) - Dictionary of results and MAPE comparison DataFrame
    """
    all_results = {}
    
    # First load the data to check series IDs
    data_df, _ = load_m1_data(file_path)
    
    for target_series_id in series_ids:
        # Find the actual series ID with correct spacing
        actual_series_id = find_matching_series(data_df, target_series_id)
        
        if actual_series_id:
            print(f"Found matching series '{actual_series_id}' for requested '{target_series_id}'")
            _, _, _, _, results = analyze_series(file_path, actual_series_id, test_size)
            if results is not None:
                all_results[target_series_id] = results
        else:
            print(f"Could not find series matching '{target_series_id}'")
    
    if not all_results:
        print("No valid results to compare.")
        return None, None
    
    # Compare MAPE across all series
    mape_comparison = {}
    for series_id, results in all_results.items():
        mape_comparison[series_id] = results['MAPE']
    
    mape_df = pd.DataFrame(mape_comparison)
    
    # Plot MAPE comparison
    plt.figure(figsize=(14, 8))
    mape_df.plot(kind='bar')
    plt.title('MAPE Comparison Across Series')
    plt.ylabel('MAPE (%)')
    plt.xlabel('Forecasting Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return all_results, mape_df
