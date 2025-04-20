"""
Visualization utilities for M1 Competition time series.
This module contains functions for plotting and visualizing time series and forecasts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12


def plot_time_series(ts_df, title=None, figsize=(14, 6)):
    """
    Plot a time series
    
    Parameters:
    -----------
    ts_df : DataFrame
        DataFrame containing time series data with 'value' column
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure data is numeric
    values = pd.to_numeric(ts_df['value'], errors='coerce')
    
    # Plot time series
    ax.plot(ts_df.index, values, marker='o')
    
    # Set title and labels
    if title:
        ax.set_title(title)
    ax.set_ylabel('Value')
    ax.set_xlabel('Date')
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_functions(ts_df, max_lags=None, figsize=(14, 8)):
    """
    Plot ACF and PACF for a time series
    
    Parameters:
    -----------
    ts_df : DataFrame
        DataFrame containing time series data with 'value' column
    max_lags : int, optional
        Maximum number of lags to include
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Ensure data is numeric
    values = pd.to_numeric(ts_df['value'], errors='coerce').dropna()
    
    # Calculate appropriate lag for ACF/PACF based on dataset size
    # For PACF, max lag should be less than 50% of sample size
    # For ACF, we can use more lags
    if max_lags is None:
        max_acf_lags = min(20, len(values)-1)
        max_pacf_lags = min(10, int(len(values)*0.4))  # Ensure it's less than 50%
    else:
        max_acf_lags = max_pacf_lags = max_lags
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot the ACF
    try:
        plot_acf(values, ax=axes[0], lags=max_acf_lags)
        axes[0].set_title('Autocorrelation Function (ACF)')
    except Exception as e:
        print(f"Error plotting ACF: {e}")
        axes[0].text(0.5, 0.5, "Error plotting ACF", 
                     horizontalalignment='center', verticalalignment='center')
    
    # Plot the PACF
    try:
        plot_pacf(values, ax=axes[1], lags=max_pacf_lags)
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
    except Exception as e:
        print(f"Error plotting PACF: {e}")
        axes[1].text(0.5, 0.5, "Error plotting PACF", 
                    horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    return fig


def plot_forecasts(train, test, predictions_dict, title=None, figsize=(14, 8)):
    """
    Plot actual values vs forecasts for multiple models
    
    Parameters:
    -----------
    train : DataFrame
        Training data with 'value' column
    test : DataFrame
        Test data with 'value' column
    predictions_dict : dict
        Dictionary of DataFrames containing forecast values
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure data is numeric
    train_values = pd.to_numeric(train['value'], errors='coerce')
    test_values = pd.to_numeric(test['value'], errors='coerce')
    
    # Plot training data
    ax.plot(train.index, train_values, label='Training Data', color='black', marker='o')
    
    # Plot test data
    ax.plot(test.index, test_values, label='Actual Values', color='blue', linestyle='--', marker='o')
    
    # Plot predictions for each model
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        pred_values = pd.to_numeric(predictions['forecast'], errors='coerce')
        ax.plot(predictions.index, pred_values, label=f'{model_name} Forecast', color=color, marker='s')
    
    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Forecast Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    # Add a text annotation for the first and last data points
    for i, (idx, val) in enumerate(test_values.items()):
        if i == 0 or i == len(test_values) - 1:
            ax.annotate(f'{val:.1f}', (idx, val), textcoords="offset points", 
                        xytext=(0,10), ha='center')
    
    plt.tight_layout()
    return fig


def plot_forecast_errors(actual, predictions_dict, figsize=(14, 6)):
    """
    Plot forecast errors for multiple models
    
    Parameters:
    -----------
    actual : Series
        Actual values
    predictions_dict : dict
        Dictionary of DataFrames containing forecast values
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure actual is numeric
    actual_values = pd.to_numeric(actual, errors='coerce')
    
    # Calculate and plot errors for each model
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        pred_values = pd.to_numeric(predictions['forecast'], errors='coerce')
        
        # Align indices
        common_idx = actual_values.index.intersection(pred_values.index)
        if len(common_idx) == 0:
            continue
            
        aligned_actual = actual_values.loc[common_idx]
        aligned_pred = pred_values.loc[common_idx]
        
        # Calculate errors
        errors = aligned_actual - aligned_pred
        
        # Plot errors
        color = colors[i % len(colors)]
        ax.plot(common_idx, errors, label=f'{model_name} Error', color=color, marker='o')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='--')
    
    # Set title and labels
    ax.set_title('Forecast Errors')
    ax.set_xlabel('Date')
    ax.set_ylabel('Error (Actual - Forecast)')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_mape_comparison(mape_df, title="MAPE Comparison Across Models", figsize=(14, 8)):
    """
    Plot MAPE comparison across models and series
    
    Parameters:
    -----------
    mape_df : DataFrame
        DataFrame containing MAPE values, models as index and series as columns
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot MAPE comparison
    mape_df.plot(kind='bar', ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_ylabel('MAPE (%)')
    ax.set_xlabel('Forecasting Method')
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig
