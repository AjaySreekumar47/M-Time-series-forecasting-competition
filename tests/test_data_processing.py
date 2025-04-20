"""
Tests for data processing functions.
"""

import os
import pandas as pd
import numpy as np
import pytest
from src.data_processing import (
    find_matching_series,
    prepare_time_series,
    train_test_split,
    check_seasonality
)

# Sample data for testing
@pytest.fixture
def sample_data():
    # Create a simple DataFrame that mimics the M1 competition format
    data = {
        'Series': ['YAF2  ', 'YAF3  '],
        'N Obs': [22, 23],
        'Seasonality': [1, 1],
        'NF': [6, 6],
        'Type': ['YEARLY', 'YEARLY'],
        'Starting date': [1972, 1974],
        'Category': ['MICRO1', 'MICRO1'],
    }
    
    # Add numeric columns (1-28) with sample values
    for i in range(1, 29):
        data[i] = [i*100, i*200]
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_seasonality_data():
    # Create a simple DataFrame that mimics the seasonality sheet
    data = {
        'Series': ['YAF2  ', 'YAF3  '],
        'Seasonality': [1, 1]
    }
    
    # Add 12 columns for monthly seasonality indicators
    for i in range(1, 13):
        data[i] = [1 if i == 1 else 0, 1 if i == 1 else 0]
    
    return pd.DataFrame(data)

def test_find_matching_series(sample_data):
    # Test exact match
    assert find_matching_series(sample_data, 'YAF2  ') == 'YAF2  '
    
    # Test match with stripped spaces
    assert find_matching_series(sample_data, 'YAF2') == 'YAF2  '
    
    # Test no match
    assert find_matching_series(sample_data, 'NONEXISTENT') is None

def test_prepare_time_series(sample_data):
    # Test with exact series ID
    ts_df, metadata = prepare_time_series(sample_data, 'YAF2  ')
    
    # Check DataFrame structure
    assert isinstance(ts_df, pd.DataFrame)
    assert 'value' in ts_df.columns or 'value' == ts_df.index.name
    
    # Check metadata
    assert metadata['category'] == 'MICRO1'
    assert metadata['start_year'] == 1972
    
    # Check time series values
    assert len(ts_df) == 22  # N Obs value
    
    # Test with stripped series ID
    ts_df2, metadata2 = prepare_time_series(sample_data, 'YAF2')
    assert len(ts_df2) == 22
    
    # Test with nonexistent series
    with pytest.raises(ValueError):
        prepare_time_series(sample_data, 'NONEXISTENT')

def test_train_test_split():
    # Create a simple time series
    dates = pd.date_range(start='2000-01-01', periods=10, freq='YS')
    values = range(10)
    ts_df = pd.DataFrame({'value': values}, index=dates)
    
    # Test default split (80/20)
    train, test = train_test_split(ts_df)
    assert len(train) == 8
    assert len(test) == 2
    
    # Test custom split
    train, test = train_test_split(ts_df, test_size=0.3)
    assert len(train) == 7
    assert len(test) == 3
    
    # Test with small dataset
    small_ts = ts_df.iloc[:3]
    train, test = train_test_split(small_ts, test_size=0.33)
    assert len(train) == 2
    assert len(test) == 1

def test_check_seasonality(sample_seasonality_data):
    # Test non-seasonal yearly series
    has_seasonality, period = check_seasonality(sample_seasonality_data, 'YAF2  ')
    assert has_seasonality is False
    assert period == 1
    
    # Test with stripped series ID
    has_seasonality2, period2 = check_seasonality(sample_seasonality_data, 'YAF2')
    assert has_seasonality2 is False
    assert period2 == 1
    
    # Test with nonexistent series
    has_seasonality3, period3 = check_seasonality(sample_seasonality_data, 'NONEXISTENT')
    assert has_seasonality3 is False
    assert period3 == 1
    
    # Modify data to create a seasonal series
    seasonal_data = sample_seasonality_data.copy()
    seasonal_data.loc[0, 1] = 1
    seasonal_data.loc[0, 4] = 1  # Add a second seasonal peak
    
    has_seasonality4, period4 = check_seasonality(seasonal_data, 'YAF2')
    assert has_seasonality4 is True
    assert period4 == 12  # Monthly seasonality
