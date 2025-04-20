"""
M1 Competition Forecasting package.

This package provides tools for analyzing and forecasting time series from the M1 Competition.
"""

from .forecasting import (
    load_m1_data,
    find_matching_series,
    prepare_time_series,
    explore_time_series,
    train_test_split,
    naive_forecast,
    exp_smoothing_forecast,
    arima_forecast,
    evaluate_forecasts,
    plot_forecasts,
    analyze_series,
    analyze_multiple_series,
)

__version__ = '0.1.0'
