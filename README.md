# M1 Competition Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive toolkit for analyzing and forecasting time series from the M1 Competition dataset. This project implements multiple forecasting methods and provides evaluation tools to compare their performance.

## About the M1 Competition

The M1 Competition was the first of a series of forecasting competitions organized by Spyros Makridakis in 1982. It used 1001 time series (numbered 1001-3003) to evaluate and compare the accuracy of different forecasting methods. The dataset includes yearly, quarterly, and monthly time series representing data from various categories including micro, macro, industry, and demographic data.

## Features

- Data loading and preprocessing for M1 Competition Excel files
- Comprehensive time series exploration with statistics and visualizations
- Multiple forecasting methods:
  - Naive (last value) forecasting
  - Exponential Smoothing
  - ARIMA modeling
- Statistical evaluation with multiple metrics (MSE, RMSE, MAE, MAPE)
- Visualization tools for comparing forecast performance
- Support for batch analysis of multiple series

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/m1-competition-forecasting.git
cd m1-competition-forecasting
pip install -r requirements.txt
```

## Usage

### Google Colab

The easiest way to use this project is through Google Colab:

1. Upload your M1 Competition Excel file to Google Drive
2. Open the [forecasting_notebook.ipynb](notebooks/forecasting_notebook.ipynb) in Google Colab
3. Mount your Google Drive and specify the path to your Excel file
4. Run the notebook cells to analyze your data

### Local Python Environment

To use the project locally:

```python
from src.forecasting import load_m1_data, analyze_series, analyze_multiple_series

# Load the data
file_path = 'path/to/your/MC1001.xls'
data_df, _ = load_m1_data(file_path)

# Analyze a single series
series_id = 'YAF2'  # Series ID without trailing spaces
ts_df, train, test, predictions, results = analyze_series(file_path, series_id)

# Analyze multiple series
series_ids = ['YAF2', 'YAF3']
all_results, mape_comparison = analyze_multiple_series(file_path, series_ids)
```

## Directory Structure

```
m1-competition-forecasting/
│
├── data/                          # Sample data files and resources
│   ├── sample_data.xls            # Sample M1 Competition series
│   └── README.md                  # Data directory information
│
├── notebooks/                     # Jupyter notebooks
│   ├── forecasting_notebook.ipynb # Main notebook for Google Colab
│   └── examples.ipynb             # Example analyses and use cases
│
├── src/                           # Source code
│   ├── __init__.py                # Package initialization
│   ├── forecasting.py             # Core forecasting functionality
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── evaluation.py              # Metrics and evaluation tools
│   └── visualization.py           # Plotting and visualization tools
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_forecasting.py
│   ├── test_data_processing.py
│   └── test_evaluation.py
│
├── .gitignore                     # Git ignore file
├── LICENSE                        # Project license
├── requirements.txt               # Project dependencies
├── setup.py                       # Package setup script
└── README.md                      # Project documentation (this file)
```

## Example

![Forecast Example](docs/images/forecast_example.png)

```python
# Quick start example
from src.forecasting import analyze_series

file_path = 'data/sample_data.xls'
series_id = 'YAF2'
ts_df, train, test, predictions, results = analyze_series(file_path, series_id)

print(results)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Spyros Makridakis and the organizers of the M Competitions
- The International Institute of Forecasters for maintaining the dataset
- All contributors to the statsmodels, pandas, and matplotlib libraries
