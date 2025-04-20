# Data Directory

This directory contains example data files and resources for the M1 Competition forecasting project.

## Sample Data

- `sample_data.xls`: A sample Excel file containing selected series from the M1 Competition
- Add your own M1 Competition data files to this directory

## Data Structure

The M1 Competition Excel files typically have the following structure:

1. **Main data sheet** (typically named "MC1001" or similar):
   - Series: Identifier for each time series (e.g., "YAF2")
   - N Obs: Number of observations in the series
   - Seasonality: Value indicating seasonality (1 = yearly, no seasonality)
   - NF: Code indicating forecasting horizon
   - Type: Time interval of the series (YEARLY, QUARTERLY, MONTHLY, OTHER)
   - Starting date: Year when the time series begins
   - Category: Type of data (MICRO1, MICRO2, MICRO3, MACRO1, MACRO2, DEMOGR, INDUST)
   - Columns 1-N: The actual time series values

2. **Seasonality indicators sheet** (typically named "MCSeasInd"):
   - Contains seasonality indicators for each series
   - A value of 1 for month 1 and 0 for others indicates yearly data

## Obtaining the Full Dataset

The complete M1 Competition dataset can be obtained from:

1. The International Institute of Forecasters: [https://forecasters.org/resources/time-series-data/](https://forecasters.org/resources/time-series-data/)
2. Rob J. Hyndman's Time Series Data Library: [https://pkg.robjhyndman.com/Mcomp/](https://pkg.robjhyndman.com/Mcomp/)
3. Through the R package "Mcomp": `install.packages("Mcomp")`

## Data Usage

Please note that while the M1 Competition data is publicly available for research purposes, you should cite the original paper when using it:

Makridakis, S., Andersen, A., Carbone, R., Fildes, R., Hibon, M., Lewandowski, R., Newton, J., Parzen, E., & Winkler, R. (1982). The accuracy of extrapolation (time series) methods: results of a forecasting competition. Journal of Forecasting, 1, 111-153.

## Data File Guidelines

- Keep your data files in this directory for ease of access
- Large data files should not be committed to the repository (they are ignored in .gitignore)
- Include a small sample dataset for testing purposes
