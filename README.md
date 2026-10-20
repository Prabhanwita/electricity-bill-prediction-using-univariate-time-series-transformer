# Electricity Generation Forecasting Using Transformer

A time-series forecasting project that uses a Transformer-based deep
learning model to predict electricity generation at monthly, daily, and
hourly resolutions.

> **Scope note:** The notebook is named
> `electricity_bill_prediction.ipynb`, but the described implementation
> focuses on forecasting electricity generation in GWh. It does not
> currently calculate an individual consumer's electricity bill.

## Overview

This project demonstrates a machine-learning workflow for
electricity-generation time series. It includes data preprocessing,
time-series sequence construction, a Transformer encoder model, and
forecasting workflows for multiple time resolutions.

The notebook can use synthetic/mock data in its data-generation
workflow. Forecasts from synthetic data are useful for demonstrating the
pipeline, but they do not establish real-world forecasting accuracy.

## Objectives

-   Load and preprocess electricity-generation data.
-   Convert date fields and organize observations chronologically.
-   Aggregate generation records into a time series.
-   Construct input sequences and forecast targets using a lookback
    window and forecast horizon.
-   Train a Transformer-based forecasting model.
-   Generate monthly, daily, and hourly forecast outputs.
-   Evaluate predictions against held-out actual data when such data are
    available.

## Features

-   CSV-based data processing with Pandas.
-   Time-series preparation and aggregation.
-   PyTorch Dataset for lookback and target sequences.
-   Transformer encoder with positional encoding and multi-head
    attention.
-   Forecasting workflows for multiple time resolutions.
-   Optional plotting and analysis using Matplotlib.

## Technologies

-   **Python**
-   **Pandas** --- data loading and transformation
-   **NumPy** --- numerical operations
-   **PyTorch** --- Transformer model and training
-   **Scikit-learn** --- scaling utilities
-   **Matplotlib** --- visualization

## Model Architecture

The described Transformer pipeline consists of:

1.  Linear input projection
2.  Sinusoidal positional encoding
3.  Transformer encoder with multi-head attention
4.  Output projection for forecast values

The notebook's described configuration includes `d_model = 64`,
`nhead = 4`, three encoder layers, and dropout of `0.1`.

## Forecasting Workflows

  -----------------------------------------------------------------------
  Frequency               Period represented in   Purpose
                          the notebook            
  ----------------------- ----------------------- -----------------------
  Monthly                 January 2019 --         Monthly generation
                          December 2025           forecasting

  Daily                   January 2024 -- June    Daily generation
                          2026                    forecasting

  Hourly                  April -- June 2026      Hourly generation
                                                  forecasting
  -----------------------------------------------------------------------

These periods describe the notebook's workflows; they should not be
interpreted as proof that all records are real-world observations. Check
the notebook's data-generation cells to determine whether a workflow
uses synthetic or actual data.

## Project Structure

``` text
.
├── electricity_bill_prediction.ipynb
└── README.md
```

Add any datasets, output files, or supporting Python modules to this
structure if they are part of your repository. Avoid committing private
or large raw datasets unless you have permission and a clear reason to
include them.

## Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your GitHub repository URL and folder
name.

### 2. Create a virtual environment (recommended)

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install pandas numpy torch scikit-learn matplotlib jupyter
```

If the notebook imports additional libraries, install those packages as
well.

### 4. Launch Jupyter Notebook

``` bash
jupyter notebook
```

Open `electricity_bill_prediction.ipynb` and run its cells in order.
Ensure any required input files are available at the paths expected by
the notebook.

## Workflow

1.  **Data loading:** Read electricity data from CSV files or generate
    mock data where configured.
2.  **Preprocessing:** Convert dates, filter the relevant generation
    records, pivot and aggregate observations, and sort them
    chronologically.
3.  **Sequence creation:** Create historical input windows and future
    target sequences.
4.  **Model training:** Train the Transformer on the prepared sequences.
5.  **Forecast generation:** Run the relevant monthly, daily, or hourly
    workflow.
6.  **Visualization and evaluation:** Inspect forecast output and, when
    actual held-out observations are available, calculate evaluation
    metrics.

## Evaluation

Forecast plots or predicted values alone do not demonstrate model
accuracy. For a reliable evaluation:

-   Split time-series data chronologically into training, validation,
    and test sets.
-   Avoid using future observations during training or preprocessing.
-   Compare forecasts with actual held-out values.
-   Report appropriate metrics such as:
    -   **MAE** --- Mean Absolute Error
    -   **RMSE** --- Root Mean Squared Error
    -   **MAPE** --- Mean Absolute Percentage Error, with care when
        actual values are zero or close to zero.

Do not report performance scores unless they have been calculated from a
clearly defined evaluation set.

## Limitations

-   Synthetic/mock data may be used in the notebook and may not reflect
    real electricity-generation patterns.
-   Real-world predictive accuracy has not been established by forecast
    output alone.
-   Aggregating state-level generation using a mean or sum changes the
    meaning of the target; choose the method that matches the
    forecasting objective.
-   Data scaling should be fitted on training data only, and predictions
    should be inverse-transformed if required.
-   The current generation-forecasting workflow is not a consumer bill
    calculator.

## Future Improvements

-   Validate the model on authentic electricity-generation data.
-   Add chronological train/validation/test evaluation and document MAE,
    RMSE, and MAPE.
-   Compare the Transformer with baseline and alternative forecasting
    models.
-   Save and reload trained model checkpoints.
-   Improve logging, missing-data handling, and configuration
    management.
-   Add a dashboard for forecast exploration.
-   If electricity-bill prediction is the intended goal, add
    consumer-level consumption data and applicable tariff rules, fixed
    charges, and taxes.

## Contributing

Contributions are welcome. For substantial changes, consider opening an
issue first to discuss the proposed improvement. Include clear steps to
reproduce results and avoid committing sensitive data or credentials.

## License

No license is specified here. Add a `LICENSE` file if you want to define
how others may use, modify, and distribute this project.
