# AI Cloud Cost Optimizer

AI Cloud Cost Optimizer is a Streamlit-based analytics dashboard for monitoring cloud spend, identifying abnormal cost behavior, estimating future spending, and surfacing potential waste. The project combines rule-based operational insights with machine learning techniques so teams can review cloud usage patterns from a single interface.

The application supports three data sources:

- A bundled NASA traffic dataset used as a demo dataset
- User-uploaded CSV files
- AWS Cost Explorer data fetched directly from an AWS account

## Key Features

- Interactive dashboard built with Streamlit
- Cost anomaly detection using `IsolationForest`
- Moving-average cost forecasting
- LSTM-based next-period cost prediction using TensorFlow/Keras
- Waste detection for low-utilization or low-spend periods
- Rule-based autoscaling recommendation engine
- Load-balancing visualization across departments or services
- Executive summary metrics for quick decision-making
- CSV export of detected anomalies

## How The System Works

The dashboard follows a simple processing pipeline:

1. A data source is selected in the sidebar.
2. Data is loaded and normalized into a common structure.
3. Anomaly labels are added to the dataset.
4. Visual and numeric summaries are generated for forecasting, waste detection, scaling guidance, and department-level spend analysis.

### Standardized Data Model

Regardless of source, the application tries to produce a DataFrame with these core fields:

- `date`: timestamp or period date
- `cost`: numeric spend value
- `department`: category used for grouping and balancing views

Some datasets may also include:

- `count`: traffic or usage count
- `anomaly`: anomaly label added by the model (`-1` for anomaly, `1` for normal)
- `forecast`: moving-average forecast column generated in the app

## Main Modules

### `app.py`

This is the main Streamlit application. It is responsible for:

- Rendering the UI
- Handling data-source selection
- Accepting CSV uploads
- Accepting AWS credentials and query range
- Calling the data loader and anomaly detector
- Computing scaling recommendations
- Building charts, metrics, tables, and downloadable reports

### `data_loader.py`

This module handles all input pipelines.

- `load_nasa_data()`
  Loads `NASA Dataset for LSTM.csv`, renames the time column, derives a synthetic cost value from traffic counts, and classifies rows into load bands.

- `load_uploaded_file(uploaded_file)`
  Accepts a user CSV, detects suitable date and cost columns, normalizes the data, and creates a `department` grouping when one is missing.

- `fetch_aws_data(...)`
  Queries AWS Cost Explorer using `boto3`, groups results by AWS service, and returns normalized cost data.

- `load_data(...)`
  Acts as the main dispatcher used by the app.

### `anomaly_model.py`

This module runs anomaly detection using Scikit-learn's `IsolationForest`. The model trains on the `cost` column and appends an `anomaly` label to each row.

### `lstm_forecast.py`

This module handles sequence preparation, LSTM model construction, and next-step cost prediction.

Core logic includes:

- Min-max scaling of cost values
- Sequence generation for time-series learning
- TensorFlow/Keras LSTM model training
- Fallback behavior for small datasets

## Dashboard Sections

The dashboard is organized into the following functional areas:

### 1. Autoscaling and Load Balancing

The app calculates recent cost and traffic trends, then produces a recommendation:

- `Scale Up`
- `Scale Down`
- `Stable`

It also simulates load-balancer distribution by showing how total cost is distributed across departments or services.

### 2. Cost Forecast

A rolling moving average is calculated over recent cost values to estimate the next period's cost.

### 3. LSTM Neural Network Forecast

An LSTM model predicts the next cost value using historical cost sequences. The app caches results to avoid retraining when inputs do not change.

### 4. Waste Detector

Waste is identified differently depending on the dataset:

- If a `count` column exists, low-traffic periods are flagged
- Otherwise, very low-cost periods are flagged

### 5. Anomaly Detection

The app highlights outlier cost points on a time-series chart and provides a downloadable anomaly report in CSV format.

### 6. Executive Summary

This section provides a compact KPI view of:

- Total spend
- Average daily cost
- Highest cost
- Next-period forecast
- Total records analyzed
- Number of anomalies
- Number of wasteful periods
- Overall system health

### 7. Cost By Department or Service

Spend is aggregated and shown as a bar chart for business-unit or service-level comparison.

### 8. Raw Data

The processed dataset is displayed in tabular form for inspection and validation.

## Installation

### Requirements

- Python 3.10 or later recommended
- Pip
- Internet access if AWS Cost Explorer will be used

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependency List

From `requirements.txt`, the project currently uses:

- `streamlit`
- `pandas`
- `plotly`
- `scikit-learn`
- `boto3`
- `python-dotenv`
- `tensorflow`

## Running The Project

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Using The Dashboard

### Option 1: NASA Demo Dataset

Choose `NASA Dataset` in the sidebar to run the app using the bundled sample data.

### Option 2: Upload A CSV File

Choose `Upload CSV File` and provide a CSV with:

- A date-like column such as `date`, `time`, `timestamp`, `day`, `period`, or `datetime`
- A numeric spend column such as `cost`, `amount`, `spend`, `spending`, `charge`, `total`, or `unblendedcost`

If no cost column exists but a `count` column is available, the app derives cost automatically.

### Option 3: AWS Cost Explorer

Choose `AWS Cost Explorer` and provide:

- AWS Access Key ID
- AWS Secret Access Key
- AWS Region
- Historical window in days

You can also leave credentials blank and rely on environment variables or IAM-based resolution if your runtime is already configured.

## Environment Variables

The project loads environment variables through `python-dotenv`. For AWS access, these standard variables can be used:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Expected CSV Behavior

When uploaded data does not include a `department` column, the app derives one automatically:

- From `count` using load bands
- From `service` if present
- From `region` if present
- Otherwise from spend quartiles

This lets the dashboard keep a consistent grouping model across different data sources.

## Project Structure

```text
AI_Cost_Project/
|-- app.py
|-- data_loader.py
|-- anomaly_model.py
|-- lstm_forecast.py
|-- requirements.txt
|-- NASA Dataset for LSTM.csv
|-- cloud_cost.csv
|-- AI OPT.py
|-- beautify_app.py
|-- reorder_sidebar.py
```

## Limitations

- Anomaly detection is currently based only on the `cost` column.
- The contamination rate for `IsolationForest` is fixed in code.
- The autoscaling recommendation is heuristic, not policy-driven.
- The LSTM model is retrained inside the app and may be slow on large datasets.
- AWS integration currently focuses on Cost Explorer grouped by service.
- No automated tests or CI workflow are included in the repository at this time.

## Suggested Improvements

- Add configuration options for anomaly sensitivity and contamination
- Persist trained forecasting models for reuse
- Add support for Azure and GCP billing sources
- Introduce unit tests for loaders and forecasting logic
- Separate business logic from UI for easier maintenance
- Add authentication and role-based access if deployed for teams

## Notes On Source Material

This documentation is based on the current project implementation in the repository. A scanned PDF was provided as source context, but the local environment available in this session did not include working OCR or PDF-text extraction tools, so no unsupported claims from the scan were added here.
