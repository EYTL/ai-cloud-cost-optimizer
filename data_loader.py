import pandas as pd
import boto3
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# NASA DATA
# ─────────────────────────────────────────
def load_nasa_data():
    df = pd.read_csv('NASA Dataset for LSTM.csv')
    df = df.rename(columns={'minute': 'date'})
    df['cost'] = df['count'] * 0.000001

    def classify_load(count):
        if count > 100:
            return 'High Load'
        elif count > 40:
            return 'Normal Load'
        else:
            return 'Low Load'

    df['department'] = df['count'].apply(classify_load)
    return df


# ─────────────────────────────────────────
# FILE UPLOAD DATA
# Replicates NASA pipeline on any uploaded CSV
# Required columns: date + one of [cost, count, amount, spend]
# ─────────────────────────────────────────
def load_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    df.columns = [c.strip().lower() for c in df.columns]

    # --- Find date column ---
    date_col = None
    for candidate in ['date', 'time', 'timestamp', 'day', 'period', 'datetime']:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError(
            "No date column found. Your CSV must have a column named: "
            "date, time, timestamp, day, period, or datetime."
        )
    df = df.rename(columns={date_col: 'date'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # --- Find cost column ---
    cost_col = None
    for candidate in ['cost', 'amount', 'spend', 'spending', 'charge', 'total', 'unblendedcost']:
        if candidate in df.columns:
            cost_col = candidate
            break

    if cost_col is None:
        # If no cost col, check for count (like NASA) and derive cost
        if 'count' in df.columns:
            df['cost'] = pd.to_numeric(df['count'], errors='coerce').fillna(0) * 0.000001
        else:
            raise ValueError(
                "No cost column found. Your CSV must have a column named: "
                "cost, amount, spend, spending, charge, total, or count."
            )
    else:
        df['cost'] = pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
        if cost_col != 'cost':
            df = df.rename(columns={cost_col: 'cost'})

    # --- Department / Load classification ---
    if 'department' not in df.columns:
        if 'count' in df.columns:
            def classify_load(count):
                if count > 100:
                    return 'High Load'
                elif count > 40:
                    return 'Normal Load'
                else:
                    return 'Low Load'
            df['department'] = df['count'].apply(classify_load)
        elif 'service' in df.columns:
            df['department'] = df['service']
        elif 'region' in df.columns:
            df['department'] = df['region']
        else:
            # Classify by cost quartile
            q33 = df['cost'].quantile(0.33)
            q66 = df['cost'].quantile(0.66)
            def classify_cost(c):
                if c > q66:
                    return 'High Spend'
                elif c > q33:
                    return 'Normal Spend'
                else:
                    return 'Low Spend'
            df['department'] = df['cost'].apply(classify_cost)

    df = df.sort_values('date').reset_index(drop=True)
    return df


# ─────────────────────────────────────────
# AWS COST EXPLORER
# ─────────────────────────────────────────
def fetch_aws_data(days=90, granularity='DAILY',
                   aws_access_key=None, aws_secret_key=None, aws_region='us-east-1'):
    """
    Fetch AWS cost data.
    Credentials priority:
      1. Passed directly (from UI input)
      2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
      3. IAM role / ~/.aws/credentials (boto3 default chain)
    """
    session_kwargs = {'region_name': aws_region}

    if aws_access_key and aws_secret_key:
        session_kwargs['aws_access_key_id'] = aws_access_key
        session_kwargs['aws_secret_access_key'] = aws_secret_key

    client = boto3.client('ce', **session_kwargs)

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')

    response = client.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity=granularity,
        Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )

    rows = []
    for result in response['ResultsByTime']:
        date = result['TimePeriod']['Start']
        if result['Groups']:
            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                rows.append({'date': date, 'cost': cost, 'department': service})
        else:
            cost = float(result['Total']['UnblendedCost']['Amount'])
            rows.append({'date': date, 'cost': cost, 'department': 'AWS'})

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df



# ─────────────────────────────────────────
# MAIN LOADER (called by app.py)
# ─────────────────────────────────────────
def load_data(source, uploaded_file=None,
              aws_access_key=None, aws_secret_key=None, aws_region='us-east-1', aws_days=90):

    if source == 'nasa':
        return load_nasa_data()
    elif source == 'upload':
        return load_uploaded_file(uploaded_file)
    elif source == 'aws':
        return fetch_aws_data(
            days=aws_days,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region
        )
    else:
        raise ValueError(f"Unknown source: {source}")