import pandas as pd
import boto3
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def load_nasa_data():
    df = pd.read_csv('Nasa Dataset for LSTM.csv')
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

def fetch_aws_data():
    client = boto3.client('ce', region_name='us-east-1')
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )
    
    dates = []
    costs = []
    for result in response['ResultsByTime']:
        dates.append(result['TimePeriod']['Start'])
        costs.append(float(result['Total']['UnblendedCost']['Amount']))
    
    return pd.DataFrame({'date': dates, 'cost': costs, 'department': 'AWS'})

def load_data(source):
    if source == 'aws':
        return fetch_aws_data()
    elif source == 'nasa':
        return load_nasa_data()