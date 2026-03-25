from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    model = IsolationForest(contamination=0.05)
    model.fit(df[['cost']])
    df['anomaly'] = model.predict(df[['cost']])
    return df

from data_loader import load_data

df  = load_data('nasa')
df = detect_anomalies(df)

print(df['anomaly'].value_counts())