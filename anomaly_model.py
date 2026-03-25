from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    model = IsolationForest(contamination=0.05)
    model.fit(df[['cost']])
    df['anomaly'] = model.predict(df[['cost']])
    return df