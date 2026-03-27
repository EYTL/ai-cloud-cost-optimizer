import numpy as np
import pandas as pd
from sklearn .preprocessing import MinMaxScaler

def perpare_data(df):
    values = df['cost'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    return scaled, scaler

def create_sequences(data, n_steps=60):
    X, y = [], []
    for i in range (len(data) -n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(n_steps):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def forecast_lstm(df, n_steps = 60, epochs = 10):
    scaled, scaler = perpare_data(df)
    X, y = create_sequences(scaled, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_model(n_steps)
    model.fit(X, y, epochs=epochs, verbose = 0)

    last_seqence = scaled[-n_steps:]
    last_seqence = last_seqence.reshape((1, n_steps, 1))
    predicted_scaled = model.predict(last_seqence)
    predicted = scaler.inverse_transform(predicted_scaled)

    return float(predicted[0][0])

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data('nasa')
    print("Training LSTM model...")
    prediction = forecast_lstm(df, n_steps=60, epochs=5)
    print(f"Next period predicted cost: ${prediction:.6f}")