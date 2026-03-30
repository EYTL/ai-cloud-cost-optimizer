import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_data(df):
    values = df['cost'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
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


def forecast_lstm(df, n_steps=60, epochs=10):
    scaled, scaler = prepare_data(df)

    total_rows = len(scaled)

    # --- FIX: Guard against insufficient data ---
    # Need at least n_steps + 1 rows to make even one sequence
    # If not enough data, reduce n_steps to half the available rows
    if total_rows <= n_steps:
        n_steps = max(2, total_rows // 2)  # minimum n_steps of 2

    # After adjusting n_steps, check we still have enough for at least 1 sequence
    if total_rows <= n_steps:
        # Absolute fallback: return rolling mean if data is too small
        fallback = df['cost'].rolling(window=2).mean().dropna().iloc[-1]
        return float(fallback)

    X, y = create_sequences(scaled, n_steps)

    # Safety check: if sequences are still empty
    if len(X) == 0:
        fallback = df['cost'].rolling(window=2).mean().dropna().iloc[-1]
        return float(fallback)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_model(n_steps)
    model.fit(X, y, epochs=epochs, verbose=0)

    last_sequence = scaled[-n_steps:]
    last_sequence = last_sequence.reshape((1, n_steps, 1))
    predicted_scaled = model.predict(last_sequence)
    predicted = scaler.inverse_transform(predicted_scaled)

    return float(predicted[0][0])


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data('nasa')
    print("Training LSTM on NASA data...")
    prediction = forecast_lstm(df, n_steps=60, epochs=5)
    print(f"NASA predicted cost: ${prediction:.6f}")

    df_aws = load_data('aws')
    print("Training LSTM on AWS data...")
    prediction_aws = forecast_lstm(df_aws, n_steps=60, epochs=5)
    print(f"AWS predicted cost: ${prediction_aws:.6f}")