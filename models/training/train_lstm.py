import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Navigate to stock-dashboard root directory (Two levels up from training/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Correct directory paths
DATA_DIR = os.path.join(BASE_DIR, "web_data")  # Path to stock data
MODEL_DIR = os.path.join(BASE_DIR, "models", "LSTM")  # Store trained models
RESULTS_DIR = os.path.join(BASE_DIR, "models", "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

# List of tickers and corresponding file names
TICKERS = {
    "NVDA": "nvidia_stock_data.csv",
    "AAPL": "apple_stock_data.csv",
    "MSFT": "microsoft_stock_data.csv",
    "AAL": "aal_stock_data.csv",
    "GOOGL": "googl_stock_data.csv",
    "TSLA": "tsla_stock_data.csv",
    "AMZN": "amzn_stock_data.csv",
    "META": "meta_stock_data.csv",
    "INTC": "intc_stock_data.csv",
    "HPE": "hpe_stock_data.csv",
}

# Prepare Data with Train-Test Split
def prepare_data(df, target_column="Close", sequence_length=60, test_size=0.2):
    features = df.drop(columns=[target_column]).values
    target = df[target_column].values

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i - sequence_length : i])
        y.append(target_scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test, scaler, df.index[split_idx + sequence_length]

# Load Stock Data
def load_stock_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    return df

# Custom Weighted Loss Function
def custom_weighted_loss(y_true, y_pred):
    weights = tf.range(1, tf.shape(y_true)[0] + 1, dtype=tf.float32)
    weights = tf.math.exp(weights / tf.reduce_max(weights))
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = mse(y_true, y_pred)
    return tf.reduce_mean(loss * weights)

# Function to Train and Save LSTM Model
def custom_lstm_model(stock, filepath):
    print(f"Training LSTM model for {stock}...")

    # Load data
    df = load_stock_data(filepath)
    X_train, y_train, X_test, y_test, scaler, prediction_start_date = prepare_data(df)

    # Build model
    model = Sequential([
        Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1),
        Bidirectional(LSTM(units=50, return_sequences=False)),
        Dropout(0.1),
        Dense(units=25),
        Dense(units=1)
    ])

    # Compile model with custom weighted loss
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=custom_weighted_loss)

    # Define callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Save trained model
    model_path = os.path.join(MODEL_DIR, f"{stock}_lstm.h5")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    # Predict only for the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate Model
    mse = tf.keras.losses.MeanSquaredError()(y_test_actual, predictions).numpy()
    r2 = 1 - (np.sum((y_test_actual - predictions) ** 2) / np.sum((y_test_actual - np.mean(y_test_actual)) ** 2))

    print(f"Stock: {stock} | MSE: {mse:.5f}, R²: {r2:.5f}")

    # Save evaluation results
    metrics = pd.DataFrame({
        "Stock": [stock],
        "MSE": [mse],
        "R²": [r2],
        "Prediction Start Date": [prediction_start_date.strftime("%Y-%m-%d")]
    })
    metrics_path = os.path.join(METRICS_DIR, f"{stock}_lstm_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    # Save predictions for later analysis
    results_df = pd.DataFrame({
        "Date": df.index[-len(y_test_actual):],
        "Actual": y_test_actual.flatten(),
        "Predicted": predictions.flatten()
    })
    predictions_path = os.path.join(PREDICTIONS_DIR, f"{stock}_lstm_results.csv")
    results_df.to_csv(predictions_path, index=False)

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-len(y_test_actual):], y_test_actual, label="Actual Close Price", color="blue")
    plt.plot(df.index[-len(y_test_actual):], predictions, label="Predicted Close Price", color="red")
    plt.title(f"{stock} Stock Price Prediction using LSTM")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

# Loop Through All Stocks and Train LSTM Models
for stock, filename in TICKERS.items():
    filepath = os.path.join(DATA_DIR, filename)
    custom_lstm_model(stock, filepath)

print("All LSTM models have been trained and saved.")