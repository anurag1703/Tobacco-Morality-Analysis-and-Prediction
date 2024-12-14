import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Set up MLflow environment variables for DagsHub
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/anurag1703/Tobacco-Morality-Analysis-and-Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "anurag1703"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "7dd797a7be400379260b080ab37bcd63b4bc455e"


# Set MLflow tracking URI to connect to the remote Dagshub server
mlflow.set_tracking_uri("https://dagshub.com/anurag1703/Tobacco-Morality-Analysis-and-Prediction.mlflow")

# Start a new MLflow experiment with a specific name
mlflow.set_experiment("Trend Analysis for Smoking Prevalence")

with mlflow.start_run():

    # Log hyperparameters and other details
    mlflow.log_param("sequence_length", 3)
    mlflow.log_param("batch_size", 8)
    mlflow.log_param("epochs", 50)

    # Load dataset
    data_path = r"C:\Users\anura\Desktop\Tobacco Morality Prediction and Analysis\Notebooks\output.csv"  
    df = pd.read_csv(data_path)

    # Features for smoking prevalence
    age_groups = ["16-24", "25-34", "35-49", "50-59", "60 and Over"]
    target = "Smoking Prevalence"

    # Normalize features
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[age_groups + [target]])

    # Convert to supervised learning format
    def create_sequences(data, target_index, sequence_length=3):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length, :-1])  # Features (exclude target)
            y.append(data[i + sequence_length, target_index])  # Target
        return np.array(X), np.array(y)

    sequence_length = 3  # Number of past time steps
    X, y = create_sequences(df_scaled, target_index=-1, sequence_length=sequence_length)

    # Split into train and test sets
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, X.shape[2])),
        tf.keras.layers.Dense(1)  # Single output
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    print(model.summary())

    # Train model
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

    # Log training metrics
    mlflow.log_metric("final_loss", history.history['loss'][-1])
    mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Log evaluation metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Plot predictions vs actual values
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("LSTM Smoking Prevalence Predictions")
    plt.show()

    # Log the model to MLflow
    mlflow.keras.log_model(model, "model")

    # Save the model locally and with DVC
    model.save("lstm_smoking_prevalence_model.h5")
