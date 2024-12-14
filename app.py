import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError
import joblib
import os

# Set Streamlit app configuration
st.set_page_config(page_title="Tobacco Analysis App", layout="wide")

# Utility functions
def load_classification_model():
    """Load the classification model."""
    model_path = "scripts/models/tobacco_mortality_classification_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    st.error("Classification model not found.")
    return None

def load_lstm_model():
    """Load the LSTM model for smoking prevalence prediction."""
    model_path = "scripts/models/lstm_smoking_prevalence_model.h5"
     # Register the 'mse' loss/metric explicitly
    custom_objects = {
        "mse": MeanSquaredError(),
    }
    try:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        return None

def predict_with_lstm(model, age_inputs):
    """Predict using the LSTM model."""
    try:
        input_data = np.array(age_inputs, dtype=float)
        timesteps = 3  # Number of timesteps used in training
        num_features = len(age_inputs) // timesteps
        input_data_reshaped = input_data.reshape((1, timesteps, num_features))
        prediction = model.predict(input_data_reshaped)
        return prediction[0][0]  # Scalar prediction
    except Exception as e:
        return f"Error during prediction: {e}"

def show_classification_page():
    """Classification task interface."""
    st.title("Tobacco Mortality Classification")

    # Input fields for the classification task
    st.subheader("Enter the required inputs for classification:")
    features = {
        "Value_x": st.number_input("Value_x (Normalized Income)", value=0.0, step=0.1),
        "Net Ingredient Cost of Bupropion (Zyban)": st.number_input("Net Ingredient Cost of Bupropion", value=0.0, step=0.1),
        "Net Ingredient Cost of Varenicline (Champix)": st.number_input("Net Ingredient Cost of Varenicline", value=0.0, step=0.1),
        "Tobacco Affordability": st.number_input("Tobacco Affordability", value=0.0, step=0.1),
        "Cessation Success Rate": st.number_input("Cessation Success Rate", value=0.0, step=0.1),
        "Smoking Prevalence": st.number_input("Smoking Prevalence", value=0.0, step=0.1),
        "Normalized Income (Value_x)": st.number_input("Normalized Income", value=0.0, step=0.1),
        "Normalized Cost (Value_y)": st.number_input("Normalized Cost", value=0.0, step=0.1),
        "Normalized Bupropion Cost": st.number_input("Normalized Bupropion Cost", value=0.0, step=0.1),
        
    }

    # Predict button
    if st.button("Predict Class"):
        model = load_classification_model()
        if model:
            input_data = np.array([list(features.values())])
            try:
                prediction = model.predict(input_data)
                st.success(f"Predicted Mortality Class: {int(prediction[0])}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

def show_lstm_page():
    """LSTM task interface."""
    st.title("Smoking Prevalence Prediction with LSTM")

    # Input fields for LSTM task
    st.subheader("Enter the smoking prevalence data for the last 3 timesteps:")
    age_groups = ["16-24", "25-34", "35-49", "50-59", "60 and Over"]
    timesteps = 3
    input_data = []

    for t in range(1, timesteps + 1):
        st.write(f"Timestep {t}")
        for age_group in age_groups:
            value = st.number_input(f"Age group {age_group} (Timestep {t})", value=0.0, step=0.1)
            input_data.append(value)

    # Predict button
    if st.button("Predict Smoking Prevalence"):
        model = load_lstm_model()
        if model:
            prediction = predict_with_lstm(model, input_data)
            st.success(f"Predicted Smoking Prevalence: {prediction}")

def show_eda_page():
    """EDA Report interface."""
    st.title("Exploratory Data Analysis Report")

    # Path to the EDA report
    html_file_path = "EDA Report/EDA Report.html"

    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            report_html = f.read()
        st.components.v1.html(report_html, height=1000, scrolling=True)
    else:
        st.error("EDA report not found.")

# Main app logic
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Classification", "LSTM for Smoking Prevalence", "EDA Report"])

if options == "Classification":
    show_classification_page()
elif options == "LSTM for Smoking Prevalence":
    show_lstm_page()
else:
    show_eda_page()
