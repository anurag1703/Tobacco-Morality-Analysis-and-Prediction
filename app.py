import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# Set up the Streamlit app
st.set_page_config(page_title="Tobacco Mortality Analysis", layout="wide")


# Utility functions
def load_classification_model():
    model_path = "scripts/models/tobacco_mortality_classification_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Classification model not found. Please ensure the model file exists.")
        return None


def load_lstm_model():
    model_path = "scripts/models/lstm_smoking_prevalence_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("LSTM model not found. Please ensure the model file exists.")
        return None


def show_classification_page():
    st.title("Tobacco Mortality Classification")
    st.subheader("Predict Mortality Risk Categories")

    # Input fields
    inputs = {
        "Value_x": st.number_input("Value_x (Normalized Income)", value=0.0, format="%.2f"),
        "Net Ingredient Cost of Bupropion (Zyban)": st.number_input("Bupropion Cost", value=0.0, format="%.2f"),
        "Net Ingredient Cost of Varenicline (Champix)": st.number_input("Varenicline Cost", value=0.0, format="%.2f"),
        "Tobacco Affordability": st.number_input("Tobacco Affordability", value=0.0, format="%.2f"),
        "Cessation Success Rate": st.number_input("Cessation Success Rate", value=0.0, format="%.2f"),
        "Smoking Prevalence": st.number_input("Smoking Prevalence", value=0.0, format="%.2f"),
        "Normalized Income (Value_x)": st.number_input("Normalized Income", value=0.0, format="%.2f"),
        "Normalized Cost (Value_y)": st.number_input("Normalized Cost", value=0.0, format="%.2f"),
        "Normalized Bupropion Cost": st.number_input("Normalized Bupropion Cost", value=0.0, format="%.2f"),
        
    }

    if st.button("Classify"):
        model = load_classification_model()
        if model:
            input_data = np.array([[inputs[key] for key in inputs]])
            prediction = model.predict(input_data)
            st.success(f"Predicted Mortality Risk Category: {int(prediction[0])}")


def show_lstm_page():
    st.title("Smoking Prevalence Trend Analysis")
    st.subheader("Analyze Trends Using LSTM Model")

    # Input sequence
    st.write("Provide past smoking prevalence data:")
    age_groups = ["16-24", "25-34", "35-49", "50-59", "60 and Over"]
    sequence_length = 3

    past_data = {}
    for group in age_groups:
        past_data[group] = [
            st.number_input(f"{group} (Step {i+1})", value=0.0, format="%.2f") for i in range(sequence_length)
        ]

    if st.button("Predict Smoking Prevalence"):
        model = load_lstm_model()
        if model:
            # Prepare input data
            scaler = MinMaxScaler()
            past_values = np.array([past_data[group] for group in age_groups]).T
            past_values_scaled = scaler.fit_transform(past_values)

            input_sequence = past_values_scaled.reshape(1, sequence_length, len(age_groups))
            prediction = model.predict(input_sequence)
            predicted_value = scaler.inverse_transform(
                np.concatenate([np.zeros((1, len(age_groups))), prediction], axis=1)
            )[-1, -1]

            st.success(f"Predicted Smoking Prevalence: {predicted_value:.2f}")


def show_eda_page():
    st.title("Exploratory Data Analysis Report")
    eda_file_path = "EDA Report/EDA Report.html"

    if os.path.exists(eda_file_path):
        with open(eda_file_path, "r", encoding="utf-8") as f:
            eda_content = f.read()
        st.components.v1.html(eda_content, height=1000, scrolling=True)
    else:
        st.error("EDA report not found. Please ensure the file exists.")


# Navigation logic
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose Section:", ["Classification", "Smoking Prevalence", "EDA Report"]
)

if section == "Classification":
    show_classification_page()
elif section == "Smoking Prevalence":
    show_lstm_page()
elif section == "EDA Report":
    show_eda_page()
