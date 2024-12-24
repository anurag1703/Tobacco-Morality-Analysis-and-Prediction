import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError
import joblib
import os
import matplotlib.pyplot as plt

# Set Streamlit app configuration
st.set_page_config(page_title="Tobacco Analysis App", layout="wide")

# Utility functions
def load_classification_model():
    """Load the classification model."""
    model_path = "scripts/models/tobacco_mortality_classification_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Classification model not found. Ensure the file path is correct and the model exists.")
        return None

def load_lstm_model():
    """Load the LSTM model for smoking prevalence prediction."""
    model_path = "scripts/models/lstm_smoking_prevalence_model.h5"
    try:
        custom_objects = {"mse": MeanSquaredError()}
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
        return f"Error during LSTM prediction: {e}"

# Home Page
def show_home_page():
    """Home page interface."""
    st.title("Welcome to the Tobacco Analysis App")
    st.image("pic.jpg", use_container_width=True)
    st.write("Navigate through the sidebar to explore the app's features:")
    st.markdown("""
        - **Classification**: Predict tobacco-related mortality classes.
        - **Smoking Prevalence Prediction**: Use LSTM to forecast smoking prevalence.
        - **EDA Report**: Explore the detailed Exploratory Data Analysis report.
    """)

# Classification Page
def show_classification_page():
    """Classification task interface."""
    st.title("Tobacco Mortality Prediction")
    st.subheader("Enter the data to predict mortality class:")
    
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

    if st.button("Predict Mortality Class"):
        model = load_classification_model()
        if model:
            input_data = np.array([list(features.values())])
            try:
                prediction = model.predict(input_data)
                st.success(f"Predicted Mortality Class: {int(prediction[0])}")
                
                # Generate a simple bar graph
                st.write("Prediction Visualized:")
                plt.bar(["Actual", "Predicted"], [0, int(prediction[0])], color=["blue", "green"])
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error during classification prediction: {e}")


# LSTM Page
def show_lstm_page():
    """LSTM task interface."""
    st.title("Smoking Prevalence Prediction")
    st.subheader("Enter data for the last 3 time periods:")

    age_groups = ["16-24", "25-34", "35-49", "50-59", "60+"]
    timesteps = 3
    input_data = []

    for t in range(1, timesteps + 1):
        st.write(f"Time Period {t}")
        for age_group in age_groups:
            value = st.number_input(f"{age_group} Smoking Prevalence (Time {t})", value=0.0, step=0.1)
            input_data.append(value)

    if st.button("Predict Smoking Prevalence"):
        model = load_lstm_model()
        if model:
            prediction = predict_with_lstm(model, input_data)
            st.success(f"Predicted Smoking Prevalence: {prediction}")

            # Line graph for actual vs predicted
            st.write("Prediction Visualized:")
            actual = [val for val in input_data]  # Replace with actual values if available
            predicted = [prediction for _ in range(len(actual))]
            plt.plot(range(len(actual)), actual, label="Actual", marker="o")
            plt.plot(range(len(predicted)), predicted, label="Predicted", linestyle="--")
            plt.legend()
            st.pyplot(plt)


# EDA Page
def show_eda_page():
    """EDA Report interface."""
    st.title("Exploratory Data Analysis")
    html_file_path = "EDA Report/EDA Report.html"

    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            report_html = f.read()
        st.components.v1.html(report_html, height=1000, scrolling=True)
    else:
        st.error("EDA report not found. Please check the file path.")


# Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Classification", "Smoking Prevalence Prediction", "EDA Report"])

if options == "Home":
    show_home_page()
elif options == "Classification":
    show_classification_page()
elif options == "Smoking Prevalence Prediction":
    show_lstm_page()
else:
    show_eda_page()
