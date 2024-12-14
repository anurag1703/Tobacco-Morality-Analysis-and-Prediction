import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained models
classification_model = joblib.load("scripts/models/tobacco_mortality_classification_model.pkl")

lstm_model = tf.keras.models.load_model("scripts/models/lstm_smoking_prevalence_model.h5")


# Function to preprocess data for classification
def preprocess_classification_data(data):
    df = pd.DataFrame(data, index=[0])
    features = df[
        [
            "Value_x",
            "Net Ingredient Cost of Bupropion (Zyban)",
            "Net Ingredient Cost of Varenicline (Champix)",
            "Tobacco Affordability",
            "Cessation Success Rate",
            "Smoking Prevalence",
            "Normalized Income (Value_x)",
            "Normalized Cost (Value_y)",
            "Normalized Bupropion Cost",
            "Normalized Varenicline Cost",
        ]
    ]

    # Handle missing values (assuming similar strategy as before)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy="median")
    features_imputed = imputer.fit_transform(features)

    # Scale features within the function
    scaler = StandardScaler() 
    features_scaled = scaler.fit_transform(features_imputed)
    return features_scaled[0]


# Function to preprocess data for LSTM
def preprocess_lstm_data(data, sequence_length):
    df = pd.DataFrame(data, index=[0])
    features = df[["16-24", "25-34", "35-49", "50-59", "60 and Over"]]

    # Scale features within the function
    scaler = MinMaxScaler() 
    features_scaled = scaler.fit_transform(features)

    # Create sequence for prediction
    X = np.array([features_scaled[-sequence_length:]])
    return X


# Streamlit app layout
st.title("Tobacco Mortality and Smoking Prevalence Prediction")

# Navigation between tasks
task = st.radio("Select Task", ("Mortality Risk Prediction", "Smoking Prevalence Prediction"))

if task == "Mortality Risk Prediction":
    st.header("Mortality Risk Prediction")

    # User input fields for classification
    data = {}
    for feature in [
        "Value_x",
        "Net Ingredient Cost of Bupropion (Zyban)",
        "Net Ingredient Cost of Varenicline (Champix)",
        "Tobacco Affordability",
        "Cessation Success Rate",
        "Smoking Prevalence",
        "Normalized Income (Value_x)",
        "Normalized Cost (Value_y)",
        "Normalized Bupropion Cost",
        "Normalized Varenicline Cost",
    ]:
        data[feature] = st.number_input(feature)

    if st.button("Predict Mortality Risk"):
        # Preprocess user input
        processed_data = preprocess_classification_data(data)

        # Make prediction
        prediction = classification_model.predict([processed_data])[0]

        # Map prediction back to original class labels
        class_mapping = {"0": "Low Risk", "1": "Medium Risk", "2": "High Risk", "3": "Very High Risk"}
        predicted_class = class_mapping.get(str(prediction))

        # Display prediction result
        st.write(f"Predicted Mortality Risk: {predicted_class}")

elif task == "Smoking Prevalence Prediction":
    st.header("Smoking Prevalence Prediction")

    # User input fields for LSTM
    data = {}
    for age_group in ["16-24", "25-34", "35-49", "50-59", "60 and Over"]:
        data[age_group] = st.number_input(age_group)

    # Prediction button with sequence length information
    sequence_length = st.text_input("Prediction Sequence Length (Default: 3)", "3")
    sequence_length = int(sequence_length)

    if st.button("Predict Prevalence"):
        # Preprocess user input
        processed_data = preprocess_lstm_data(data, sequence_length)

        # Make prediction
        prediction = lstm_model.predict(processed_data)[0][0]

        # Display prediction result
        st.write(f"Predicted Smoking Prevalence: {prediction}")

# EDA Report Button
if st.button("View EDA Report"):
    st.markdown(open("EDA Report/EDA Report.html", "r").read())
