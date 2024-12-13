import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os


# Set up MLflow environment variables for DagsHub
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/anurag1703/Tobacco-Morality-Analysis-and-Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "anurag1703"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "7dd797a7be400379260b080ab37bcd63b4bc455e"

# Load dataset
data_path = r"C:\Users\anura\Desktop\Tobacco Morality Prediction and Analysis\Notebooks\output.csv"
df = pd.read_csv(data_path)

# Define bins and labels for classification
bins = [0, 1.5e9, 1.8e9, 2.0e9, np.inf]
labels = [0, 1, 2, 3]  # Very Low, Low, Medium, High
df["Value_y_binned"] = pd.cut(df["Value_y"], bins=bins, labels=labels, include_lowest=True)

# Check class distribution
print("Class distribution after binning:")
print(df["Value_y_binned"].value_counts())

# Define features and target
target = "Value_y_binned"
features = df.drop(columns=["Year", "Value_y", target])
y = pd.factorize(df[target])[0]

# Handle missing values and scale features
features.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy="median")
features_imputed = imputer.fit_transform(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, y, test_size=0.2, random_state=42)

# MLflow experiment setup
mlflow.set_tracking_uri("https://dagshub.com/anurag1703/Tobacco-Morality-Analysis-and-Prediction.mlflow")
mlflow.set_experiment("Tobacco Mortality Analysis")

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
}

# Train and log models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)

        # Log metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_params", model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Save the best model
best_model = LogisticRegression(max_iter=500, class_weight="balanced")
best_model.fit(X_train, y_train)
os.makedirs("./models", exist_ok=True)
joblib.dump(best_model, "./models/tobacco_mortality_classification_model.pkl")
mlflow.sklearn.log_model(best_model, "best_model")

print("Best model saved and logged to MLflow.")
