import mlflow
from mlflow.tracking import MlflowClient
import os

# Set up MLflow environment variables for DagsHub
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/anurag1703/Tobacco-Morality-Analysis-and-Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "anurag1703"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "7dd797a7be400379260b080ab37bcd63b4bc455e"

# Function to set up MLflow and verify connection
def setup_mlflow():
    # Set the tracking URI (already set via environment variable, but can be explicitly done here)
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    # Set the experiment
    mlflow.set_experiment("Tobacco Mortality Analysis")

    # Verify connection and retrieve experiment details
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Tobacco Mortality Analysis")

    if experiment:
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
    else:
        print("Experiment creation failed or the experiment does not exist.")

# Run the setup
if __name__ == "__main__":
    setup_mlflow()
