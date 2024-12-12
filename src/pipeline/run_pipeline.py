from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
import os
import pandas as pd
from src.data.preprocess import preprocess_data
from src.models.train import train_model
from src.models.tune_hyperparams import tune_random_forest
from src.models.evaluate import evaluate_and_save
from google.cloud import storage
from src.models.evaluate import evaluate_and_save
from sklearn.model_selection import train_test_split


def upload_directory_to_gcs(local_dir, bucket_name, gcs_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_dir, relative_path)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")

def verify_gcs_dataset(bucket_name, blob_name, local_path="/tmp/temp_dataset.csv"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded dataset from GCS to {local_path}")

    # Load and verify dataset contents
    data = pd.read_csv(local_path)
    print("Columns in the downloaded dataset:", data.columns.tolist())
    return data

# Dummy HTTP server to satisfy Cloud Run's port requirement
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def start_http_server():
    """Start a simple HTTP server for health checks."""
    port = int(os.getenv("PORT", 8080))  # Default Cloud Run port is 8080
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"Starting HTTP server on port {port}...")
    server.serve_forever()

def verify_processed_data(processed_data_path):
    data = pd.read_csv(processed_data_path)
    print("Columns in the processed dataset:", data.columns.tolist())
    print("Sample rows from the processed dataset:")
    print(data.head())

def run_pipeline():
    """Main logic for running the ML pipeline."""
    # Paths and configurations
    raw_data_path = "gs://jobapplicationprediction/data/raw/stackoverflow_full.csv"
    encoder_dir = "encoders/"
    model_dir = "models/"
    gcs_bucket_name = "jobapplicationprediction"
    gcs_model_path = "models/best_model.pkl"
    gcs_metrics_path = "metrics/best_model_metrics.json"

    # Preprocess data
    print("Step 1: Preprocessing data...")
    X, y = preprocess_data(raw_data_path, encoder_dir, gcs_bucket_name)

    # Split data into train and test
    print("Step 2: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Hyperparameter tuning
    print("Step 3: Hyperparameter tuning...")
    #best_xgb = tune_xgboost(X_train, y_train)
    #best_catboost = tune_catboost(X_train, y_train)
    best_rf = tune_random_forest(X_train, y_train)

    # Evaluate the best models
    print("Step 4: Evaluating the best models...")
    models = {
        #"XGBoost": best_xgb,
        #"CatBoost": best_catboost,
        "Random Forest": best_rf,
    }
    best_model_name, best_model = evaluate_and_save(
        models,
        X_test,
        y_test,
        model_dir,
        gcs_bucket_name,
        gcs_model_path,
        gcs_metrics_path,
    )

    print(f"Pipeline completed successfully! Best Model: {best_model_name}")


if __name__ == "__main__":
    run_pipeline()