from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os 
from google.cloud import storage

def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """Upload a local file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")

def train_model(X_train, y_train, model_dir="models/", gcs_bucket_name=None, gcs_model_path=None):
    """
    Train the Random Forest model and save it locally and to GCS.
    """
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved locally at {model_path}")

    # Upload model to GCS
    if gcs_bucket_name and gcs_model_path:
        upload_to_gcs(model_path, gcs_bucket_name, gcs_model_path)

    return model