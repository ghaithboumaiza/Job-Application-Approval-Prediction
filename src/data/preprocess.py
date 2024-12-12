import pandas as pd
from google.cloud import storage
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download a file from GCS to the local container."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")


def preprocess_data(input_path, output_path, encoder_dir="encoders/", scaler_dir="scaler/"):
    # Load the dataset
    data = pd.read_csv(input_path)

    # Drop unnecessary columns (adjust based on your new dataset)
    if 'app_id' in data.columns:
        data.drop('app_id', axis=1, inplace=True)

    # Fill missing values
    for col in data.select_dtypes(include='object').columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
    for col in data.select_dtypes(include='number').columns:
        data[col].fillna(data[col].median(), inplace=True)

    # Encode categorical variables and save encoders
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'employed']
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    # Save encoders
    os.makedirs(encoder_dir, exist_ok=True)
    for col, encoder in encoders.items():
        joblib.dump(encoder, os.path.join(encoder_dir, f"{col}_encoder.pkl"))

    # Scale numerical features and save scaler
    scaler = StandardScaler()
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['employed']]
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Save scaler
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.pkl'))

    # Save processed data
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")