import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib
from google.cloud import storage


def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """Upload a local file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")


def preprocess_data(input_path, encoder_dir="encoders/", gcs_bucket_name=None):
    """
    Preprocess the dataset and save the encoders.
    """
    # Load the dataset
    print(f"Loading dataset from {input_path}...")
    data = pd.read_csv(input_path)

    # Drop rows with missing values
    print("Dropping rows with missing values...")
    data.dropna(inplace=True)

    # Separate features and target
    X = data.drop(['Employed', 'Unnamed: 0'], axis=1)
    y = data['Employed']

    # Specify categorical and text columns
    categorical_cols = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Country']
    text_col = 'HaveWorkedWith'

    # Define and fit vectorizer
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    vectorizer.fit(X[text_col])

    # Define and fit one-hot encoder
    print("Fitting OneHotEncoder...")
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoder.fit(X[categorical_cols])

    # Transform data
    print("Transforming data...")
    text_transformed = vectorizer.transform(X[text_col])
    categorical_transformed = one_hot_encoder.transform(X[categorical_cols])

    # Combine transformed data
    from scipy.sparse import hstack
    X_transformed = hstack([text_transformed, categorical_transformed])

    # Save encoders
    os.makedirs(encoder_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(encoder_dir, "vectorizer.pkl"))
    joblib.dump(one_hot_encoder, os.path.join(encoder_dir, "one_hot_encoder.pkl"))
    print(f"Encoders saved locally in {encoder_dir}")

    # Upload encoders to GCS
    if gcs_bucket_name:
        upload_to_gcs(os.path.join(encoder_dir, "vectorizer.pkl"), gcs_bucket_name, "encoders/vectorizer.pkl")
        upload_to_gcs(os.path.join(encoder_dir, "one_hot_encoder.pkl"), gcs_bucket_name, "encoders/one_hot_encoder.pkl")

    return X_transformed, y