import os
import gradio as gr
import joblib
import pandas as pd
from google.cloud import storage
import csv

# Configuration
GCS_BUCKET_NAME = "jobapplicationprediction"  # Update for the project
BEST_MODEL_PATH = "models/best_model.pkl"     # Path to the model in GCS
GCS_ENCODER_DIR = "encoders/"                # Path in GCS for encoders
GCS_SAVED_INFERENCES_PATH = "feedback/saved_inferences.csv"  # Path in GCS for saved inferences
SAVED_INFERENCES_PATH = "/tmp/saved_inferences.csv"          # Local path to save inferences
LOCAL_MODEL_PATH = "/tmp/best_model.pkl"     # Local path for the model
LOCAL_ENCODER_DIR = "/tmp/encoders/"         # Local directory for encoders

# Ensure the saved inferences file exists
if not os.path.exists(SAVED_INFERENCES_PATH):
    pd.DataFrame(columns=[
        "HaveWorkedWith", "Age", "Accessibility", "EdLevel", "Gender", 
        "MentalHealth", "MainBranch", "Country", "Prediction"
    ]).to_csv(SAVED_INFERENCES_PATH, index=False)

def save_inference(data, prediction):
    """
    Save inference data and prediction to a CSV file.
    """
    header = [
        "HaveWorkedWith", "Age", "Accessibility", "EdLevel", "Gender",
        "MentalHealth", "MainBranch", "Country", "Prediction"
    ]
    file_exists = os.path.isfile(SAVED_INFERENCES_PATH)
    with open(SAVED_INFERENCES_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)  # Write header if file doesn't exist
        writer.writerow(data + [prediction])

def upload_to_gcs(local_file_path, bucket_name, gcs_file_path):
    """
    Upload a file to Google Cloud Storage.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to gs://{bucket_name}/{gcs_file_path}")

def download_from_gcs(bucket_name, source_path, destination_path):
    """
    Download a file from Google Cloud Storage.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)
    print(f"Downloaded {source_path} to {destination_path}")

def download_directory_from_gcs(bucket_name, gcs_dir, local_dir):
    """
    Download all files from a GCS directory to a local directory.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_dir)
    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        if not blob.name.endswith('/'):
            local_path = os.path.join(local_dir, os.path.relpath(blob.name, gcs_dir))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")

def load_resources():
    """
    Download and load the model and encoders from GCS.
    """
    download_from_gcs(GCS_BUCKET_NAME, BEST_MODEL_PATH, LOCAL_MODEL_PATH)
    model = joblib.load(LOCAL_MODEL_PATH)
    print("Model loaded.")

    download_directory_from_gcs(GCS_BUCKET_NAME, GCS_ENCODER_DIR, LOCAL_ENCODER_DIR)
    encoders = {}
    vectorizer_path = os.path.join(LOCAL_ENCODER_DIR, "vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)
    encoders["HaveWorkedWith"] = vectorizer

    one_hot_encoder_path = os.path.join(LOCAL_ENCODER_DIR, "one_hot_encoder.pkl")
    one_hot_encoder = joblib.load(one_hot_encoder_path)
    encoders["categorical"] = one_hot_encoder

    print("Encoders loaded.")
    return model, encoders

def predict(HaveWorkedWith, Age, Accessibility, EdLevel, Gender, MentalHealth, MainBranch, Country):
    """
    Perform prediction on user input.
    """
    model, encoders = load_resources()

    input_data = pd.DataFrame(
        [[HaveWorkedWith, Age, Accessibility, EdLevel, Gender, MentalHealth, MainBranch, Country]],
        columns=["HaveWorkedWith", "Age", "Accessibility", "EdLevel", "Gender", "MentalHealth", "MainBranch", "Country"],
    )

    # Apply text vectorizer for "HaveWorkedWith"
    vectorizer = encoders["HaveWorkedWith"]
    text_vectorized = vectorizer.transform(input_data["HaveWorkedWith"])

    # Apply one-hot encoder for categorical features
    one_hot_encoder = encoders["categorical"]
    categorical_data = one_hot_encoder.transform(input_data[["Age", "Accessibility", "EdLevel", "Gender", "MentalHealth", "MainBranch", "Country"]])

    # Combine processed data
    combined_data = pd.concat([pd.DataFrame(text_vectorized.toarray()), pd.DataFrame(categorical_data.toarray())], axis=1)

    # Perform prediction
    prediction = model.predict(combined_data)
    result = "Approved" if prediction[0] == 1 else "Not Approved"

    save_inference(
        [HaveWorkedWith, Age, Accessibility, EdLevel, Gender, MentalHealth, MainBranch, Country],
        result,
    )

    return result

def feedback():
    """
    Display and allow modification of saved inferences.
    """
    if os.path.exists(SAVED_INFERENCES_PATH):
        saved_data = pd.read_csv(SAVED_INFERENCES_PATH)
    else:
        saved_data = pd.DataFrame(columns=[
            "HaveWorkedWith", "Age", "Accessibility", "EdLevel", "Gender", 
            "MentalHealth", "MainBranch", "Country", "Prediction"
        ])
        saved_data.to_csv(SAVED_INFERENCES_PATH, index=False)

    def update_feedback(updated_data):
        updated_data.to_csv(SAVED_INFERENCES_PATH, index=False)
        upload_to_gcs(SAVED_INFERENCES_PATH, GCS_BUCKET_NAME, GCS_SAVED_INFERENCES_PATH)
        return "Feedback data saved to cloud!"

    return gr.Interface(
        fn=update_feedback,
        inputs=gr.DataFrame(value=saved_data, interactive=True),
        outputs="text",
        title="Feedback",
        description="Modify predictions and save updates to the cloud."
    )

if __name__ == "__main__":
    predict_tab = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(label="HaveWorkedWith"),
            gr.Dropdown(["Low", "Medium", "High"], label="Age"),
            gr.Dropdown(["Yes", "No"], label="Accessibility"),
            gr.Dropdown(["High School", "Bachelor's", "Master's", "PhD"], label="EdLevel"),
            gr.Dropdown(["Male", "Female", "Other"], label="Gender"),
            gr.Dropdown(["Yes", "No"], label="MentalHealth"),
            gr.Dropdown(["Software Development", "Data Science", "Other"], label="MainBranch"),
            gr.Textbox(label="Country"),
        ],
        outputs="text",
        title="Job Application Approval Prediction",
        description="Enter application details to predict approval."
    )

    feedback_tab = feedback()

    gr.TabbedInterface(
        [predict_tab, feedback_tab],
        ["Prediction", "Feedback"]
    ).launch(server_name="0.0.0.0", server_port=8080)