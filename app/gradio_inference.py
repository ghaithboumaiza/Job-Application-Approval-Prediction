import os
import gradio as gr
import joblib
import pandas as pd
from google.cloud import storage
import csv

# Configuration
GCS_BUCKET_NAME = "jobapplicationprediction"  # Update for the new project
BEST_MODEL_PATH = "models/best_model.pkl"     # Path to the model in GCS
GCS_ENCODER_DIR = "encoders/"                # Path in GCS for encoders
GCS_SCALER_PATH = "scaler/scaler.pkl"        # Path in GCS for scaler
SAVED_INFERENCES_PATH = "/tmp/saved_inferences.csv"  # Local path to save inferences
GCS_SAVED_INFERENCES_PATH = "feedback/saved_inferences.csv"  # Path in GCS for saved inferences
LOCAL_MODEL_PATH = "/tmp/best_model.pkl"     # Local path for the model
LOCAL_ENCODER_DIR = "/tmp/encoders/"         # Local directory for encoders
LOCAL_SCALER_PATH = "/tmp/scaler.pkl"        # Local path for scaler

# Ensure the saved inferences file exists
if not os.path.exists(SAVED_INFERENCES_PATH):
    pd.DataFrame(columns=[
        "Education", "Experience", "Referred", "Applied_Previously", "Skill_Score", "Prediction"
    ]).to_csv(SAVED_INFERENCES_PATH, index=False)

def save_inference(data, prediction):
    """
    Save inference data and prediction to a CSV file.
    """
    header = [
        "Education", "Experience", "Referred", "Applied_Previously", "Skill_Score", "Prediction"
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
    Download and load the model, encoders, and scaler from GCS.
    """
    download_from_gcs(GCS_BUCKET_NAME, BEST_MODEL_PATH, LOCAL_MODEL_PATH)
    model = joblib.load(LOCAL_MODEL_PATH)
    print("Model loaded.")

    download_directory_from_gcs(GCS_BUCKET_NAME, GCS_ENCODER_DIR, LOCAL_ENCODER_DIR)
    encoders = {}
    for col in ['Education', 'Referred', 'Applied_Previously']:
        encoder_path = os.path.join(LOCAL_ENCODER_DIR, f"{col}_encoder.pkl")
        encoders[col] = joblib.load(encoder_path)
    print("Encoders loaded.")

    download_from_gcs(GCS_BUCKET_NAME, GCS_SCALER_PATH, LOCAL_SCALER_PATH)
    scaler = joblib.load(LOCAL_SCALER_PATH)
    print("Scaler loaded.")

    return model, encoders, scaler

def predict(Education, Experience, Referred, Applied_Previously, Skill_Score):
    """
    Perform prediction on user input.
    """
    model, encoders, scaler = load_resources()

    input_data = pd.DataFrame(
        [[Education, Experience, Referred, Applied_Previously, Skill_Score]],
        columns=["Education", "Experience", "Referred", "Applied_Previously", "Skill_Score"],
    )

    # Apply encoders
    categorical_cols = ["Education", "Referred", "Applied_Previously"]
    for col in categorical_cols:
        encoder = encoders[col]
        input_data[col] = encoder.transform([input_data[col][0]])

    # Convert numerical inputs to float and scale them
    input_data[["Experience", "Skill_Score"]] = scaler.transform(input_data[["Experience", "Skill_Score"]])

    # Perform prediction
    prediction = model.predict(input_data)
    result = "Approved" if prediction[0] == 1 else "Not Approved"

    save_inference([Education, Experience, Referred, Applied_Previously, Skill_Score], result)

    return result

def feedback():
    """
    Display and allow modification of saved inferences.
    """
    if os.path.exists(SAVED_INFERENCES_PATH):
        try:
            saved_data = pd.read_csv(SAVED_INFERENCES_PATH)
        except pd.errors.ParserError:
            saved_data = pd.DataFrame(columns=[
                "Education", "Experience", "Referred", "Applied_Previously", "Skill_Score", "Prediction"
            ])
            saved_data.to_csv(SAVED_INFERENCES_PATH, index=False)
    else:
        saved_data = pd.DataFrame(columns=[
            "Education", "Experience", "Referred", "Applied_Previously", "Skill_Score", "Prediction"
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
            gr.Dropdown(["Yes", "No"], label="Education"),
            gr.Number(label="Experience"),
            gr.Dropdown(["Yes", "No"], label="Referred"),
            gr.Dropdown(["Yes", "No"], label="Applied Previously"),
            gr.Number(label="Skill Score"),
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