# Job Application Approval Prediction Pipeline

## Overview
This repository implements an end-to-end Machine Learning pipeline to predict job application approvals. It leverages preprocessing, hyperparameter tuning, training, and model evaluation, with support for deployment on Google Cloud.

## Features
- **Data Preprocessing:** Handles missing values, encodes categorical features, and scales numerical data.
- **Hyperparameter Tuning:** Supports XGBoost, CatBoost, and Random Forest with RandomizedSearchCV.
- **Model Training and Evaluation:** Trains multiple models and evaluates them to select the best-performing one.
- **Cloud Integration:** Fetches raw data from Google Cloud Storage (GCS) and uploads models and metrics back to GCS.
- **Health Check API:** Includes a simple HTTP server for Cloud Run deployment.

## Project Structure
```
Job_Application_Approval/
|-- data/
|   |-- raw/                # Raw data files
|   |-- processed/          # Preprocessed data
|-- encoders/               # Encoders for categorical variables
|-- models/                 # Trained models
|-- scaler/                 # Scalers for numerical features
|-- src/
|   |-- data/
|   |   |-- preprocess.py   # Preprocessing logic
|   |-- models/
|   |   |-- evaluate.py     # Model evaluation logic
|   |   |-- train.py        # Model training logic
|   |   |-- tune_hyperparams.py  # Hyperparameter tuning
|   |-- pipeline/
|   |   |-- run_pipeline.py # Main pipeline script
|-- README.md
```

## Setup

### Prerequisites
- Python 3.8+
- Google Cloud SDK installed and authenticated
- A GCS bucket with appropriate permissions

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Job_Application_Approval
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Preprocess Data
Download raw data from GCS, preprocess it, and save the processed dataset locally:
```bash
python src/data/preprocess.py
```

### 2. Train Models
Train a logistic regression model:
```bash
python src/models/train.py
```

### 3. Hyperparameter Tuning
Tune hyperparameters for XGBoost, CatBoost, and Random Forest:
```bash
python src/models/tune_hyperparams.py
```

### 4. Evaluate Models
Evaluate trained models and save the best one:
```bash
python src/models/evaluate.py
```

### 5. Run the Pipeline
Execute the entire pipeline, including preprocessing, training, tuning, and evaluation:
```bash
python src/pipeline/run_pipeline.py
```

## Environment Variables
For deployment on Google Cloud Run, set the following environment variables:
- `PORT`: Port for the HTTP server (default: 8080).
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to the GCP service account key.

## Google Cloud Integration

### Uploading Data to GCS
Use the provided script to upload data to your GCS bucket:
```bash
python -c "from src.utils import upload_to_gcs; upload_to_gcs('/path/to/data.csv', 'your-bucket-name', 'data/raw/data.csv')"
```

### Running on Cloud Run
1. Build a Docker image:
   ```bash
   docker build -t job-application-approval .
   ```
2. Push the image to Google Container Registry:
   ```bash
   docker tag job-application-approval gcr.io/<project-id>/job-application-approval
   docker push gcr.io/<project-id>/job-application-approval
   ```
3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy job-application-approval --image gcr.io/<project-id>/job-application-approval --platform managed
   ```

## Key Scripts

### `src/data/preprocess.py`
- Downloads raw data from GCS.
- Preprocesses the data: fills missing values, encodes categorical variables, and scales numerical features.
- Saves processed data locally and uploads encoders/scalers to GCS.

### `src/models/tune_hyperparams.py`
- Tunes hyperparameters for XGBoost, CatBoost, and Random Forest using RandomizedSearchCV.
- Prints the best parameters and saves the best models.

### `src/models/evaluate.py`
- Evaluates multiple models on test data using metrics like accuracy, precision, recall, F1 score, and AUC.
- Saves the best model and its metrics locally and uploads them to GCS.

### `src/pipeline/run_pipeline.py`
- Runs the complete ML pipeline, including preprocessing, training, tuning, and evaluation.
- Includes a health check HTTP server for Cloud Run deployment.

## Monitoring and Logging
- Metrics (e.g., accuracy, precision, recall) are saved as JSON files.
- System logs can be monitored via Cloud Run logs.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any changes.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For any inquiries or feedback, please contact me 
