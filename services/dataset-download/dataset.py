import os
import kagglehub
from google.cloud import storage

KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME_HERE"
KAGGLE_KEY = "YOUR_KAGGLE_KEY_HERE"
GCS_BUCKET = os.getenv("GCS_BUCKET", "mlops-codelab")

storage_client = storage.Client(project="mlops-codelab")

def upload_files(bucket_name, source_folder):
    bucket = storage_client.get_bucket(bucket_name)
    for filename in os.listdir(source_folder):
        blob = bucket.blob(filename)
        blob.upload_from_filename(os.path.join(source_folder, filename))

# Download latest version
path = kagglehub.dataset_download("priyamchoksi/rotten-tomato-movie-reviews-1-44m-rows")

print("Path to dataset files:", path)

upload_files(GCS_BUCKET, path)