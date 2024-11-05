import os
import kagglehub
from google.cloud import storage

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")

def upload_files(bucket_name, source_folder):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    for root, _, files in os.walk(source_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            gcs_file_path = os.path.relpath(local_file_path, source_folder)
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{gcs_file_path}")

# Download latest version
path = kagglehub.dataset_download("priyamchoksi/rotten-tomato-movie-reviews-1-44m-rows")

print("Path to dataset files:", path)

upload_files(GCS_BUCKET, path)
