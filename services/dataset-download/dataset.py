import os
import kagglehub
from google.cloud import storage

KAGGLE_USERNAME = "laurentgrangeau"
KAGGLE_KEY = "c38a65c9f6e37ea0c29f07f078a24764"
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