import os
import kagglehub
from google.cloud import storage

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

# Download latest version
path = kagglehub.dataset_download("priyamchoksi/rotten-tomato-movie-reviews-1-44m-rows")

print("Path to dataset files:", path)
destination_blob_name = "rotten_tomatoes_movie_reviews.csv"
source_file_name = f"{path}/{destination_blob_name}"

upload_blob(GCS_BUCKET, source_file_name, destination_blob_name)
