# * Copyright 2022 Google LLC
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

import os
import kagglehub
from google.cloud import storage

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
BUCKET_DATA_NAME = os.getenv("BUCKET_DATA_NAME")

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

upload_blob(BUCKET_DATA_NAME, source_file_name, destination_blob_name)
