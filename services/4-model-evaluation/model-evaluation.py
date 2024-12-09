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
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig

from trl import SFTTrainer
from google.cloud import storage

# OPTIONAL: Test finetuned model manually via a test prompt
# Configuration for GCS and local paths
MODEL_PATH_GCS = "fine_tuned_model"     # GCS directory where model is saved
MODEL_LOCAL_DIR = "./temp_model"        # Local directory for temporary model storage
TEST_PROMPT = "How is the movie beavers?"

# Initialize GCS client and download model from GCS
def download_model_from_gcs(bucket_name, model_gcs_path, local_dir):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_gcs_path)
    
    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        local_file_path = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")

# Download the model from GCS
download_model_from_gcs(BUCKET_DATA_NAME, MODEL_PATH_GCS, MODEL_LOCAL_DIR)

# Load the tokenizer and model from the local directory
print("Loading tokenizer and model from local directory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL_DIR)
print("Model loaded successfully.")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize the input prompt and generate a response
inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,              # Adjust max length based on prompt size
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )

# Decode and print the response
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", generated_text)
