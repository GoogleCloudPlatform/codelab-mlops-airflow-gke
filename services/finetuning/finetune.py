# finetune.py

import os
import json
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from google.cloud import storage

# Configuration Parameters
MODEL_ID = os.getenv("MODEL_ID")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
NEW_MODEL_NAME = os.getenv("NEW_MODEL", "fine_tuned_model")
USE_4BIT = os.getenv("USE_4BIT", "True") == "True"
LORA_ALPHA = int(os.getenv("LORA_ALPHA", 16))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.1))
LORA_R = int(os.getenv("LORA_R", 8))
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", 3))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 2))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-5))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", 1000))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", 200))
GCS_BUCKET = os.getenv("GCS_BUCKET")  # Set your GCS bucket name as an env variable
PREPROCESSED_DATA_URL = os.getenv("PREPROCESSED_DATA_URL", "gs://finetuning-data-bucket/preprocessed_data.txt")

# Load model and tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
print("Tokenizer loaded successfully!")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",  # or specify your device map
)

# Load the preprocessed dataset
print(f"Loading preprocessed dataset from {PREPROCESSED_DATA_URL}...")
df_full = pd.read_json(PREPROCESSED_DATA_URL, lines=True)

# Set the training dataset
train_dataset = df_full[['input_ids', 'scoreSentiment']].copy()
train_dataset.rename(columns={'scoreSentiment': 'labels'}, inplace=True)

# Define the LoRA configuration
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap the model with PeftModel to apply LoRA
model = PeftModel(model, peft_config)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,  # Use FP16 for better performance on supported GPUs
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_arguments,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)

# Merge and unload model with LoRA weights
model = PeftModel.from_pretrained(base_model, NEW_MODEL_NAME)
model = model.merge_and_unload()

# Function to upload model to GCS
def upload_to_gcs(bucket_name, model_dir):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            gcs_file_path = os.path.join(model_dir, file)
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to {gcs_file_path}.")

# Upload the fine-tuned model to GCS
upload_to_gcs(GCS_BUCKET, NEW_MODEL_NAME)
