import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset
from google.cloud import storage

# Configuration Parameters
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2-2b")
TOKENIZED_DATA_URL = os.getenv("TOKENIZED_DATA_URL", "gs://finetuning-data-bucket/tokenized_data.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
NEW_MODEL_NAME = os.getenv("NEW_MODEL", "fine_tuned_model")
GCS_BUCKET = os.getenv("GCS_BUCKET", "finetuning-data-bucket")
USE_4BIT = os.getenv("USE_4BIT", "True") == "True"
LORA_ALPHA = int(os.getenv("LORA_ALPHA", 16))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.1))
LORA_R = int(os.getenv("LORA_R", 8))
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", 3))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 2))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-5))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", 1000))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", 200))

# Load the model and tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Tokenizer loaded successfully!")

# Load the pre-trained model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
print("Model loaded successfully.")

# Load the tokenized dataset from GCS
print(f"Loading tokenized dataset from {TOKENIZED_DATA_URL}...")
tokenized_data = load_dataset("json", data_files=TOKENIZED_DATA_URL, split="train")
print("Tokenized dataset loaded successfully.")

# Configure LoRA for fine-tuning
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",  # Fine-tuning a causal language model
)

# Apply LoRA to the model
model = PeftModel(model, peft_config)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True if torch.cuda.is_available() else False,
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="input_ids",
)

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed.")

# Save the fine-tuned model locally
model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print(f"Fine-tuned model saved to {NEW_MODEL_NAME}.")

# Upload the fine-tuned model to GCS
def upload_to_gcs(bucket_name, model_dir):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            gcs_file_path = os.path.relpath(local_file_path, model_dir)
            blob = bucket.blob(os.path.join(model_dir, gcs_file_path))
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{gcs_file_path}")

upload_to_gcs(GCS_BUCKET, NEW_MODEL_NAME)
print("Fine-tuned model successfully uploaded to GCS.")