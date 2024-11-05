import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig

from trl import SFTTrainer
from google.cloud import storage

# Configuration Parameters
PREPARED_DATASET_NAME = os.getenv("PREPARED_DATA_URL", "prepared_data.jsonl")

PREPARED_DATAS_URL = os.getenv("PREPARED_DATA_URL", "gs://finetuning-data-bucket/prepared_data.jsonl")
gcs_bucket = os.getenv("GCS_BUCKET", "finetuning-data-bucket")
new_model_name = os.getenv("NEW_MODEL", "fine_tuned_model")
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2b")

PREPARED_DATASET_URL = f"gs://{BUCKET_DATA_NAME}/{PREPARED_DATASET_NAME}"

# LoRA attention dimension
lora_r = int(os.getenv("LORA_R", 4))

# Alpha parameter for LoRA scaling
lora_alpha = int(os.getenv("LORA_ALPHA", 8))

# Dropout probability for LoRA layers
#LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.1))
lora_dropout = 0.1

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

output_dir = "./output"
num_train_epochs = int(os.getenv("NUM_TRAIN_EPOCHS", 1))

# Enable fp16 training 
fp16 = True

per_device_train_batch_size = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))

# Batch size per GPU for evaluation
per_device_eval_batch_size = int(os.getenv("EVAL_BATCH_SIZE", "2"))

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = int(os.getenv("LOGGING_STEPS", "50"))


# Maximum sequence length to use
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "512"))

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {'':torch.cuda.current_device()}

# Load the prepared dataset from GCS
print(f"Loading dataset from {prepared_data_url}...")
dataset = load_dataset("json", data_files=prepared_data_url, split="train")
print("Dataset loaded successfully.")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16")
        print("=" * 80)

# Load the model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.config.use_cache = False
model.config.pretraining_tp = 1
print("Model loaded successfully.")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
print("Tokenizer loaded successfully!")


# Configure LoRA for fine-tuning
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed.")


""" # Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
) """

# Save the model and tokenizer locally before uploading
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Function to upload the fine-tuned model to GCS
def upload_to_gcs(bucket_name, model_dir):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            gcs_file_path = os.path.relpath(local_file_path, model_dir)
            blob = bucket.blob(os.path.join(new_model_name, gcs_file_path))  # Use new_model_name
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{new_model_name}/{gcs_file_path}")

# Upload the fine-tuned model and tokenizer to GCS
upload_to_gcs(gcs_bucket, output_dir)
print(f"Fine-tuned model {new_model_name} successfully uploaded to GCS.")



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
download_model_from_gcs(gcs_bucket, MODEL_PATH_GCS, MODEL_LOCAL_DIR)

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