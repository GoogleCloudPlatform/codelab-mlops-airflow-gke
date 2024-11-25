import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig

from trl import SFTTrainer
from google.cloud import storage

# Configuration Parameters
BUCKET_DATA_NAME = os.getenv("BUCKET_DATA_NAME")
PREPARED_DATASET_NAME = os.getenv("PREPARED_DATA_URL", "prepared_data.jsonl")

NEW_MODEL_NAME = os.getenv("NEW_MODEL_NAME", "fine_tuned_model")
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
print(f"Loading dataset from {PREPARED_DATASET_URL}...")
dataset = load_dataset("json", data_files=PREPARED_DATASET_URL, split="train")
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
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.config.use_cache = False
model.config.pretraining_tp = 1
print("Model loaded successfully.")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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
            blob = bucket.blob(os.path.join(NEW_MODEL_NAME, gcs_file_path))  # Use new_model_name
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{NEW_MODEL_NAME}/{gcs_file_path}")

# Upload the fine-tuned model and tokenizer to GCS
upload_to_gcs(BUCKET_DATA_NAME, output_dir)
print(f"Fine-tuned model {NEW_MODEL_NAME} successfully uploaded to GCS.")