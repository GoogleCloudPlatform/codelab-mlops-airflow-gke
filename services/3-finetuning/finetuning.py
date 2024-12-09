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
import bitsandbytes
from accelerate import Accelerator
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from google.cloud import storage

# Environment variables
BUCKET_DATA_NAME = os.getenv("BUCKET_DATA_NAME")
PREPARED_DATA_URL = os.getenv("PREPARED_DATA_URL", "prepared_data.jsonl")
# Finetuned model name
new_model = os.getenv("NEW_MODEL_NAME", "fine_tuned_model")
# Base model from the Hugging Face hub
model_name = os.getenv("MODEL_ID", "google/gemma-2-9b-it")
# Root path for saving the finetuned model
save_model_path = os.getenv("MODEL_PATH", "./output")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded successfully!")

# Load dataset
EOS_TOKEN = tokenizer.eos_token
dataset = load_dataset(
    "json", data_files=f"gs://{BUCKET_DATA_NAME}/{PREPARED_DATA_URL}", split="train")
print(dataset)

################################################################################
# LoRA parameters
################################################################################
# LoRA attention dimension
lora_r = int(os.getenv("LORA_R", "8"))
# Alpha parameter for LoRA scaling
lora_alpha = int(os.getenv("LORA_ALPHA", "16"))
# Dropout probability for LoRA layers
lora_dropout = float(os.getenv("LORA_DROPOUT", "0.1"))

################################################################################
# TrainingArguments parameters
################################################################################
# Number of training epochs
num_train_epochs = int(os.getenv("EPOCHS", 1))
# Set fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
# Batch size per GPU for training
per_device_train_batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
# Batch size per GPU for evaluation
per_device_eval_batch_size = 1
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
# Save strategy: steps, epoch, no
save_strategy = os.getenv("CHECKPOINT_SAVE_STRATEGY", "steps")
# Save total limit of checkpoints
save_total_limit = int(os.getenv("CHECKPOINT_SAVE_TOTAL_LIMIT", "5"))
# Save checkpoint every X updates steps
save_steps = int(os.getenv("CHECKPOINT_SAVE_STEPS", "1000"))
# Log every X updates steps
logging_steps = 50

################################################################################
# SFT parameters
################################################################################
# Maximum sequence length to use
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "512"))
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load base model
print(f"Loading base model started")
model = AutoModelForCausalLM.from_pretrained(
    attn_implementation="eager",
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float16,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
print("Loading base model completed")

# Configure fine-tuning with LoRA
print(f"Configuring fine tuning started")
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# Set training parameters
training_arguments = SFTConfig(
        bf16=bf16,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False, 
        },
        dataset_text_field="text",
        disable_tqdm=True,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=group_by_length,
        log_on_each_node=False,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        max_seq_length=max_seq_length,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        optim=optim,
        output_dir=save_model_path,
        packing=packing,
        per_device_train_batch_size=per_device_train_batch_size,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
    )

print(f"Configuring fine tuning completed")

# Initialize the SFTTrainer
print(f"Creating trainer started")
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

print(f"Creating trainer completed")

# Finetune the model
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed.")

# Save the fine-tuned model
print("Saving new model started")
trainer.model.save_pretrained(new_model)
print("Saving new model completed")

# Merge LoRA weights with the base model
print(f"Merging the new model with base model started")
base_model = AutoModelForCausalLM.from_pretrained(
    low_cpu_mem_usage=True,
    pretrained_model_name_or_path=model_name,
    return_dict=True,
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(
    model=base_model,
    model_id=new_model,
)
model = model.merge_and_unload()

print(f"Merging the new model with base model completed")

accelerator = Accelerator()
print(f"Accelerate unwrap model started")
unwrapped_model = accelerator.unwrap_model(model)
print(f"Accelerate unwrap model completed")

print(f"Save unwrapped model started")
unwrapped_model.save_pretrained(
    is_main_process=accelerator.is_main_process,
    save_directory=save_model_path,
    save_function=accelerator.save,
)
print(f"Save unwrapped model completed")

print(f"Save new tokenizer started")
if accelerator.is_main_process:
    tokenizer.save_pretrained(save_model_path)
print(f"Save new tokenizer completed")

# Upload the model to GCS
def upload_to_gcs(bucket_name, model_dir):
    """Uploads a directory to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            gcs_file_path = os.path.relpath(local_file_path, model_dir)
            blob = bucket.blob(os.path.join(new_model, gcs_file_path))  # Use new_model_name
            blob.upload_from_filename(local_file_path)

# Upload the fine-tuned model and tokenizer to GCS
upload_to_gcs(BUCKET_DATA_NAME, save_model_path)
print(f"Fine-tuned model {new_model} successfully uploaded to GCS.")
