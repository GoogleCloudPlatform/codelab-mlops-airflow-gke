import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os

# Constants
MODEL_ID = "google/gemma-2-2b"
TRAIN_DATA_URL = "gs://finetuning-data-bucket/train_data.csv"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, token=os.environ['HF_TOKEN'])
print("Model and tokenizer loaded successfully!")

# Load preprocessed training data
print("Loading dataset...")
data = load_dataset("csv", data_files=TRAIN_DATA_URL)["train"]
data = data.map(lambda samples: tokenizer(samples["reviewText"]), batched=True)

# Formatting function to create prompts and responses
def formatting_func(example):
    prompt = f"What is the review for {example['movie_name']}?"
    response = f"Review: {example['reviewText']} Sentiment: {example['scoreSentiment']}."
    return [prompt + "\n" + response + "<eos>"]

# Set up training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=1000,  # Adjust this based on your dataset size, if needed
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="outputs",
    optim="paged_adamw_8bit"
)

# Set up trainer
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    args=training_args,
    peft_config=bnb_config,
    formatting_func=formatting_func
)

# Start training
trainer.train()
print("Training completed.")
