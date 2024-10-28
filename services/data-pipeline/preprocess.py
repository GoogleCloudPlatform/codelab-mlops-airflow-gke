import pandas as pd
from sklearn.model_selection import train_test_split
import gcsfs
import json
import torch
from transformers import AutoTokenizer
from datasets import Dataset

# Constants
BUCKET_DATA_URL = "gs://finetuning-data-bucket/rotten_tomatoes_movie_reviews.csv"
PREPROCESSED_DATA_URL = "gs://finetuning-data-bucket/preprocessed_data.txt"
TOKENIZED_DATA_URL = "gs://finetuning-data-bucket/tokenized_data.jsonl"
TRAIN_DATA_URL = "gs://finetuning-data-bucket/train_data.jsonl"
TEST_DATA_URL = "gs://finetuning-data-bucket/test_data.jsonl"
MAX_ROWS = 10  # Process a limited number of rows, for testing the pipeline reduced to small amount, increase later
MODEL_ID = "google/gemma-2-2b"

# Load the dataset
print(f"Loading dataset from {BUCKET_DATA_URL}...")

try:
    df = pd.read_csv(BUCKET_DATA_URL, nrows=MAX_ROWS)
    print("Dataset loaded successfully.")

    # Drop rows with NaN values before preprocessing
    df = df.dropna()

    # Construct input text for each review while including the movie name and score
    df['input_text'] = df.apply(
        lambda row: f"Review for '{row['id']}' with score '{row['originalScore']}': {row['reviewText']}",
        axis=1
    )

    # Use the sentiment score as the label for classification
    #df['label'] = df['scoreSentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0, 'fresh': 1, 'rotten': 0})
    df['label'] = df['scoreSentiment']
    
    # Convert to Hugging Face Dataset for easier handling
    dataset = Dataset.from_pandas(df[['input_text', 'label']])

    # Tokenization 
    # Initialize the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Tokenizer loaded successfully!")

    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", max_length=512, truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Save the tokenized dataset to JSON lines format
    tokenized_data = tokenized_dataset.to_dict()
    with gcsfs.GCSFileSystem(project="mlops-airflow2").open(TOKENIZED_DATA_URL, 'w') as f:
        for item in tokenized_data['input_ids']:
            f.write(json.dumps({"input_ids": item}) + "\n")

    print(f"Tokenized dataset saved to {TOKENIZED_DATA_URL}")

    # Split into train and validation datasets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1) 
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Save train and test splits as JSON lines
    train_dataset.to_json(TRAIN_DATA_URL, orient='records', lines=True)
    eval_dataset.to_json(TEST_DATA_URL, orient='records', lines=True)
    
    print("Train and test datasets successfully saved.")
    
except Exception as e:
    print(f"Error during data loading or preprocessing: {e}")
    import traceback
    print(traceback.format_exc())