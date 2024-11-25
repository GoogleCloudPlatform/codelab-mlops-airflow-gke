import os
import pandas as pd
import gcsfs
import json
from datasets import Dataset

# Constants - if environment variables are not provided, take given defaults
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_DATA_NAME = os.getenv("BUCKET_DATA_NAME")

DATASET_NAME = os.getenv("DATASET_NAME", "rotten_tomatoes_movie_reviews.csv")
PREPARED_DATASET_NAME = os.getenv("PREPARED_DATA_URL", "prepared_data.jsonl")
DATASET_LIMIT = int(os.getenv("DATASET_LIMIT", "100"))  # Process a limited number of rows, used 100 during testing phase but can be increased

DATASET_URL = f"gs://{BUCKET_DATA_NAME}/{DATASET_NAME}"
PREPARED_DATASET_URL = f"gs://{BUCKET_DATA_NAME}/{PREPARED_DATASET_NAME}"

# Load the dataset
print(f"Loading dataset from {DATASET_URL}...")

def transform(data):
    question = f"Review analysis for movie '{data['id']}'"
    context = data['reviewText']
    answer = data['scoreSentiment']
    template = "Question: {question}\nContext: {context}\nAnswer: {answer}"
    return {'text': template.format(question=question, context=context, answer=answer)}

try:
    df = pd.read_csv(DATASET_URL, nrows=DATASET_LIMIT)
    print("Dataset loaded successfully.")

    # Drop rows with NaN values in relevant columns
    df = df.dropna(subset=['id', 'reviewText', 'scoreSentiment'])

    # Apply transformation to the DataFrame
    transformed_data = df.apply(transform, axis=1).tolist()

    # Convert transformed data to a DataFrame and then to a Hugging Face Dataset
    transformed_df = pd.DataFrame(transformed_data)
    dataset = Dataset.from_pandas(transformed_df)

    # Print out one preprocessed sample data point
    print(dataset[0])

    # Save the prepared dataset to JSON lines format
    with gcsfs.GCSFileSystem(project=PROJECT_ID).open(PREPARED_DATASET_URL, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Prepared dataset saved to {PREPARED_DATASET_URL}")
    
except Exception as e:
    print(f"Error during data loading or preprocessing: {e}")
    import traceback
    print(traceback.format_exc())