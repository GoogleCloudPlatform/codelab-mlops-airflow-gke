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
import pandas as pd
import gcsfs
import json
from datasets import Dataset

# Environment variables
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_DATA_NAME = os.getenv("BUCKET_DATA_NAME")

DATASET_NAME = os.getenv("DATASET_NAME", "rotten_tomatoes_movie_reviews.csv")
PREPARED_DATASET_NAME = os.getenv("PREPARED_DATA_URL", "prepared_data.jsonl")
DATASET_LIMIT = int(os.getenv("DATASET_LIMIT"))  # Process a limited number of rows, we use 1000 during testing phase but can be increased

DATASET_URL = f"gs://{BUCKET_DATA_NAME}/{DATASET_NAME}"
PREPARED_DATASET_URL = f"gs://{BUCKET_DATA_NAME}/{PREPARED_DATASET_NAME}"

# Load the dataset
print(f"Loading dataset from {DATASET_URL}...")

def transform(data):
    """
    Transforms a row of the DataFrame into the desired format for fine-tuning.

    Args:
      data: A pandas Series representing a row of the DataFrame.

    Returns:
      A dictionary containing the formatted text.
    """ 
    question = f"Review analysis for movie '{data['id']}'"
    context = data['reviewText']
    answer = data['scoreSentiment']
    template = "Question: {question}\nContext: {context}\nAnswer: {answer}"
    return {'text': template.format(question=question, context=context, answer=answer)}

try:
    df = pd.read_csv(DATASET_URL, nrows=DATASET_LIMIT)
    print(f"Dataset loaded successfully.")

    # Drop rows with NaN values in relevant columns
    df = df.dropna(subset=['id', 'reviewText', 'scoreSentiment'])

    # Apply transformation to the DataFrame
    transformed_data = df.apply(transform, axis=1).tolist()

    # Convert transformed data to a DataFrame and then to a Hugging Face Dataset
    transformed_df = pd.DataFrame(transformed_data)
    dataset = Dataset.from_pandas(transformed_df)

    # Save the prepared dataset to JSON lines format
    with gcsfs.GCSFileSystem(project=GCP_PROJECT_ID).open(PREPARED_DATASET_URL, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Prepared dataset saved to {PREPARED_DATASET_URL}")
    
except Exception as e:
    print(f"Error during data loading or preprocessing: {e}")
    import traceback
    print(traceback.format_exc())
