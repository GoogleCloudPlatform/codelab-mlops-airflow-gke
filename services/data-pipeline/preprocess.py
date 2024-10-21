import pandas as pd
from sklearn.model_selection import train_test_split
import gcsfs
import json
from transformers import AutoTokenizer

# Constants
BUCKET_DATA_URL = "gs://finetuning-data-bucket/rotten_tomatoes_movie_reviews.csv"  # GCS Path
PREPROCESSED_DATA_URL = "gs://finetuning-data-bucket/preprocessed_data.txt"
CHUNKSIZE = 10000  # Adjust to control memory usage
MAX_ROWS = 100000  # Process a limited number of rows
TOKENIZER_MODEL = "google/gemma-2-2b"

# Initialize the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
print("Tokenizer loaded successfully!")

# Function to process and tokenize each chunk
def process_chunk(df_chunk):
    print(f"Processing a chunk of size: {len(df_chunk)} rows")
    
    # Select relevant columns
    df_chunk = df_chunk[['id', 'reviewText', 'scoreSentiment']].dropna()

    # Tokenize review text
    df_chunk["input_ids"] = df_chunk["reviewText"].apply(
        lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=512)["input_ids"]
    )
    
    # Convert to JSON lines format
    preprocessed_data = []
    for _, row in df_chunk.iterrows():
        json_data = {
            "movie_name": row["id"],
            "reviewText": row["reviewText"],
            "scoreSentiment": row["scoreSentiment"],
            "input_ids": row["input_ids"]
        }
        preprocessed_data.append(json.dumps(json_data) + "\n")

    return preprocessed_data

# Load and process the dataset in chunks
print(f"Loading dataset from {BUCKET_DATA_URL}...")
preprocessed_data = []
df_iterator = pd.read_csv(BUCKET_DATA_URL, chunksize=CHUNKSIZE)
total_rows_processed = 0

for df_chunk in df_iterator:
    preprocessed_data.extend(process_chunk(df_chunk))
    total_rows_processed += len(df_chunk)
    if total_rows_processed >= MAX_ROWS:
        break

print(f"Processed {total_rows_processed} rows. Saving preprocessed data...")

# Save preprocessed data
fs = gcsfs.GCSFileSystem(project="your_project_id")
with fs.open(PREPROCESSED_DATA_URL, 'w') as f:
    f.writelines(preprocessed_data)

print(f"Preprocessed data saved to {PREPROCESSED_DATA_URL}")

# Split the data into training and testing sets
df_full = pd.DataFrame([json.loads(x) for x in preprocessed_data])
train_data, test_data = train_test_split(df_full, test_size=0.1, random_state=42)

# Save train and test splits as CSVs
train_data.to_csv("gs://finetuning-data-bucket/train_data.csv", index=False)
test_data.to_csv("gs://finetuning-data-bucket/test_data.csv", index=False)
print("Train and test splits successfully saved.")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer
# import gcsfs
# import json


# # Constants
# BUCKET_DATA_URL = "gs://finetuning-data-bucket/rotten_tomatoes_movie_reviews.csv"  # GCS Path
# PREPROCESSED_DATA_URL = "gs://finetuning-data-bucket/preprocessed_data.txt"
# CHUNKSIZE = 10000  # Adjust this chunk size to control memory usage
# MAX_ROWS = 100000  # TO CHECK: Container keeps crashing after some time. Testing with smaller data set

# # Initialize the tokenizer
# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
# print("Tokenizer loaded successfully!")

# # Create an empty list to store tokenized data
# preprocessed_data = []

# # Function to process each chunk
# def process_chunk(df_chunk):
#     print(f"Processing a chunk of size: {len(df_chunk)} rows")
    
#     # Dropping duplicate rows
#     num_dup = df_chunk.duplicated().sum()
#     if num_dup > 0:
#         df_chunk = df_chunk.drop_duplicates()  # Avoid inplace modification
#         print(f"Dropped {num_dup} duplicate rows from the dataframe chunk.")
#     else:
#         print("No duplicate rows found in the chunk.")

#     # Select the relevant columns for fine-tuning: "reviewText" and "scoreSentiment"
#     df_chunk = df_chunk[['reviewText', 'scoreSentiment']]
#     print("Selected 'reviewText' and 'scoreSentiment' columns.")

#     # Dropping rows where 'reviewText' or 'scoreSentiment' are missing (NaN)
#     initial_length = len(df_chunk)
#     df_chunk = df_chunk.dropna(subset=['reviewText', 'scoreSentiment'])  # Avoid inplace modification
#     rows_dropped_na = initial_length - len(df_chunk)
#     print(f"Dropped {rows_dropped_na} rows due to missing 'reviewText' or 'scoreSentiment'.")

#     # Check and remove rows where 'reviewText' is empty after trimming whitespace
#     initial_length = len(df_chunk)
#     df_chunk.loc[:, 'reviewText'] = df_chunk['reviewText'].str.strip()  # Use .loc[] to avoid SettingWithCopyWarning
#     df_chunk = df_chunk[df_chunk['reviewText'] != '']
#     rows_dropped_empty = initial_length - len(df_chunk)
#     print(f"Dropped {rows_dropped_empty} rows where 'reviewText' was empty after trimming.")

#     # Tokenize the text in the 'reviewText' column
#     print("Starting tokenization of chunk...")
#     df_chunk["input_ids"] = df_chunk["reviewText"].apply(
#         lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=512)["input_ids"]
#     )
#     print("Tokenization completed for this chunk.")

#     # Convert the chunk to the required JSON format and add it to the preprocessed data
#     for _, row in df_chunk.iterrows():
#         json_data = {
#             "text": row["reviewText"],
#             "label": str(row["scoreSentiment"]),
#             "input_ids": row["input_ids"]
#             }
        
#         preprocessed_data.append(json.dumps(json_data) + "\n")  # Serialize each row as JSON

#     print(f"Finished processing chunk of size: {len(df_chunk)}")

# # Load the dataset in chunks and process each chunk
# print(f"Loading dataset in chunks from {BUCKET_DATA_URL}...")

# # Count the total number of rows processed
# total_rows_processed = 0

# df_iterator = pd.read_csv(BUCKET_DATA_URL, chunksize=CHUNKSIZE)
# for i, df_chunk in enumerate(df_iterator):
#     print(f"Processing chunk {i + 1}")
#     process_chunk(df_chunk)
#     total_rows_processed += len(df_chunk)

#     # Break the loop if the desired number of rows has been processed
#     if total_rows_processed >= MAX_ROWS:
#         break

# print("All chunks processed. Saving preprocessed data...")

# # Save the preprocessed data in JSON lines format
# with open("preprocessed_data.txt", "w") as f:
#     f.writelines(preprocessed_data)
# print("Preprocessed data saved to preprocessed_data.txt")

# # Optional: Upload the preprocessed data to GCS
# print(f"Uploading preprocessed data to {PREPROCESSED_DATA_URL}...")
# fs = gcsfs.GCSFileSystem(project="mlops-airflow2")
# with fs.open(PREPROCESSED_DATA_URL, 'w') as f:
#     f.writelines(preprocessed_data)
# print(f"Preprocessed data successfully uploaded to {PREPROCESSED_DATA_URL}")

# # Split the data into training and testing sets
# print("Splitting data into training and testing sets...")

# # Use json.loads() to safely parse the preprocessed data
# df_full = pd.DataFrame([json.loads(x) for x in preprocessed_data])
# print(f"Successfully parsed {len(df_full)} rows from preprocessed data.")

# # Perform the train-test split
# train_data, test_data = train_test_split(df_full, test_size=0.1, random_state=42)
# print(f"Data split completed. Training set size: {len(train_data)}, Test set size: {len(test_data)}")

# # Save train and test splits as CSVs (optional)
# print("Saving train and test data splits to GCS...")
# train_data.to_csv("gs://finetuning-data-bucket/train_data.csv", index=False)
# test_data.to_csv("gs://finetuning-data-bucket/test_data.csv", index=False)
# print("Train and test splits successfully saved to GCS.")
