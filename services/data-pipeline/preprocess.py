import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import gcsfs

# Constants
BUCKET_DATA_URL = "gs://finetuning-data-bucket/rotten_tomatoes_movie_reviews.csv"  # Update this path for local or GCS
PREPROCESSED_DATA_URL = "gs://finetuning-data-bucket/preprocessed_data.txt"

# Load the dataset
df = pd.read_csv(BUCKET_DATA_URL)

# Dropping duplicate rows from the dataframe, if any
num_dup = df.duplicated().sum()
if num_dup > 0:
    df.drop_duplicates(inplace=True)
    print(f"Dropped {num_dup} duplicate rows from the dataframe.")
else:
    print(f"No duplicate rows found in the dataframe.")
print(f"Now the dataframe has {len(df)} rows.")

# Select the relevant columns for fine-tuning: "reviewText" and "scoreSentiment"
df = df[['reviewText', 'scoreSentiment']]

# Dropping rows where 'reviewText' or 'scoreSentiment' are missing (NaN)
initial_length = len(df)
df.dropna(subset=['reviewText', 'scoreSentiment'], inplace=True)
rows_dropped_na = initial_length - len(df)
print(f"Dropped {rows_dropped_na} rows due to missing 'reviewText' or 'scoreSentiment'.")

# Check and remove rows where 'reviewText' is empty (after trimming whitespace)
df['reviewText'] = df['reviewText'].str.strip()  # Remove leading/trailing spaces
df = df[df['reviewText'] != '']
rows_dropped_empty = initial_length - len(df)
print(f"Dropped {rows_dropped_empty} rows where 'reviewText' was empty after trimming.")


# Now the dataframe is clean and ready for tokenization

# Tokenizer initialization (Check max_length for Gemma-2b)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Apply tokenization (using padding, truncation, and max_length as necessary)
df["input_ids"] = df["reviewText"].apply(
    lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=512)["input_ids"]
)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.1)

# Save preprocessed data in JSON lines format
with open("preprocessed_data.txt", "w") as f:
    for _, row in train_data.iterrows():
        # Assuming 'reviewText' and 'scoreSentiment' columns exist
        f.write(
            '{"text": "' + row["reviewText"] + '", "label": "' + str(row["scoreSentiment"])
            + '", "input_ids": ' + str(row["input_ids"]) + "}\n"
        )

# Optional: Upload preprocessed data to GCS
fs = gcsfs.GCSFileSystem(project="mlops-airflow2")
with fs.open(PREPROCESSED_DATA_URL, 'w') as f:
    for _, row in train_data.iterrows():
        f.write(
            '{"text": "' + row["reviewText"] + '", "label": "' + str(row["scoreSentiment"])
            + '", "input_ids": ' + str(row["input_ids"]) + "}\n"
        )

# Save train and test splits as CSVs (optional)
train_data.to_csv("gs://finetuning-data-bucket/train_data.csv", index=False)
test_data.to_csv("gs://finetuning-data-bucket/test_data.csv", index=False)
