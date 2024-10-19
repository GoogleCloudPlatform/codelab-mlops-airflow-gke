import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Constants
GS_DATA_URL = 'gs://finetuning-data-bucket/rotten_tomatoes_movie_reviews.csv'

# Load the dataset
df = pd.read_csv(GS_DATA_URL)

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained('gemma-2b')

# Apply tokenization
df['input_ids'] = df['review'].apply(lambda x: tokenizer.encode(x, truncation=True))

# Split the data
train_data, test_data = train_test_split(df, test_size=0.1)
train_data.to_csv(GS_DATA_URL, index=False)
test_data.to_csv(GS_DATA_URL, index=False)
