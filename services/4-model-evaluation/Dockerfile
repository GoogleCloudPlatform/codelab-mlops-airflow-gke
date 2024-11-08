# Using the NVIDIA CUDA base image
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Install necessary system packages
RUN apt-get update && \
    apt-get -y --no-install-recommends install python3-dev gcc python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy your finetune script into the container
COPY model-evaluation.py .

# Set the environment variable to ensure output is flushed
ENV PYTHONUNBUFFERED 1
ENV MODEL_ID "google/gemma-2-2b"
ENV GCS_BUCKET "finetuning-data-bucket"
 
# Set the command to run the finetuning script with CUDA device
CMD ["python3", "model-evaluation.py", "--device", "cuda"]
