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
COPY finetuning.py .

# Set the command to run the finetuning script with CUDA device
CMD ["python3", "finetuning.py"]
