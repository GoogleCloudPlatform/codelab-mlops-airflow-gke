# Using the NVIDIA CUDA base image
FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

# Install necessary system packages
RUN apt-get update && \
    apt-get -y --no-install-recommends install python3-dev gcc python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install Python packages from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your finetune script into the container
COPY finetune.py .

# Set the environment variable to ensure output is flushed
ENV PYTHONUNBUFFERED 1

# Set the command to run the finetuning script with CUDA device
CMD ["python3", "finetune.py", "--device", "cuda"]



# FROM python:3.11

# WORKDIR /app

# COPY requirements.txt ./

# RUN pip install --no-cache-dir -r requirements.txt

# COPY finetune.py .

# #EXPOSE 8080

# CMD ["python", "finetune.py"]