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
ENV MODEL_ID "google/gemma-2-2b"
ENV GCS_BUCKET "finetuning-data-bucket"
 
# Set the command to run the finetuning script with CUDA device
CMD ["python3", "finetune.py", "--device", "cuda"]

# FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

# RUN apt-get update && \
#     apt-get -y --no-install-recommends install python3-dev gcc python3-pip git && \
#     rm -rf /var/lib/apt/lists/*

# RUN pip3 install --no-cache-dir \
#     accelerate==0.30.1 bitsandbytes==0.43.1 \
#     datasets==2.19.1 transformers==4.41.0 \
#     peft==0.11.1 trl==0.8.6 torch==2.3.0

# COPY finetune.py /finetune.py

# ENV PYTHONUNBUFFERED 1

# CMD python3 /finetune.py --device cuda