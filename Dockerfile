FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set non-interactive to avoid hanging on apt-get prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install core dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using torch with CUDA 12.1 support, and the latest Hugging Face ecosystem libraries
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories for HF cache and model weights
RUN mkdir -p /root/.cache/huggingface

# Copy the rest of the application code
COPY . .

# Set environment variables for better performance and debugging
ENV PYTHONUNBUFFERED=1
ENV NUMEXPR_MAX_THREADS=4
# HF Hub token can be passed at runtime
# ENV HUGGING_FACE_HUB_TOKEN="your_token_here"

# Default command can be overridden to run training or inference
CMD ["/bin/bash"]
