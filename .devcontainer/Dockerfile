# Use the CUDA 12.6.2 devel base image with cuDNN
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Install only the minimal system packages needed for Cupy and development
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Copy over the environment file (assumes you have an environment.yml at the project root)
COPY environment.yml .

# Create the conda environment from environment.yml (ensure it creates "neuralnet-env")
RUN conda env create -f environment.yml

# Set the default conda environment to neuralnet-env
ENV CONDA_DEFAULT_ENV=neuralnet-env

# Ensure subsequent RUN/CMD instructions use the neuralnet-env environment
SHELL ["conda", "run", "-n", "neuralnet-env", "/bin/bash", "-c"]

# Optionally, install Cupy here if it’s not part of your environment.yml.
# For example, if you need Cupy built for CUDA 12:
# RUN pip install cupy-cuda12x

# Set PYTHONPATH to include the project root
ENV PYTHONPATH="/workspace/NeuralNet:${PYTHONPATH}"

# Set the working directory to the project root inside the container
WORKDIR /workspace/NeuralNet

# Copy all project files into the container (optional if you’re using bind mounts)
COPY . /workspace/NeuralNet/
