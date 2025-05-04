# Use NVIDIA CUDA 12.4 base image with Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10 \
    CONDA_DIR=/opt/conda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && ${CONDA_DIR}/bin/conda clean -tipsy

# Add conda to PATH
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Create and activate conda environment
RUN conda create -n gr00t python=${PYTHON_VERSION} -y \
    && echo "source activate gr00t" >> ~/.bashrc

# Make RUN commands use the new environment
SHELL ["/bin/bash", "--login", "-c"]

# Install Python dependencies
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git \
    && cd Isaac-GR00T \
    && pip install --upgrade pip setuptools \
    && pip install -e . \
    && pip install --no-build-isolation flash-attn==2.7.1.post4

# Set the working directory
WORKDIR /workspace/Isaac-GR00T

# Default command (can be overridden)
CMD ["/bin/bash"]