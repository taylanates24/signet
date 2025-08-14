# Signet - Deep Learning Project

This repository contains a deep learning project with support for PyTorch, ONNX, and TensorRT inference. The project includes a Docker environment for easy setup and deployment.

## 🐳 Docker Environment Setup

### Prerequisites

Before building the Docker environment, ensure you have the following installed:

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **NVIDIA Docker**: [Install NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **NVIDIA GPU**: Compatible with CUDA 12.4
- **Docker Compose** (optional): For easier container management
- **Latest NVIDIA driver**

### Building the Docker Image

The Dockerfile is based on NVIDIA's PyTorch container and includes all necessary dependencies for deep learning, ONNX, and TensorRT inference.

#### Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/taylanates24/signet.git
   cd signet
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t signet:latest -f Dockerfile .
   ```

   This will create a Docker image named `signet:latest` with all dependencies installed.

### Running the Docker Container

#### Basic Usage

```bash
# Run the container with GPU support
docker run --gpus all -it signet:latest

# Run with a specific GPU
docker run --gpus '"device=0"' -it signet:latest

# Run with volume mounting for data persistence
docker run --gpus all -v $(pwd):/workspace -it --ipc host signet:latest
```

### Data Management

#### Volume Mounting

Mount your local directories to persist data and models:

```bash
# Mount current directory and data directory
docker run --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -it signet:latest
```

#### Data Directory Structure

The container expects the following directory structure:
```
/workspace/
├── data/           # Training/validation data
├── experiments/    # Experiment outputs
├── *.py           # Python scripts
├── *.onnx         # ONNX models
└── *.trt          # TensorRT models
```


## Datasets

Download the following datasets and place them in the `data/` directory.

- **ICDAR 2011**: [Download](https://drive.google.com/file/d/14v35pUmlbIWq2JbkTc2d8wpIhMibngO4/view?usp=sharing)
- **CEDAR**: [Download](https://drive.google.com/file/d/1iX2blo--6B5Ol55tj6aamP0qnj2OyouM/view?usp=drive_link)

## Creating a Dataset Subset

For faster experimentation and hyperparameter tuning, you can create a smaller subset of your dataset using the `create_subset.py` script.

### How to Use

To create a 10% subset of the training data, run the following command from the root of the project:

```bash
python create_subset.py data/CEDAR/train.csv --subset_size 0.1 --output_dir data/CEDAR_subset --stratify
```

This will create a new file named `train_subset_0.1.csv` inside the `data/CEDAR_subset` directory.

### What is Stratify?

In the context of creating a dataset subset, **stratify** refers to **stratified sampling**. It's a method of sampling that ensures the new, smaller dataset (the subset) has the same proportion of categories or classes (e.g., "genuine" and "forged" signatures) as the original, larger dataset. This is crucial for obtaining reliable results during training and evaluation.

