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
├── sign_data/      # Sign language data
├── experiments/    # Experiment outputs
├── *.py           # Python scripts
├── *.onnx         # ONNX models
└── *.trt          # TensorRT models
```

