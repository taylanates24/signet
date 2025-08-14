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


## Hyperparameter Search

Hyperparameter tuning is the process of finding the optimal combination of hyperparameters for a machine learning model to achieve the best performance. This project includes a script, `tune.py`, that uses [Optuna](https://optuna.org/) to automate this process.

### How it Works

The tuning script works in two main phases:

1.  **Grid Search:** It first explores a predefined set of hyperparameter combinations to quickly find promising regions in the search space.
2.  **Bayesian Optimization:** It then uses a more sophisticated Bayesian optimization algorithm (TPE) to intelligently search for the best hyperparameters, focusing on the most promising areas.

### How to Use

To start the hyperparameter tuning process, run the `tune.py` script from your terminal. You can customize the search by providing various command-line arguments.

**Note on Tuning Time:** Hyperparameter tuning can be a lengthy process, as it involves training the model multiple times with different parameter combinations. To accelerate the search, it is highly recommended to first run the tuning script on a smaller subset of the dataset. While the optimal parameters for a subset may not be identical to those for the full dataset, this approach provides a much faster way to identify a strong set of baseline hyperparameters. You can use create_subset.py script for this prupose.

Here is a basic example of how to run the script:

```bash
python tune.py --dataset sign_data --output_dir my_tuning_runs
```

This command will start the tuning process on the `sign_data` dataset, and all the results will be saved in a unique, timestamped directory inside `my_tuning_runs`.

### Parameters Explained

Here is a detailed explanation of the most important arguments you can use to customize the tuning process:

*   `--lr` (float, default: `1e-4`): The learning rate determines how much the model's weights are updated during training. A smaller value can lead to more accurate results but may take longer to train, while a larger value can speed up training but may risk overshooting the optimal solution.
*   `--batch_size` (int, default: `16`): The batch size is the number of samples processed before the model is updated. A larger batch size can lead to faster training but requires more memory, while a smaller batch size can provide a regularizing effect but may lead to a more unstable training process.
*   `--weight_decay` (float, default: `2e-4`): Weight decay is a regularization technique that helps prevent overfitting by adding a penalty to the loss function. This encourages the model to use smaller weights, which can lead to better generalization.
*   `--dropout_conv_p` (float, default: `0.2`): The dropout probability for convolutional layers. Dropout is a regularization technique where randomly selected neurons are ignored during training, which helps prevent overfitting. This parameter controls the dropout rate for the convolutional layers of the model.
*   `--dropout_fc_p` (float, default: `0.3`): The dropout probability for fully connected layers. This works similarly to `dropout_conv_p`, but it is applied to the fully connected layers of the model.
*   `--random_trial_coeff` (float, default: `1.2`): This coefficient determines the number of initial grid search trials. The total number of grid search trials is calculated as `len(param_combinations) * random_trial_coeff`. This allows you to control the exploration phase of the tuning process.
*   `--bayesian_trial_coeff` (float, default: `1.2`): This coefficient determines the total number of Bayesian optimization trials. The total number of trials is calculated as `start_trials * bayesian_trial_coeff`. This allows you to control the exploitation phase of the tuning process.
*   `--output_dir` (str, default: `tuning_results`): The directory where all the tuning results will be saved. Each run will create a unique, timestamped subdirectory inside this directory to store the Optuna database, the best hyperparameters found, and a summary of the results.
*   `--params_to_search` (list, default: `['lr', 'batch_size', 'dropout_conv_p', 'dropout_fc_p', 'weight_decay']`): Specifies the list of hyperparameters to tune. This allows you to conduct targeted experiments by focusing the search on a specific subset of parameters. Any hyperparameter not included in this list will be fixed to its default value during the tuning process.

By using these parameters, you can customize the hyperparameter tuning process to fit your specific needs and find the best combination of hyperparameters for your model.

## PyTorch Inference

The `inference.py` script allows you to run inference directly with a trained PyTorch model (`.pt` file). This is useful for quick testing and evaluation without needing to convert the model to another format.

**Key Features:**

- **Direct PyTorch Inference**: Runs inference using the original PyTorch model, which is great for debugging and validation.
- **Single Pair and Batch Prediction**: Supports both single image pair predictions and batch predictions.
- **Threshold Finding**: Includes a function to find the optimal classification threshold by evaluating the model on a dataset. This helps in tuning the model for the best balance between False Accept Rate (FAR) and False Reject Rate (FRR).
- **Dataset Evaluation**: Can evaluate the model on a dataset to compute key performance metrics.

**How to Use:**

To run inference on a pair of images:

```bash
python inference.py \
    --model_path model.pt \
    --image1 data/CEDAR/full_org/original_1_1.png \
    --image2 data/CEDAR/full_forg/forgeries_1_1.png \
    --threshold 0.0641
```

To find the optimal threshold on a dataset:

```bash
python inference.py \
    --model_path model.pt \
    --dataset data/CEDAR \
    --find_optimal_threshold
```

**Arguments:**

- `--model_path`: Path to the trained PyTorch model (`.pt` file).
- `--image1`, `--image2`: Paths to the two signature images to compare.
- `--threshold`: The distance threshold for classification.
- `--dataset`: Path to the dataset directory for evaluation or threshold finding.
- `--batch_size`: The batch size to use for evaluation. Default is `1`.
- `--img_height`, `--img_width`: The dimensions to which the input images will be resized. These arguments override any dimensions stored in the model checkpoint, allowing you to experiment with different input sizes. Defaults are `155` and `220`.
- `--find_optimal_threshold`: If specified, the script will find the optimal threshold on the provided dataset.
- `--threshold_min`, `--threshold_max`, `--num_thresholds`: These parameters control the search for the optimal threshold. They define the range and the number of steps for the search.

## ONNX Conversion and Inference

This project supports ONNX for cross-platform model inference. The following sections detail the process of converting the PyTorch model to ONNX, and running inference with the ONNX model.

### ONNX Conversion

The `convert_to_onnx.py` script handles the conversion of a trained PyTorch model (a `.pt` file) into the ONNX format.

**Key Features:**

- **Dynamic Axes Support**: The script can export the model with dynamic axes, allowing for variable batch sizes during inference. This is controlled by the `--dynamic_axes` flag.
- **Verification**: After conversion, the script can verify the ONNX model by comparing its output with the original PyTorch model to ensure consistency. This is enabled by default and can be controlled with the `--verify` flag.
- **Inference Test**: The script can run a quick inference test with a pair of sample images to confirm that the ONNX model is working correctly.

**How to Use:**

To convert a PyTorch model to ONNX, use the following command:

```bash
python convert_to_onnx.py \
    --model_path experiments/your_experiment/best_model.pt \
    --output_path model.onnx \
    --dynamic_axes
```

**Arguments:**

- `--model_path`: Path to the input PyTorch model (`.pt` file).
- `--output_path`: Path to save the output ONNX model (`.onnx` file).
- `--batch_size`: Batch size for the model. Default is `1`.
- `--img_height`, `--img_width`: Image dimensions. Defaults are `155` and `220`.
- `--opset_version`: The ONNX opset version to use. Default is `19`.
- `--dynamic_axes`: Enables dynamic axes for variable batch sizes.
- `--device`: The device to use for conversion (`cpu` or `cuda`). Default is `cuda`.
- `--verify`: Verifies the ONNX model against the PyTorch model.
- `--test_inference`: Runs a test inference with sample images.

### ONNX Inference

The `onnx_inference.py` script allows you to run inference using a converted ONNX model.

**Key Features:**

- **Single Pair and Batch Prediction**: The script supports both single image pair predictions and batch predictions.
- **Benchmarking**: Includes a benchmarking mode to measure the performance of the ONNX model, including average inference time and throughput (FPS).

**How to Use:**

To run inference on a pair of images:

```bash
python onnx_inference.py \
    --model_path model.onnx \
    --image1 data/CEDAR/full_org/original_1_1.png \
    --image2 data/CEDAR/full_forg/forgeries_1_1.png \
    --threshold 0.0641
```

To run a benchmark:

```bash
python onnx_inference.py --model_path model.onnx --benchmark
```

**Arguments:**

- `--model_path`: Path to the ONNX model file.
- `--image1`, `--image2`: Paths to the two signature images to compare.
- `--threshold`: The distance threshold for classifying signatures as "genuine" or "forged".
- `--benchmark`: Runs a performance benchmark.
- `--evaluate_dataset`: Path to a dataset for evaluation.
- `--output_json`: Path to save the results in a JSON file.

## TensorRT Conversion and Inference

For maximum performance on NVIDIA GPUs, this project supports TensorRT. The following sections describe how to convert an ONNX model to a TensorRT engine and how to run inference with it.

### TensorRT Conversion

To convert an ONNX model to a TensorRT engine, you can use the `trtexec` command-line tool, which is included with the TensorRT installation in the Docker environment. This tool optimizes the model and creates a highly efficient engine for inference.

**How to Use:**

Use the following command to convert an ONNX model to a TensorRT engine with FP16 precision, which offers a good balance between performance and accuracy:

```bash
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

**Arguments:**

- `--onnx`: Specifies the path to the input ONNX model.
- `--saveEngine`: Specifies the path to save the output TensorRT engine (`.trt` file).
- `--fp16`: Enables FP16 precision for faster inference. For the highest possible precision, you can omit this flag to use FP32.

### TensorRT Inference

The `tensorrt_inference.py` script is used to run inference with a TensorRT engine. It is optimized for low latency and high throughput.

**Key Features:**

- **High-Performance Inference**: Leverages TensorRT for the fastest possible inference on NVIDIA GPUs.
- **Batch Processing**: Supports batch inference to process multiple image pairs at once, maximizing GPU utilization.
- **Benchmarking**: Includes a robust benchmarking tool to measure performance with different batch sizes.

**How to Use:**

To run inference on a single pair of images:

```bash
python tensorrt_inference.py \
    --engine_path model.trt \
    --image1 data/CEDAR/full_org/original_1_1.png \
    --image2 data/CEDAR/full_forg/forgeries_1_1.png \
    --threshold 0.0641
```

To run a benchmark with different batch sizes:

```bash
python tensorrt_inference.py --engine_path model.trt --benchmark --benchmark_batch_sizes 1 8 16
```

**Arguments:**

- `--engine_path`: Path to the TensorRT engine file (`.trt`).
- `--image1`, `--image2`: Paths to the two signature images to compare.
- `--threshold`: The distance threshold for classification.
- `--benchmark`: Runs a performance benchmark.
- `--benchmark_batch_sizes`: A list of batch sizes to use for benchmarking.
- `--evaluate_dataset`: Path to a dataset for evaluation.
- `--output_json`: Path to save the results in a JSON file.

