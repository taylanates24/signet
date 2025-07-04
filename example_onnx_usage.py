#!/usr/bin/env python3
"""
Example usage scripts for ONNX SigNet inference
This file demonstrates various ways to use the ONNX model for signature verification
"""

from onnx_inference import ONNXSigNetInference
import requests
import json
import base64
import time
from PIL import Image
import os

def example_basic_inference():
    """Basic example of ONNX inference"""
    print("=== Basic ONNX Inference Example ===")
    
    # Initialize the inferencer
    inferencer = ONNXSigNetInference(
        model_path='signet.onnx',
        threshold=0.5
    )
    
    # Test with sample images
    image1_path = 'data/CEDAR/full_org/original_38_1.png'
    image2_path = 'data/CEDAR/full_org/original_38_3.png'
    
    if os.path.exists(image1_path) and os.path.exists(image2_path):
        result = inferencer.predict_pair(image1_path, image2_path)
        
        print(f"Image 1: {image1_path}")
        print(f"Image 2: {image2_path}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    else:
        print("Sample images not found. Please check the paths.")

def example_batch_inference():
    """Example of batch inference"""
    print("\n=== Batch ONNX Inference Example ===")
    
    inferencer = ONNXSigNetInference(
        model_path='signet.onnx',
        threshold=0.5
    )
    
    # Create sample image pairs
    image_pairs = [
        ('data/CEDAR/full_org/original_38_1.png', 'data/CEDAR/full_org/original_38_3.png'),
        ('data/CEDAR/full_org/original_38_1.png', 'data/CEDAR/full_forg/forgeries_38_13.png'),
    ]
    
    # Filter pairs that exist
    valid_pairs = []
    for img1, img2 in image_pairs:
        if os.path.exists(img1) and os.path.exists(img2):
            valid_pairs.append((img1, img2))
    
    if valid_pairs:
        results = inferencer.predict_batch(valid_pairs)
        
        print(f"Processed {results['summary']['total_pairs']} pairs:")
        print(f"- Genuine predictions: {results['summary']['genuine_predictions']}")
        print(f"- Forged predictions: {results['summary']['forged_predictions']}")
        print(f"- Average distance: {results['summary']['avg_distance']:.4f}")
        print(f"- Average time per pair: {results['summary']['avg_time_per_pair_ms']:.2f} ms")
        
        # Print individual results
        for i, result in enumerate(results['results']):
            print(f"\nPair {i+1}:")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Prediction: {result['prediction']}")
    else:
        print("No valid image pairs found.")

def example_performance_benchmark():
    """Example of performance benchmarking"""
    print("\n=== Performance Benchmark Example ===")
    
    inferencer = ONNXSigNetInference(
        model_path='signet.onnx',
        threshold=0.5
    )
    
    # Run benchmark
    benchmark_results = inferencer.benchmark(num_iterations=50)
    
    print(f"Benchmark Results ({benchmark_results['num_iterations']} iterations):")
    print(f"- Average inference time: {benchmark_results['avg_inference_time_ms']:.2f} ms")
    print(f"- Min/Max time: {benchmark_results['min_inference_time_ms']:.2f}/{benchmark_results['max_inference_time_ms']:.2f} ms")
    print(f"- Standard deviation: {benchmark_results['std_inference_time_ms']:.2f} ms")
    print(f"- Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
    print(f"- Execution providers: {benchmark_results['providers']}")

def example_pil_image_inference():
    """Example using PIL Image objects directly"""
    print("\n=== PIL Image Inference Example ===")
    
    inferencer = ONNXSigNetInference(
        model_path='signet.onnx',
        threshold=0.5
    )
    
    image_path = 'data/CEDAR/full_org/original_38_1.png'
    
    if os.path.exists(image_path):
        # Load image with PIL
        image1 = Image.open(image_path)
        image2 = Image.open(image_path)  # Same image for demonstration
        
        result = inferencer.predict_pair(image1, image2)
        
        print(f"Using PIL Images (same image twice):")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Prediction: {result['prediction']}")
        print(f"Should be genuine (same image): {result['prediction'] == 'genuine'}")
    else:
        print("Sample image not found.")

def example_api_client():
    """Example of using the REST API"""
    print("\n=== REST API Client Example ===")
    
    base_url = "http://localhost:5000"
    
    # Check if API is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print("API server is not running. Start it with: python onnx_api.py")
            return
    except requests.exceptions.RequestException:
        print("API server is not running. Start it with: python onnx_api.py")
        return
    
    # Example 1: Single prediction with file paths
    print("\n1. Single prediction with file paths:")
    payload = {
        "image1": "data/CEDAR/full_org/original_38_1.png",
        "image2": "data/CEDAR/full_org/original_38_3.png",
        "threshold": 0.5
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=payload)
        result = response.json()
        
        if result.get('success'):
            pred = result['result']
            print(f"  Distance: {pred['distance']:.4f}")
            print(f"  Prediction: {pred['prediction']}")
            print(f"  Inference time: {pred['inference_time_ms']:.2f} ms")
        else:
            print(f"  Error: {result.get('error')}")
    except Exception as e:
        print(f"  Request failed: {e}")
    
    # Example 2: Get model info
    print("\n2. Model information:")
    try:
        response = requests.get(f"{base_url}/model_info")
        result = response.json()
        
        if result.get('success'):
            info = result['model_info']
            print(f"  Model path: {info['model_path']}")
            print(f"  Providers: {info['providers']}")
            print(f"  Threshold: {info['threshold']}")
            print(f"  Image dimensions: {info['image_dimensions']}")
        else:
            print(f"  Error: {result.get('error')}")
    except Exception as e:
        print(f"  Request failed: {e}")
    
    # Example 3: Benchmark
    print("\n3. API Benchmark:")
    try:
        response = requests.post(f"{base_url}/benchmark", json={"iterations": 20})
        result = response.json()
        
        if result.get('success'):
            bench = result['benchmark']
            print(f"  Average time: {bench['avg_inference_time_ms']:.2f} ms")
            print(f"  Throughput: {bench['throughput_fps']:.1f} FPS")
        else:
            print(f"  Error: {result.get('error')}")
    except Exception as e:
        print(f"  Request failed: {e}")

def example_base64_api():
    """Example using base64 encoded images with API"""
    print("\n=== Base64 API Example ===")
    
    base_url = "http://localhost:5000"
    
    # Check if API is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print("API server is not running.")
            return
    except requests.exceptions.RequestException:
        print("API server is not running.")
        return
    
    image_path = 'data/CEDAR/full_org/original_38_1.png'
    
    if not os.path.exists(image_path):
        print("Sample image not found.")
        return
    
    # Convert image to base64
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Create data URL
    data_url = f"data:image/png;base64,{base64_image}"
    
    payload = {
        "image1": data_url,
        "image2": data_url,  # Same image
        "threshold": 0.5
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=payload)
        result = response.json()
        
        if result.get('success'):
            pred = result['result']
            print(f"Base64 prediction (same image):")
            print(f"  Distance: {pred['distance']:.4f}")
            print(f"  Prediction: {pred['prediction']}")
            print(f"  Should be genuine: {pred['prediction'] == 'genuine'}")
        else:
            print(f"Error: {result.get('error')}")
    except Exception as e:
        print(f"Request failed: {e}")

def example_threshold_optimization():
    """Example of finding optimal threshold"""
    print("\n=== Threshold Optimization Example ===")
    
    inferencer = ONNXSigNetInference(
        model_path='signet.onnx',
        threshold=0.5
    )
    
    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    image1_path = 'data/CEDAR/full_org/original_38_1.png'
    genuine_path = 'data/CEDAR/full_org/original_38_3.png'
    forged_path = 'data/CEDAR/full_forg/forgeries_38_13.png'
    
    if all(os.path.exists(p) for p in [image1_path, genuine_path, forged_path]):
        print("Testing different thresholds:")
        
        for threshold in thresholds:
            inferencer.set_threshold(threshold)
            
            # Test genuine pair
            genuine_result = inferencer.predict_pair(image1_path, genuine_path)
            genuine_correct = genuine_result['prediction'] == 'genuine'
            
            # Test forged pair
            forged_result = inferencer.predict_pair(image1_path, forged_path)
            forged_correct = forged_result['prediction'] == 'forged'
            
            accuracy = (genuine_correct + forged_correct) / 2
            
            print(f"  Threshold {threshold:.1f}: "
                  f"Genuine={genuine_result['distance']:.3f}({genuine_result['prediction']}), "
                  f"Forged={forged_result['distance']:.3f}({forged_result['prediction']}), "
                  f"Accuracy={accuracy:.1%}")
    else:
        print("Required sample images not found.")

def example_model_comparison():
    """Example comparing different execution providers"""
    print("\n=== Execution Provider Comparison ===")
    
    providers_to_test = [
        ['CPUExecutionProvider'],
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ]
    
    for providers in providers_to_test:
        try:
            print(f"\nTesting with providers: {providers}")
            
            inferencer = ONNXSigNetInference(
                model_path='signet.onnx',
                threshold=0.5,
                providers=providers
            )
            
            # Run small benchmark
            benchmark_results = inferencer.benchmark(num_iterations=20)
            
            print(f"  Active providers: {benchmark_results['providers']}")
            print(f"  Average time: {benchmark_results['avg_inference_time_ms']:.2f} ms")
            print(f"  Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
            
        except Exception as e:
            print(f"  Failed to initialize with {providers}: {e}")

if __name__ == "__main__":
    print("SigNet ONNX Inference Examples")
    print("=" * 50)
    
    # Check if ONNX model exists
    if not os.path.exists('signet.onnx'):
        print("ONNX model 'signet.onnx' not found.")
        print("Please run the conversion script first:")
        print("python convert_to_onnx.py --model_path your_model.pt --output_path signet.onnx")
        exit(1)
    
    # Run examples
    try:
        example_basic_inference()
        example_batch_inference()
        example_performance_benchmark()
        example_pil_image_inference()
        example_threshold_optimization()
        example_model_comparison()
        
        # API examples (these require the API server to be running)
        print("\n" + "=" * 50)
        print("API Examples (requires server running)")
        print("Start server with: python onnx_api.py")
        print("=" * 50)
        
        example_api_client()
        example_base64_api()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Examples completed!") 