#!/usr/bin/env python3
"""
Example usage scripts for TensorRT SigNet conversion and inference
This file demonstrates the complete pipeline from ONNX to TensorRT optimization
"""

import os
import time
import numpy as np
from PIL import Image

def example_conversion_pipeline():
    """Example of complete conversion pipeline: PyTorch -> ONNX -> TensorRT"""
    print("=== Complete Conversion Pipeline Example ===")
    
    # Step 1: Convert PyTorch to ONNX (if not already done)
    if not os.path.exists('signet.onnx'):
        print("Converting PyTorch model to ONNX...")
        os.system('python convert_to_onnx.py')
    
    # Step 2: Convert ONNX to TensorRT
    if not os.path.exists('signet.trt'):
        print("Converting ONNX model to TensorRT...")
        os.system('python convert_to_tensorrt.py --onnx_path signet.onnx --engine_path signet.trt --precision fp16')
    
    print("Conversion pipeline completed!")
    print("Files created:")
    if os.path.exists('signet.onnx'):
        print(f"  - ONNX model: signet.onnx ({os.path.getsize('signet.onnx')/(1024**2):.1f} MB)")
    if os.path.exists('signet.trt'):
        print(f"  - TensorRT engine: signet.trt ({os.path.getsize('signet.trt')/(1024**2):.1f} MB)")

def example_basic_tensorrt_inference():
    """Basic TensorRT inference example"""
    print("\n=== Basic TensorRT Inference Example ===")
    
    try:
        from tensorrt_inference import TensorRTSigNetInference
        
        # Initialize TensorRT inferencer
        inferencer = TensorRTSigNetInference(
            engine_path='signet.trt',
            threshold=0.0349
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
            print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        else:
            print("Sample images not found. Please check the paths.")
            
    except ImportError as e:
        print(f"TensorRT not available: {e}")
    except Exception as e:
        print(f"Error: {e}")

def example_tensorrt_vs_onnx_comparison():
    """Compare TensorRT and ONNX performance"""
    print("\n=== TensorRT vs ONNX Performance Comparison ===")
    
    try:
        from tensorrt_inference import TensorRTSigNetInference, compare_with_onnx
        
        image1_path = 'data/CEDAR/full_org/original_38_1.png'
        image2_path = 'data/CEDAR/full_org/original_38_3.png'
        
        if os.path.exists(image1_path) and os.path.exists(image2_path):
            comparison = compare_with_onnx(
                'signet.trt', 'signet.onnx',
                image1_path, image2_path
            )
            
            if comparison['comparison_successful']:
                print("Comparison Results:")
                print(f"  TensorRT time: {comparison['tensorrt']['inference_time_ms']:.2f} ms")
                print(f"  ONNX time: {comparison['onnx']['inference_time_ms']:.2f} ms")
                print(f"  Speedup: {comparison['speedup']:.1f}x")
                print(f"  Distance difference: {comparison['distance_difference']:.6f}")
                print(f"  Prediction match: {comparison['prediction_match']}")
            else:
                print(f"Comparison failed: {comparison.get('error', 'Unknown error')}")
        else:
            print("Sample images not found.")
            
    except Exception as e:
        print(f"Error: {e}")

def example_tensorrt_batch_processing():
    """Example of TensorRT batch processing"""
    print("\n=== TensorRT Batch Processing Example ===")
    
    try:
        from tensorrt_inference import TensorRTSigNetInference
        
        inferencer = TensorRTSigNetInference('signet.trt', threshold=0.0349)
        
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
        else:
            print("No valid image pairs found.")
            
    except Exception as e:
        print(f"Error: {e}")

def example_tensorrt_benchmark():
    """TensorRT performance benchmark"""
    print("\n=== TensorRT Performance Benchmark ===")
    
    try:
        from tensorrt_inference import TensorRTSigNetInference
        
        inferencer = TensorRTSigNetInference('signet.trt', threshold=0.0349)
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8] if inferencer.max_batch_size >= 8 else [1]
        
        for batch_size in batch_sizes:
            if batch_size <= inferencer.max_batch_size:
                print(f"\nBenchmarking batch size {batch_size}:")
                
                benchmark_result = inferencer.benchmark(
                    num_iterations=50,
                    batch_size=batch_size
                )
                
                print(f"  Average inference time: {benchmark_result['avg_inference_time_ms']:.2f} ms")
                print(f"  Throughput: {benchmark_result['throughput_fps']:.1f} FPS")
                print(f"  Standard deviation: {benchmark_result['std_inference_time_ms']:.2f} ms")
            else:
                print(f"Batch size {batch_size} exceeds max batch size {inferencer.max_batch_size}")
                
    except Exception as e:
        print(f"Error: {e}")

def example_precision_comparison():
    """Compare different TensorRT precision modes"""
    print("\n=== TensorRT Precision Comparison ===")
    
    precisions = ['fp32', 'fp16']
    results = {}
    
    for precision in precisions:
        engine_path = f'signet_{precision}.trt'
        
        print(f"\nTesting {precision.upper()} precision:")
        
        # Convert with specific precision
        if not os.path.exists(engine_path):
            print(f"  Converting to {precision}...")
            os.system(f'python convert_to_tensorrt.py --onnx_path signet.onnx --engine_path {engine_path} --precision {precision}')
        
        if os.path.exists(engine_path):
            try:
                from tensorrt_inference import TensorRTSigNetInference
                
                inferencer = TensorRTSigNetInference(engine_path)
                
                # Quick benchmark
                benchmark = inferencer.benchmark(num_iterations=20, batch_size=1)
                
                engine_size_mb = os.path.getsize(engine_path) / (1024**2)
                
                results[precision] = {
                    'avg_time_ms': benchmark['avg_inference_time_ms'],
                    'throughput_fps': benchmark['throughput_fps'],
                    'engine_size_mb': engine_size_mb
                }
                
                print(f"  Average time: {benchmark['avg_inference_time_ms']:.2f} ms")
                print(f"  Throughput: {benchmark['throughput_fps']:.1f} FPS")
                print(f"  Engine size: {engine_size_mb:.1f} MB")
                
            except Exception as e:
                print(f"  Error testing {precision}: {e}")
        else:
            print(f"  Failed to create {precision} engine")
    
    # Compare results
    if len(results) > 1:
        print(f"\nPrecision Comparison Summary:")
        for precision, result in results.items():
            print(f"  {precision.upper()}: {result['avg_time_ms']:.2f}ms, {result['throughput_fps']:.1f}FPS, {result['engine_size_mb']:.1f}MB")

def example_memory_optimization():
    """Example of TensorRT memory optimization"""
    print("\n=== TensorRT Memory Optimization Example ===")
    
    # Test different workspace sizes
    workspace_sizes = [256, 512, 1024, 2048]  # MB
    
    for workspace_mb in workspace_sizes:
        engine_path = f'signet_ws{workspace_mb}.trt'
        
        print(f"\nTesting workspace size: {workspace_mb} MB")
        
        if not os.path.exists(engine_path):
            print(f"  Building engine with {workspace_mb}MB workspace...")
            os.system(f'python convert_to_tensorrt.py --onnx_path signet.onnx --engine_path {engine_path} --workspace_size_mb {workspace_mb} --precision fp16')
        
        if os.path.exists(engine_path):
            try:
                from tensorrt_inference import TensorRTSigNetInference
                
                inferencer = TensorRTSigNetInference(engine_path)
                benchmark = inferencer.benchmark(num_iterations=10, batch_size=1)
                
                engine_size_mb = os.path.getsize(engine_path) / (1024**2)
                
                print(f"  Performance: {benchmark['avg_inference_time_ms']:.2f} ms")
                print(f"  Engine size: {engine_size_mb:.1f} MB")
                
                # Clean up large test engines
                if workspace_mb > 1024:
                    os.remove(engine_path)
                    
            except Exception as e:
                print(f"  Error: {e}")

def example_dynamic_shapes():
    """Example of TensorRT dynamic shapes"""
    print("\n=== TensorRT Dynamic Shapes Example ===")
    
    try:
        # Create engine with dynamic shapes
        dynamic_engine = 'signet_dynamic.trt'
        
        if not os.path.exists(dynamic_engine):
            print("Creating dynamic shapes engine...")
            os.system(f'python convert_to_tensorrt.py --onnx_path signet.onnx --engine_path {dynamic_engine} --dynamic_shapes --max_batch_size 8 --precision fp16')
        
        if os.path.exists(dynamic_engine):
            from tensorrt_inference import TensorRTSigNetInference
            
            inferencer = TensorRTSigNetInference(dynamic_engine)
            
            print(f"Max batch size: {inferencer.max_batch_size}")
            
            # Test different batch sizes
            for batch_size in [1, 2, 4]:
                if batch_size <= inferencer.max_batch_size:
                    benchmark = inferencer.benchmark(num_iterations=10, batch_size=batch_size)
                    print(f"Batch {batch_size}: {benchmark['avg_inference_time_ms']:.2f} ms/batch ({benchmark['throughput_fps']:.1f} FPS)")
        else:
            print("Failed to create dynamic shapes engine")
            
    except Exception as e:
        print(f"Error: {e}")

def example_conversion_optimization():
    """Example of conversion optimization strategies"""
    print("\n=== TensorRT Conversion Optimization Example ===")
    
    try:
        from convert_to_tensorrt import TensorRTConverter
        
        converter = TensorRTConverter('signet.onnx', 'signet_optimized.trt')
        
        # Run conversion benchmark to find optimal settings
        print("Running conversion benchmark...")
        benchmark_results = converter.benchmark_conversion(
            precisions=['fp16'],
            batch_sizes=[1, 4],
            workspace_sizes=[512*1024*1024, 1024*1024*1024]  # 512MB, 1GB
        )
        
        print("Benchmark Results:")
        for result in benchmark_results['results']:
            if result['success']:
                print(f"  {result['precision']} batch{result['batch_size']} ws{result['workspace_size_mb']}MB: "
                      f"{result['build_time_s']:.1f}s build, {result['engine_size_mb']:.1f}MB size")
        
        summary = benchmark_results['summary']
        if summary['successful'] > 0:
            print(f"\nSummary:")
            print(f"  Successful configurations: {summary['successful']}/{summary['total']}")
            print(f"  Average build time: {summary['avg_build_time_s']:.1f}s")
            print(f"  Average engine size: {summary['avg_engine_size_mb']:.1f}MB")
            
    except Exception as e:
        print(f"Error: {e}")

def check_system_requirements():
    """Check if system meets TensorRT requirements"""
    print("=== System Requirements Check ===")
    
    try:
        import tensorrt as trt
        print(f"✓ TensorRT installed: version {trt.__version__}")
    except ImportError:
        print("✗ TensorRT not installed")
        return False
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        cuda.init()
        device_count = cuda.Device.count()
        print(f"✓ CUDA devices found: {device_count}")
        
        if device_count > 0:
            device = cuda.Device(0)
            print(f"✓ Primary GPU: {device.name()}")
            
            context = device.make_context()
            free_mem, total_mem = cuda.mem_get_info()
            context.pop()
            
            print(f"✓ GPU Memory: {total_mem/(1024**3):.1f}GB total, {free_mem/(1024**3):.1f}GB free")
            
            return True
        else:
            print("✗ No CUDA devices found")
            return False
            
    except ImportError:
        print("✗ PyCUDA not installed")
        return False
    except Exception as e:
        print(f"✗ CUDA error: {e}")
        return False

if __name__ == "__main__":
    print("TensorRT SigNet Examples")
    print("=" * 50)
    
    # Check system requirements first
    if not check_system_requirements():
        print("\nSystem requirements not met. Please install TensorRT and CUDA dependencies.")
        print("See tensorrt_requirements.txt for installation instructions.")
        exit(1)
    
    print("\n" + "=" * 50)
    
    try:
        # Run examples
        example_conversion_pipeline()
        
        if os.path.exists('signet.trt'):
            example_basic_tensorrt_inference()
            example_tensorrt_batch_processing()
            example_tensorrt_benchmark()
            
            if os.path.exists('signet.onnx'):
                example_tensorrt_vs_onnx_comparison()
        
        # Advanced examples (optional)
        example_precision_comparison()
        example_memory_optimization()
        example_dynamic_shapes()
        example_conversion_optimization()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("TensorRT examples completed!")
    print("\nNext steps:")
    print("1. Optimize your engine settings based on benchmark results")
    print("2. Integrate TensorRT inference into your application")
    print("3. Deploy with appropriate hardware specifications") 