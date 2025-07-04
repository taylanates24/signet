import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Union, List, Tuple, Dict, Optional
from argparse import ArgumentParser
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorRTSigNetInference:
    def __init__(self, engine_path: str, threshold: float = 0.0349,
                 img_height: int = 155, img_width: int = 220):
        """
        Initialize TensorRT SigNet inference
        
        Args:
            engine_path (str): Path to TensorRT engine file
            threshold (float): Distance threshold for classification
            img_height (int): Image height for preprocessing
            img_width (int): Image width for preprocessing
        """
        self.engine_path = engine_path
        self.threshold = threshold
        self.img_height = img_height
        self.img_width = img_width
        
        # Initialize CUDA and TensorRT
        self._init_cuda()
        self._load_engine()
        self._allocate_buffers()
        
        # Setup image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            ImageOps.invert,
            transforms.ToTensor(),
        ])
        
        logger.info(f"TensorRT engine loaded: {engine_path}")
        logger.info(f"Threshold: {self.threshold}")
        logger.info(f"Image dimensions: {self.img_height}x{self.img_width}")
        logger.info(f"Max batch size: {self.max_batch_size}")
    
    def _init_cuda(self):
        """Initialize CUDA context"""
        cuda.init()
        self.device = cuda.Device(0)  # Use first GPU
        self.cuda_context = self.device.make_context()
        
        # Get GPU info
        free_mem, total_mem = cuda.mem_get_info()
        logger.info(f"GPU: {self.device.name()}")
        logger.info(f"GPU Memory: {free_mem/(1024**3):.1f}GB free / {total_mem/(1024**3):.1f}GB total")
    
    def _load_engine(self):
        """Load TensorRT engine"""
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine from file
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # Get engine info
        try:
            self.max_batch_size = self.engine.max_batch_size
        except AttributeError:
            # In newer TensorRT versions, infer from binding shapes
            self.max_batch_size = 1  # Default, will be updated based on shapes
            
        # Handle different TensorRT API versions
        try:
            self.num_bindings = self.engine.num_bindings
        except AttributeError:
            # Newer TensorRT versions use num_io_tensors
            self.num_bindings = self.engine.num_io_tensors
        
        # Get input/output binding info
        self.bindings = []
        self.inputs = []
        self.outputs = []
        
        for i in range(self.num_bindings):
            # Handle different TensorRT API versions
            try:
                binding_name = self.engine.get_binding_name(i)
                binding_shape = self.engine.get_binding_shape(i)
                binding_dtype = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)
            except AttributeError:
                # Newer TensorRT versions use tensor names
                binding_name = self.engine.get_tensor_name(i)
                binding_shape = self.engine.get_tensor_shape(binding_name)
                binding_dtype = self.engine.get_tensor_dtype(binding_name)
                is_input = self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT
            
            binding_info = {
                'name': binding_name,
                'shape': binding_shape,
                'dtype': binding_dtype,
                'is_input': is_input,
                'index': i
            }
            
            if is_input:
                self.inputs.append(binding_info)
                # Try to infer max batch size from input shapes
                if len(binding_shape) > 0 and binding_shape[0] > 0:
                    self.max_batch_size = max(self.max_batch_size, binding_shape[0])
                elif len(binding_shape) > 0 and binding_shape[0] == -1:
                    # Dynamic batch size, use a reasonable default
                    self.max_batch_size = 8
            else:
                self.outputs.append(binding_info)
            
            logger.info(f"Binding {i}: {binding_name}, shape={binding_shape}, dtype={binding_dtype}, input={is_input}")
    
    def _allocate_buffers(self):
        """Allocate GPU and CPU buffers"""
        self.gpu_buffers = []
        self.cpu_buffers = []
        
        for i in range(self.num_bindings):
            # Handle different TensorRT API versions
            try:
                binding_shape = self.engine.get_binding_shape(i)
                binding_dtype = self.engine.get_binding_dtype(i)
            except AttributeError:
                # Newer TensorRT versions use tensor names
                binding_name = self.engine.get_tensor_name(i)
                binding_shape = self.engine.get_tensor_shape(binding_name)
                binding_dtype = self.engine.get_tensor_dtype(binding_name)
            
            # Calculate size
            if -1 in binding_shape:
                # Dynamic shape, use max batch size
                shape = list(binding_shape)
                if shape[0] == -1:
                    shape[0] = self.max_batch_size
                size = np.prod(shape)
            else:
                size = np.prod(binding_shape)
            
            # Allocate CPU buffer
            if binding_dtype == trt.float32:
                cpu_buffer = np.empty(size, dtype=np.float32)
            elif binding_dtype == trt.float16:
                cpu_buffer = np.empty(size, dtype=np.float16)
            else:
                cpu_buffer = np.empty(size, dtype=np.float32)
            
            # Allocate GPU buffer
            gpu_buffer = cuda.mem_alloc(cpu_buffer.nbytes)
            
            self.cpu_buffers.append(cpu_buffer)
            self.gpu_buffers.append(gpu_buffer)
        
        logger.info(f"Allocated {len(self.gpu_buffers)} GPU buffers")
    
    def preprocess_image(self, image_path: Union[str, Image.Image]) -> np.ndarray:
        """
        Preprocess image for TensorRT inference
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path).convert('L')
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('L')
        else:
            raise ValueError("image_path must be a string path or PIL Image")
        
        # Apply transforms
        image_tensor = self.transform(image)
        image_array = image_tensor.numpy()
        
        return image_array
    
    def predict_pair(self, image1: Union[str, Image.Image], 
                    image2: Union[str, Image.Image]) -> Dict:
        """
        Predict if two signatures are from the same person
        
        Args:
            image1: First signature image
            image2: Second signature image
            
        Returns:
            Dict: Prediction results
        """
        start_time = time.time()
        
        # Preprocess images
        img1_array = self.preprocess_image(image1)
        img2_array = self.preprocess_image(image2)
        
        # Prepare batch (add batch dimension)
        batch_img1 = img1_array[np.newaxis, ...]  # Add batch dimension
        batch_img2 = img2_array[np.newaxis, ...]
        
        # Copy to input buffers
        input1_buffer = self.cpu_buffers[self.inputs[0]['index']]
        input2_buffer = self.cpu_buffers[self.inputs[1]['index']]
        
        np.copyto(input1_buffer[:batch_img1.size], batch_img1.flatten())
        np.copyto(input2_buffer[:batch_img2.size], batch_img2.flatten())
        
        # Copy inputs to GPU
        inference_start = time.time()
        
        cuda.memcpy_htod(self.gpu_buffers[self.inputs[0]['index']], input1_buffer)
        cuda.memcpy_htod(self.gpu_buffers[self.inputs[1]['index']], input2_buffer)
        
        # Run inference
        try:
            self.context.execute_v2(self.gpu_buffers)
        except AttributeError:
            # Newer TensorRT versions use execute_async_v3
            stream = cuda.Stream()
            self.context.execute_async_v3(stream.handle)
        
        # Copy outputs from GPU
        output1_buffer = self.cpu_buffers[self.outputs[0]['index']]
        output2_buffer = self.cpu_buffers[self.outputs[1]['index']]
        
        cuda.memcpy_dtoh(output1_buffer, self.gpu_buffers[self.outputs[0]['index']])
        cuda.memcpy_dtoh(output2_buffer, self.gpu_buffers[self.outputs[1]['index']])
        
        inference_time = time.time() - inference_start
        
        # Extract embeddings (reshape to remove batch dimension)
        embedding1 = output1_buffer[:128].copy()  # SigNet outputs 128-dim embeddings
        embedding2 = output2_buffer[:128].copy()
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Make prediction
        is_genuine = distance < self.threshold
        
        total_time = time.time() - start_time
        
        return {
            'distance': float(distance),
            'threshold': self.threshold,
            'prediction': 'genuine' if is_genuine else 'forged',
            'inference_time_ms': inference_time * 1000
        }
    
    def predict_batch(self, image_pairs: List[Tuple[Union[str, Image.Image], 
                                                  Union[str, Image.Image]]]) -> Dict:
        """
        Predict for a batch of image pairs
        
        Args:
            image_pairs: List of image pairs
            
        Returns:
            Dict: Batch prediction results
        """
        batch_size = len(image_pairs)
        if batch_size > self.max_batch_size:
            # Process in chunks
            results = []
            for i in range(0, batch_size, self.max_batch_size):
                chunk = image_pairs[i:i + self.max_batch_size]
                chunk_results = self._predict_batch_internal(chunk)
                results.extend(chunk_results)
        else:
            results = self._predict_batch_internal(image_pairs)
        
        # Calculate summary statistics
        distances = [r['distance'] for r in results]
        genuine_count = sum(1 for r in results if r['prediction'] == 'genuine')
        
        total_time = sum(r['inference_time_ms'] for r in results)
        
        summary = {
            'total_pairs': len(image_pairs),
            'genuine_predictions': genuine_count,
            'forged_predictions': len(image_pairs) - genuine_count,
            'avg_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'total_processing_time_ms': total_time,
            'avg_time_per_pair_ms': total_time / len(image_pairs)
        }
        
        return {
            'results': results,
            'summary': summary
        }
    
    def _predict_batch_internal(self, image_pairs: List[Tuple]) -> List[Dict]:
        """Internal batch prediction for actual batch size <= max_batch_size"""
        batch_size = len(image_pairs)
        start_time = time.time()
        
        # Preprocess all images
        batch_img1 = np.zeros((batch_size, 1, self.img_height, self.img_width), dtype=np.float32)
        batch_img2 = np.zeros((batch_size, 1, self.img_height, self.img_width), dtype=np.float32)
        
        for i, (img1, img2) in enumerate(image_pairs):
            batch_img1[i] = self.preprocess_image(img1)
            batch_img2[i] = self.preprocess_image(img2)
        
        # Copy to input buffers
        input1_buffer = self.cpu_buffers[self.inputs[0]['index']]
        input2_buffer = self.cpu_buffers[self.inputs[1]['index']]
        
        np.copyto(input1_buffer[:batch_img1.size], batch_img1.flatten())
        np.copyto(input2_buffer[:batch_img2.size], batch_img2.flatten())
        
        # GPU inference
        inference_start = time.time()
        
        cuda.memcpy_htod(self.gpu_buffers[self.inputs[0]['index']], input1_buffer)
        cuda.memcpy_htod(self.gpu_buffers[self.inputs[1]['index']], input2_buffer)
        
        # Set dynamic batch size if needed
        if self.max_batch_size > 1:
            for binding_idx in range(self.num_bindings):
                try:
                    shape = list(self.engine.get_binding_shape(binding_idx))
                    if shape[0] == -1:
                        shape[0] = batch_size
                        self.context.set_binding_shape(binding_idx, shape)
                except AttributeError:
                    # Newer TensorRT versions
                    binding_name = self.engine.get_tensor_name(binding_idx)
                    shape = list(self.engine.get_tensor_shape(binding_name))
                    if shape[0] == -1:
                        shape[0] = batch_size
                        self.context.set_input_shape(binding_name, shape)
        
        try:
            self.context.execute_v2(self.gpu_buffers)
        except AttributeError:
            # Newer TensorRT versions use execute_async_v3
            stream = cuda.Stream()
            self.context.execute_async_v3(stream.handle)
        
        # Copy outputs
        output1_buffer = self.cpu_buffers[self.outputs[0]['index']]
        output2_buffer = self.cpu_buffers[self.outputs[1]['index']]
        
        cuda.memcpy_dtoh(output1_buffer, self.gpu_buffers[self.outputs[0]['index']])
        cuda.memcpy_dtoh(output2_buffer, self.gpu_buffers[self.outputs[1]['index']])
        
        inference_time = time.time() - inference_start
        
        # Process results
        results = []
        for i in range(batch_size):
            # Extract embeddings for this batch item
            embedding1 = output1_buffer[i*128:(i+1)*128]
            embedding2 = output2_buffer[i*128:(i+1)*128]
            
            # Calculate distance and prediction
            distance = np.linalg.norm(embedding1 - embedding2)
            is_genuine = distance < self.threshold
            
            results.append({
                'distance': float(distance),
                'threshold': self.threshold,
                'prediction': 'genuine' if is_genuine else 'forged',
                'inference_time_ms': (inference_time / batch_size) * 1000,
                'pair_index': i
            })
        
        return results
    
    def benchmark(self, num_iterations: int = 100, batch_size: int = 1) -> Dict:
        """
        Benchmark TensorRT inference performance
        
        Args:
            num_iterations: Number of iterations to run
            batch_size: Batch size for benchmarking
            
        Returns:
            Dict: Benchmark results
        """
        logger.info(f"Running TensorRT benchmark: {num_iterations} iterations, batch size {batch_size}")
        
        # Create dummy inputs
        dummy_pairs = []
        dummy_img = np.random.randn(1, self.img_height, self.img_width).astype(np.float32)
        
        for _ in range(batch_size):
            # Convert to PIL Image for consistent preprocessing
            img_pil = Image.fromarray((dummy_img[0] * 255).astype(np.uint8), mode='L')
            dummy_pairs.append((img_pil, img_pil))
        
        # Warmup
        for _ in range(10):
            if batch_size == 1:
                self.predict_pair(dummy_pairs[0][0], dummy_pairs[0][1])
            else:
                self.predict_batch(dummy_pairs)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            
            if batch_size == 1:
                self.predict_pair(dummy_pairs[0][0], dummy_pairs[0][1])
            else:
                self.predict_batch(dummy_pairs)
            
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        return {
            'num_iterations': num_iterations,
            'batch_size': batch_size,
            'avg_inference_time_ms': np.mean(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'std_inference_time_ms': np.std(times),
            'throughput_fps': (1000 * batch_size) / np.mean(times),
            'engine_path': self.engine_path,
            'max_batch_size': self.max_batch_size
        }
    
    def set_threshold(self, new_threshold: float):
        """Update classification threshold"""
        self.threshold = new_threshold
        logger.info(f"Threshold updated to: {self.threshold}")
    
    def get_engine_info(self) -> Dict:
        """Get TensorRT engine information"""
        try:
            gpu_name = self.device.name()
        except:
            gpu_name = "Unknown GPU"
            
        return {
            'engine_path': self.engine_path,
            'max_batch_size': self.max_batch_size,
            'num_bindings': self.num_bindings,
            'inputs': [{'name': inp['name'], 'shape': inp['shape'], 'dtype': str(inp['dtype'])} for inp in self.inputs],
            'outputs': [{'name': out['name'], 'shape': out['shape'], 'dtype': str(out['dtype'])} for out in self.outputs],
            'threshold': self.threshold,
            'image_dimensions': [self.img_height, self.img_width],
            'gpu_name': gpu_name
        }
    
    def __del__(self):
        """Cleanup CUDA context"""
        try:
            if hasattr(self, 'cuda_context'):
                self.cuda_context.pop()
        except:
            pass

def compare_with_onnx(tensorrt_engine: str, onnx_model: str, 
                     test_image1: str, test_image2: str) -> Dict:
    """
    Compare TensorRT and ONNX inference results
    
    Args:
        tensorrt_engine: Path to TensorRT engine
        onnx_model: Path to ONNX model
        test_image1: Path to first test image
        test_image2: Path to second test image
        
    Returns:
        Dict: Comparison results
    """
    try:
        from onnx_inference import ONNXSigNetInference
        
        # Initialize both models
        trt_inferencer = TensorRTSigNetInference(tensorrt_engine)
        onnx_inferencer = ONNXSigNetInference(onnx_model, threshold=trt_inferencer.threshold)
        
        # Run inference
        trt_result = trt_inferencer.predict_pair(test_image1, test_image2)
        onnx_result = onnx_inferencer.predict_pair(test_image1, test_image2)
        
        # Compare results
        distance_diff = abs(trt_result['distance'] - onnx_result['distance'])
        prediction_match = trt_result['prediction'] == onnx_result['prediction']
        
        speedup = onnx_result['inference_time_ms'] / trt_result['inference_time_ms']
        
        return {
            'tensorrt': trt_result,
            'onnx': onnx_result,
            'distance_difference': distance_diff,
            'prediction_match': prediction_match,
            'speedup': speedup,
            'comparison_successful': True
        }
        
    except ImportError:
        return {'error': 'ONNX inference module not available', 'comparison_successful': False}
    except Exception as e:
        return {'error': str(e), 'comparison_successful': False}

if __name__ == "__main__":
    parser = ArgumentParser(description='TensorRT SigNet Inference')
    parser.add_argument('--engine_path', type=str, default='signet_fp32.trt',
                       help='Path to TensorRT engine file')
    parser.add_argument('--image1', type=str, 
                       default='data/CEDAR/full_org/original_38_1.png',
                       help='Path to first signature image')
    parser.add_argument('--image2', type=str, 
                       default='data/CEDAR/full_org/original_38_3.png',
                       help='Path to second signature image')
    parser.add_argument('--threshold', type=float, default=0.0349,
                       help='Distance threshold for classification')
    parser.add_argument('--img_height', type=int, default=155,
                       help='Image height for preprocessing')
    parser.add_argument('--img_width', type=int, default=220,
                       help='Image width for preprocessing')
    parser.add_argument('--benchmark', action='store_true', default=True,
                       help='Run performance benchmark')
    parser.add_argument('--benchmark_iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--benchmark_batch_sizes', nargs='+', type=int, default=[1, 4, 8],
                       help='Batch sizes for benchmarking')
    parser.add_argument('--batch_test', action='store_true',
                       help='Test batch processing')
    parser.add_argument('--compare_onnx', type=str,
                       help='Path to ONNX model for comparison')
    parser.add_argument('--engine_info', action='store_true',
                       help='Display engine information')
    parser.add_argument('--output_json', type=str,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize TensorRT inferencer
    try:
        inferencer = TensorRTSigNetInference(
            engine_path=args.engine_path,
            threshold=args.threshold,
            img_height=args.img_height,
            img_width=args.img_width
        )
    except Exception as e:
        logger.error(f"Failed to initialize TensorRT inferencer: {e}")
        exit(1)
    
    results = {}
    
    # Single pair prediction
    if args.image1 and args.image2:
        logger.info("=== Single Pair Prediction ===")
        try:
            result = inferencer.predict_pair(args.image1, args.image2)
            
            print(f"Image 1: {args.image1}")
            print(f"Image 2: {args.image2}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Prediction: {result['prediction']}")
            print(f"Inference time: {result['inference_time_ms']:.2f} ms")
            
            results['single_prediction'] = result
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
    
    # Engine info
    if args.engine_info:
        logger.info("=== Engine Information ===")
        engine_info = inferencer.get_engine_info()
        for key, value in engine_info.items():
            print(f"{key}: {value}")
        results['engine_info'] = engine_info
    
    # Benchmark
    if args.benchmark:
        logger.info("=== Performance Benchmark ===")
        benchmark_results = {}
        
        for batch_size in args.benchmark_batch_sizes:
            if batch_size <= inferencer.max_batch_size:
                try:
                    bench_result = inferencer.benchmark(
                        num_iterations=args.benchmark_iterations,
                        batch_size=batch_size
                    )
                    
                    print(f"\nBatch size {batch_size}:")
                    print(f"  Average time: {bench_result['avg_inference_time_ms']:.2f} ms")
                    print(f"  Throughput: {bench_result['throughput_fps']:.1f} FPS")
                    print(f"  Min/Max time: {bench_result['min_inference_time_ms']:.2f}/{bench_result['max_inference_time_ms']:.2f} ms")
                    
                    benchmark_results[f'batch_{batch_size}'] = bench_result
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for batch size {batch_size}: {e}")
            else:
                logger.warning(f"Batch size {batch_size} exceeds max batch size {inferencer.max_batch_size}")
        
        results['benchmark'] = benchmark_results
    
    # Batch test
    if args.batch_test and args.image1 and args.image2:
        logger.info("=== Batch Processing Test ===")
        try:
            test_pairs = [(args.image1, args.image2)] * 5
            batch_results = inferencer.predict_batch(test_pairs)
            
            print(f"Processed {batch_results['summary']['total_pairs']} pairs")
            print(f"Average time per pair: {batch_results['summary']['avg_time_per_pair_ms']:.2f} ms")
            print(f"Genuine predictions: {batch_results['summary']['genuine_predictions']}")
            print(f"Forged predictions: {batch_results['summary']['forged_predictions']}")
            
            results['batch_test'] = batch_results
            
        except Exception as e:
            logger.error(f"Batch test failed: {e}")
    
    # Compare with ONNX
    if args.compare_onnx and args.image1 and args.image2:
        logger.info("=== TensorRT vs ONNX Comparison ===")
        try:
            comparison = compare_with_onnx(
                args.engine_path, args.compare_onnx,
                args.image1, args.image2
            )
            
            if comparison['comparison_successful']:
                print(f"Distance difference: {comparison['distance_difference']:.6f}")
                print(f"Prediction match: {comparison['prediction_match']}")
                print(f"TensorRT speedup: {comparison['speedup']:.1f}x")
                print(f"TensorRT time: {comparison['tensorrt']['inference_time_ms']:.2f} ms")
                print(f"ONNX time: {comparison['onnx']['inference_time_ms']:.2f} ms")
            else:
                print(f"Comparison failed: {comparison['error']}")
            
            results['comparison'] = comparison
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
    
    # Save results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output_json}")
    
    logger.info("=== TensorRT Inference Complete ===") 