import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import atexit
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Union, List, Tuple, Dict, Optional
from argparse import ArgumentParser
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global list to track active contexts for cleanup
_active_contexts = []

def cleanup_all_contexts():
    """Cleanup all active CUDA contexts at program exit"""
    global _active_contexts
    for context in _active_contexts[:]:
        try:
            if context:
                context.pop()
                _active_contexts.remove(context)
        except:
            pass

# Register cleanup function to run at program exit
atexit.register(cleanup_all_contexts)

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
        self._cleaned_up = False
        
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
        global _active_contexts
        cuda.init()
        self.device = cuda.Device(0)  # Use first GPU
        self.cuda_context = self.device.make_context()
        _active_contexts.append(self.cuda_context)
        
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
        
        # TensorRT 10.x uses new API
        self.num_io_tensors = self.engine.num_io_tensors
        self.max_batch_size = 8  # Default for dynamic shapes
        
        # Get input/output tensor info
        self.inputs = []
        self.outputs = []
        self.tensor_names = []
        
        for i in range(self.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            is_input = tensor_mode == trt.TensorIOMode.INPUT
            
            self.tensor_names.append(tensor_name)
            
            tensor_info = {
                'name': tensor_name,
                'shape': tensor_shape,
                'dtype': tensor_dtype,
                'is_input': is_input,
                'index': i
            }
            
            if is_input:
                self.inputs.append(tensor_info)
                # Try to infer max batch size from input shapes
                if len(tensor_shape) > 0 and tensor_shape[0] > 0:
                    self.max_batch_size = max(self.max_batch_size, tensor_shape[0])
                elif len(tensor_shape) > 0 and tensor_shape[0] == -1:
                    # Dynamic batch size, use a reasonable default
                    self.max_batch_size = 8
            else:
                self.outputs.append(tensor_info)
            
            logger.info(f"Tensor {i}: {tensor_name}, shape={tensor_shape}, dtype={tensor_dtype}, input={is_input}")
        
        # For compatibility with older code
        self.num_bindings = self.num_io_tensors
    
    def _allocate_buffers(self):
        """Allocate GPU and CPU buffers for TensorRT 10.x"""
        self.tensor_buffers = {}
        
        for i in range(self.num_io_tensors):
            tensor_name = self.tensor_names[i]
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            
            # Calculate size
            if -1 in tensor_shape:
                # Dynamic shape, use max batch size
                shape = list(tensor_shape)
                if shape[0] == -1:
                    shape[0] = self.max_batch_size
                size = np.prod(shape)
            else:
                size = np.prod(tensor_shape)
            
            # Allocate CPU buffer
            if tensor_dtype == trt.float32:
                cpu_buffer = np.empty(size, dtype=np.float32)
            elif tensor_dtype == trt.float16:
                cpu_buffer = np.empty(size, dtype=np.float16)
            else:
                cpu_buffer = np.empty(size, dtype=np.float32)
            
            # Allocate GPU buffer
            gpu_buffer = cuda.mem_alloc(cpu_buffer.nbytes)
            
            self.tensor_buffers[tensor_name] = {
                'cpu': cpu_buffer,
                'gpu': gpu_buffer,
                'shape': tensor_shape,
                'dtype': tensor_dtype
            }
        
        # Keep old interface for compatibility
        self.cpu_buffers = []
        self.gpu_buffers = []
        for tensor_name in self.tensor_names:
            self.cpu_buffers.append(self.tensor_buffers[tensor_name]['cpu'])
            self.gpu_buffers.append(self.tensor_buffers[tensor_name]['gpu'])
        
        logger.info(f"Allocated {len(self.tensor_buffers)} tensor buffers")
    
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
        
        input1_name = self.inputs[0]['name']
        input2_name = self.inputs[1]['name']
        
        cuda.memcpy_htod(self.tensor_buffers[input1_name]['gpu'], input1_buffer)
        cuda.memcpy_htod(self.tensor_buffers[input2_name]['gpu'], input2_buffer)
        
        # Set tensor addresses for TensorRT 10.x
        for tensor_name in self.tensor_names:
            self.context.set_tensor_address(tensor_name, self.tensor_buffers[tensor_name]['gpu'])
        
        # Run inference
        stream = cuda.Stream()
        self.context.execute_async_v3(stream.handle)
        stream.synchronize()
        
        # Copy outputs from GPU
        output1_name = self.outputs[0]['name']
        output2_name = self.outputs[1]['name']
        
        output1_buffer = self.tensor_buffers[output1_name]['cpu']
        output2_buffer = self.tensor_buffers[output2_name]['cpu']
        
        cuda.memcpy_dtoh(output1_buffer, self.tensor_buffers[output1_name]['gpu'])
        cuda.memcpy_dtoh(output2_buffer, self.tensor_buffers[output2_name]['gpu'])
        
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
            'inference_time_ms': inference_time * 1000,
            'total_time_ms': total_time * 1000
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
        input1_name = self.inputs[0]['name']
        input2_name = self.inputs[1]['name']
        
        input1_buffer = self.tensor_buffers[input1_name]['cpu']
        input2_buffer = self.tensor_buffers[input2_name]['cpu']
        
        np.copyto(input1_buffer[:batch_img1.size], batch_img1.flatten())
        np.copyto(input2_buffer[:batch_img2.size], batch_img2.flatten())
        
        # GPU inference
        inference_start = time.time()
        
        cuda.memcpy_htod(self.tensor_buffers[input1_name]['gpu'], input1_buffer)
        cuda.memcpy_htod(self.tensor_buffers[input2_name]['gpu'], input2_buffer)
        
        # Set dynamic batch size if needed
        for tensor_name in self.tensor_names:
            tensor_shape = list(self.engine.get_tensor_shape(tensor_name))
            if len(tensor_shape) > 0 and tensor_shape[0] == -1:
                tensor_shape[0] = batch_size
                self.context.set_input_shape(tensor_name, tensor_shape)
        
        # Set tensor addresses for TensorRT 10.x
        for tensor_name in self.tensor_names:
            self.context.set_tensor_address(tensor_name, self.tensor_buffers[tensor_name]['gpu'])
        
        # Run inference
        stream = cuda.Stream()
        self.context.execute_async_v3(stream.handle)
        stream.synchronize()
        
        # Copy outputs
        output1_name = self.outputs[0]['name']
        output2_name = self.outputs[1]['name']
        
        output1_buffer = self.tensor_buffers[output1_name]['cpu']
        output2_buffer = self.tensor_buffers[output2_name]['cpu']
        
        cuda.memcpy_dtoh(output1_buffer, self.tensor_buffers[output1_name]['gpu'])
        cuda.memcpy_dtoh(output2_buffer, self.tensor_buffers[output2_name]['gpu'])
        
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
    
    def evaluate_dataset(self, dataset_pairs: List[Tuple[str, str, int]], 
                        output_path: Optional[str] = None) -> Dict:
        """
        Evaluate model on a dataset of signature pairs
        
        Args:
            dataset_pairs: List of (image1_path, image2_path, label) tuples
                          where label is 1 for genuine pairs, 0 for forged pairs
            output_path: Optional path to save detailed results
            
        Returns:
            Dict: Evaluation metrics
        """
        logger.info(f"Evaluating dataset with {len(dataset_pairs)} pairs")
        
        start_time = time.time()
        predictions = []
        true_labels = []
        distances = []
        
        # Process in batches for efficiency
        batch_size = min(self.max_batch_size, 32)  # Don't exceed reasonable batch size
        
        for i in range(0, len(dataset_pairs), batch_size):
            batch_pairs = dataset_pairs[i:i + batch_size]
            
            # Prepare batch data
            image_pairs = [(pair[0], pair[1]) for pair in batch_pairs]
            batch_labels = [pair[2] for pair in batch_pairs]
            
            # Run batch inference
            batch_results = self.predict_batch(image_pairs)
            
            # Collect results
            for j, result in enumerate(batch_results['results']):
                predictions.append(1 if result['prediction'] == 'genuine' else 0)
                distances.append(result['distance'])
                true_labels.append(batch_labels[j])
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        distances = np.array(distances)
        
        # Basic classification metrics
        tp = np.sum((predictions == 1) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        
        accuracy = (tp + tn) / len(true_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Equal Error Rate (EER) calculation
        genuine_distances = distances[true_labels == 1]
        forged_distances = distances[true_labels == 0]
        
        # Find EER threshold
        thresholds = np.linspace(0, max(distances), 1000)
        far_rates = []
        frr_rates = []
        
        for thresh in thresholds:
            far = np.sum(forged_distances < thresh) / len(forged_distances) if len(forged_distances) > 0 else 0
            frr = np.sum(genuine_distances >= thresh) / len(genuine_distances) if len(genuine_distances) > 0 else 0
            far_rates.append(far)
            frr_rates.append(frr)
        
        far_rates = np.array(far_rates)
        frr_rates = np.array(frr_rates)
        
        # Find EER point
        eer_idx = np.argmin(np.abs(far_rates - frr_rates))
        eer = (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        evaluation_time = time.time() - start_time
        
        results = {
            'total_pairs': len(dataset_pairs),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'equal_error_rate': float(eer),
            'eer_threshold': float(eer_threshold),
            'current_threshold': self.threshold,
            'avg_genuine_distance': float(np.mean(genuine_distances)) if len(genuine_distances) > 0 else 0,
            'avg_forged_distance': float(np.mean(forged_distances)) if len(forged_distances) > 0 else 0,
            'evaluation_time_s': evaluation_time,
            'pairs_per_second': len(dataset_pairs) / evaluation_time
        }
        
        # Save detailed results if requested
        if output_path:
            detailed_results = {
                'evaluation_metrics': results,
                'detailed_predictions': [
                    {
                        'image1': dataset_pairs[i][0],
                        'image2': dataset_pairs[i][1],
                        'true_label': int(true_labels[i]),
                        'predicted_label': int(predictions[i]),
                        'distance': float(distances[i]),
                        'correct': bool(predictions[i] == true_labels[i])
                    }
                    for i in range(len(dataset_pairs))
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            logger.info(f"Detailed results saved to: {output_path}")
        
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
    
    def cleanup(self):
        """Cleanup resources in proper order for TensorRT 10.x"""
        global _active_contexts
        
        if self._cleaned_up:
            return
        
        self._cleaned_up = True
        
        try:
            # First, destroy the execution context
            if hasattr(self, 'context') and self.context:
                del self.context
                self.context = None
        except:
            pass
        
        try:
            # Then destroy the engine
            if hasattr(self, 'engine') and self.engine:
                del self.engine
                self.engine = None
        except:
            pass
        
        try:
            # Free GPU buffers before destroying CUDA context
            if hasattr(self, 'tensor_buffers'):
                for tensor_name, buffers in self.tensor_buffers.items():
                    if 'gpu' in buffers and buffers['gpu']:
                        try:
                            buffers['gpu'].free()
                        except:
                            pass
                self.tensor_buffers.clear()
        except:
            pass
        
        try:
            # Clear old-style buffers if they exist
            if hasattr(self, 'gpu_buffers'):
                for gpu_buffer in self.gpu_buffers:
                    try:
                        if gpu_buffer:
                            gpu_buffer.free()
                    except:
                        pass
                self.gpu_buffers.clear()
        except:
            pass
        
        try:
            # Pop CUDA context and remove from global list
            if hasattr(self, 'cuda_context') and self.cuda_context:
                if self.cuda_context in _active_contexts:
                    _active_contexts.remove(self.cuda_context)
                self.cuda_context.pop()
                self.cuda_context = None
        except Exception as e:
            # If context is already invalid, just remove from list
            if hasattr(self, 'cuda_context') and self.cuda_context in _active_contexts:
                _active_contexts.remove(self.cuda_context)
            pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __del__(self):
        """Cleanup resources during garbage collection"""
        try:
            self.cleanup()
        except:
            pass

if __name__ == "__main__":
    parser = ArgumentParser(description='TensorRT SigNet Inference')
    parser.add_argument('--engine_path', type=str, default='model.trt',
                       help='Path to TensorRT engine file')
    parser.add_argument('--image1', type=str, 
                       default='data/CEDAR/full_org/original_1_1.png',
                       help='Path to first signature image')
    parser.add_argument('--image2', type=str, 
                       default='data/CEDAR/full_org/original_1_12.png',
                       help='Path to second signature image')
    parser.add_argument('--threshold', type=float, default=0.0641,
                       help='Distance threshold for classification')
    parser.add_argument('--img_height', type=int, default=155,
                       help='Image height for preprocessing')
    parser.add_argument('--img_width', type=int, default=220,
                       help='Image width for preprocessing')
    parser.add_argument('--benchmark', action='store_true', default=True,
                       help='Run performance benchmark')
    parser.add_argument('--benchmark_iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--benchmark_batch_sizes', nargs='+', type=int, default=[1],
                       help='Batch sizes for benchmarking')
    parser.add_argument('--batch_test', action='store_true',
                       help='Test batch processing')
    parser.add_argument('--evaluate_dataset', type=str,
                       help='Path to dataset file for evaluation (JSON format)')
    parser.add_argument('--engine_info', action='store_true',
                       help='Display engine information')
    parser.add_argument('--output_json', type=str,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize TensorRT inferencer
    try:
        with TensorRTSigNetInference(
            engine_path=args.engine_path,
            threshold=args.threshold,
            img_height=args.img_height,
            img_width=args.img_width
        ) as inferencer:
            
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
            
            # Dataset evaluation
            if args.evaluate_dataset:
                logger.info("=== Dataset Evaluation ===")
                try:
                    # Load dataset from JSON file
                    with open(args.evaluate_dataset, 'r') as f:
                        dataset_data = json.load(f)
                    
                    # Expected format: [{"image1": "path1", "image2": "path2", "label": 0/1}, ...]
                    dataset_pairs = [(item['image1'], item['image2'], item['label']) for item in dataset_data]
                    
                    # Prepare output path for detailed results
                    eval_output_path = None
                    if args.output_json:
                        base_name = os.path.splitext(args.output_json)[0]
                        eval_output_path = f"{base_name}_evaluation_details.json"
                    
                    eval_results = inferencer.evaluate_dataset(dataset_pairs, eval_output_path)
                    
                    print(f"\nDataset Evaluation Results:")
                    print(f"  Total pairs: {eval_results['total_pairs']}")
                    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
                    print(f"  Precision: {eval_results['precision']:.4f}")
                    print(f"  Recall: {eval_results['recall']:.4f}")
                    print(f"  F1-Score: {eval_results['f1_score']:.4f}")
                    print(f"  Equal Error Rate: {eval_results['equal_error_rate']:.4f}")
                    print(f"  EER Threshold: {eval_results['eer_threshold']:.4f}")
                    print(f"  Current Threshold: {eval_results['current_threshold']:.4f}")
                    print(f"  Evaluation Time: {eval_results['evaluation_time_s']:.2f} s")
                    print(f"  Processing Speed: {eval_results['pairs_per_second']:.1f} pairs/s")
                    
                    results['dataset_evaluation'] = eval_results
                    
                except Exception as e:
                    logger.error(f"Dataset evaluation failed: {e}")
            
            # Save results
            if args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {args.output_json}")
            
            logger.info("=== TensorRT Inference Complete ===")
            
    except Exception as e:
        logger.error(f"Failed to initialize TensorRT inferencer: {e}")
        exit(1) 