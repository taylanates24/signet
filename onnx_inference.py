import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import os
import time
from argparse import ArgumentParser
from typing import Union, Tuple, List, Dict
import json

class ONNXSigNetInference:
    def __init__(self, model_path: str, threshold: float = 0.5, 
                 img_height: int = 155, img_width: int = 220,
                 providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']):
        """
        Initialize ONNX SigNet inference class
        
        Args:
            model_path (str): Path to the ONNX model file
            threshold (float): Distance threshold for genuine/forged classification
            img_height (int): Image height for preprocessing
            img_width (int): Image width for preprocessing
            providers (List[str]): ONNX Runtime execution providers
        """
        self.model_path = model_path
        self.threshold = threshold
        self.img_height = img_height
        self.img_width = img_width
        
        # Set up execution providers
        if providers is None:
            # Try GPU first, fall back to CPU
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        self.providers = providers
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model input/output info
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Setup image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            ImageOps.invert,
            transforms.ToTensor(),
        ])
        
        print(f"ONNX model loaded: {model_path}")
        print(f"Execution providers: {self.session.get_providers()}")
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")
        print(f"Image dimensions: {self.img_height}x{self.img_width}")
        print(f"Threshold: {self.threshold}")

    def preprocess_image(self, image_path: Union[str, Image.Image]) -> np.ndarray:
        """
        Preprocess a single image for ONNX inference
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            np.ndarray: Preprocessed image array ready for ONNX inference
        """
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path).convert('L')
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('L')
        else:
            raise ValueError("image_path must be a string path or PIL Image")
        
        # Apply transforms and convert to numpy
        image_tensor = self.transform(image)
        image_array = image_tensor.unsqueeze(0).numpy()  # Add batch dimension
        
        return image_array

    def predict_pair(self, image1: Union[str, Image.Image], 
                    image2: Union[str, Image.Image]) -> Dict:
        """
        Predict if two signatures are from the same person
        
        Args:
            image1: First signature image (path or PIL Image)
            image2: Second signature image (path or PIL Image)
            
        Returns:
            Dict: Prediction results containing distance, prediction, etc.
        """
        start_time = time.time()
        
        # Preprocess images
        img1_array = self.preprocess_image(image1)
        img2_array = self.preprocess_image(image2)
        
        # Prepare inputs for ONNX Runtime
        ort_inputs = {
            self.input_names[0]: img1_array,
            self.input_names[1]: img2_array
        }
        
        # Run inference
        inference_start = time.time()
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        inference_time = time.time() - inference_start
        
        # Extract embeddings
        embedding1 = ort_outputs[0][0]  # Remove batch dimension
        embedding2 = ort_outputs[1][0]
        
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
            image_pairs: List of tuples (image1, image2)
            
        Returns:
            Dict: Batch prediction results with summary
        """
        results = []
        total_start = time.time()
        
        for i, (img1, img2) in enumerate(image_pairs):
            print(f"Processing pair {i+1}/{len(image_pairs)}")
            result = self.predict_pair(img1, img2)
            result['pair_index'] = i
            result['image1_path'] = str(img1) if isinstance(img1, str) else f"PIL_Image_{i}_1"
            result['image2_path'] = str(img2) if isinstance(img2, str) else f"PIL_Image_{i}_2"
            results.append(result)
        
        total_time = time.time() - total_start
        
        # Add summary statistics
        distances = [r['distance'] for r in results]
        genuine_count = sum(1 for r in results if r['prediction'] == 'genuine')
        
        summary = {
            'total_pairs': len(image_pairs),
            'genuine_predictions': genuine_count,
            'forged_predictions': len(image_pairs) - genuine_count,
            'avg_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'total_processing_time_ms': total_time * 1000,
            'avg_time_per_pair_ms': (total_time * 1000) / len(image_pairs)
        }
        
        return {
            'results': results,
            'summary': summary
        }

    def benchmark(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark the ONNX model performance
        
        Args:
            num_iterations: Number of inference iterations to run
            
        Returns:
            Dict: Benchmark results
        """
        print(f"Running benchmark with {num_iterations} iterations...")
        
        # Create dummy inputs
        dummy_img1 = np.random.randn(1, 1, self.img_height, self.img_width).astype(np.float32)
        dummy_img2 = np.random.randn(1, 1, self.img_height, self.img_width).astype(np.float32)
        
        ort_inputs = {
            self.input_names[0]: dummy_img1,
            self.input_names[1]: dummy_img2
        }
        
        # Warmup
        for _ in range(10):
            self.session.run(self.output_names, ort_inputs)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.session.run(self.output_names, ort_inputs)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        return {
            'num_iterations': num_iterations,
            'avg_inference_time_ms': np.mean(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'std_inference_time_ms': np.std(times),
            'throughput_fps': 1000 / np.mean(times),
            'providers': self.session.get_providers()
        }

    def set_threshold(self, new_threshold: float):
        """Update the classification threshold"""
        self.threshold = new_threshold
        print(f"Threshold updated to: {self.threshold}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded ONNX model"""
        inputs_info = []
        for inp in self.session.get_inputs():
            inputs_info.append({
                'name': inp.name,
                'type': inp.type,
                'shape': inp.shape
            })
        
        outputs_info = []
        for out in self.session.get_outputs():
            outputs_info.append({
                'name': out.name,
                'type': out.type,
                'shape': out.shape
            })
        
        return {
            'model_path': self.model_path,
            'providers': self.session.get_providers(),
            'inputs': inputs_info,
            'outputs': outputs_info,
            'threshold': self.threshold,
            'image_dimensions': [self.img_height, self.img_width]
        }

def evaluate_on_dataset(inferencer: ONNXSigNetInference, 
                       dataset_pairs: List[Tuple[str, str, int]]) -> Dict:
    """
    Evaluate the ONNX model on a dataset
    
    Args:
        inferencer: ONNXSigNetInference instance
        dataset_pairs: List of (img1_path, img2_path, label) where label is 1 for genuine, 0 for forged
        
    Returns:
        Dict: Evaluation metrics
    """
    print(f"Evaluating on {len(dataset_pairs)} pairs...")
    
    correct_predictions = 0
    genuine_correct = 0
    genuine_total = 0
    forged_correct = 0
    forged_total = 0
    
    all_distances = []
    all_labels = []
    all_predictions = []
    
    start_time = time.time()
    
    for img1_path, img2_path, label in dataset_pairs:
        result = inferencer.predict_pair(img1_path, img2_path)
        
        distance = result['distance']
        prediction = 1 if result['prediction'] == 'genuine' else 0
        
        all_distances.append(distance)
        all_labels.append(label)
        all_predictions.append(prediction)
        
        if prediction == label:
            correct_predictions += 1
        
        if label == 1:  # Genuine
            genuine_total += 1
            if prediction == 1:
                genuine_correct += 1
        else:  # Forged
            forged_total += 1
            if prediction == 0:
                forged_correct += 1
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    overall_accuracy = correct_predictions / len(dataset_pairs)
    genuine_accuracy = genuine_correct / genuine_total if genuine_total > 0 else 0
    forged_accuracy = forged_correct / forged_total if forged_total > 0 else 0
    
    # Calculate FAR and FRR
    false_accepts = forged_total - forged_correct
    false_rejects = genuine_total - genuine_correct
    
    far = false_accepts / forged_total if forged_total > 0 else 0
    frr = false_rejects / genuine_total if genuine_total > 0 else 0
    
    return {
        'threshold': inferencer.threshold,
        'total_samples': len(dataset_pairs),
        'correct_predictions': correct_predictions,
        'overall_accuracy': overall_accuracy,
        'genuine_accuracy': genuine_accuracy,
        'forged_accuracy': forged_accuracy,
        'false_accept_rate': far,
        'false_reject_rate': frr,
        'genuine_samples': genuine_total,
        'forged_samples': forged_total,
        'avg_distance': np.mean(all_distances),
        'evaluation_time_s': total_time,
        'avg_time_per_pair_ms': (total_time * 1000) / len(dataset_pairs)
    }

def create_dataset_from_directory(data_dir: str) -> List[Tuple[str, str, int]]:
    """
    Create dataset pairs from directory structure
    Expected structure:
    data_dir/
    ├── genuine_pairs.txt  # or similar file with pairs
    ├── forged_pairs.txt
    └── images/
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        List of (img1_path, img2_path, label) tuples
    """
    # This is a placeholder function - you'll need to adapt it to your dataset structure
    pairs = []
    
    # Example implementation for CEDAR dataset structure
    if 'CEDAR' in data_dir:
        genuine_dir = os.path.join(data_dir, 'full_org')
        forged_dir = os.path.join(data_dir, 'full_forg')
        
        if os.path.exists(genuine_dir) and os.path.exists(forged_dir):
            genuine_files = [f for f in os.listdir(genuine_dir) if f.endswith('.png')]
            forged_files = [f for f in os.listdir(forged_dir) if f.endswith('.png')]
            
            # Create some example pairs (you should implement proper pairing logic)
            for i in range(min(10, len(genuine_files) - 1)):
                # Genuine pair
                pairs.append((
                    os.path.join(genuine_dir, genuine_files[i]),
                    os.path.join(genuine_dir, genuine_files[i + 1]),
                    1
                ))
                
                # Forged pair (genuine vs forged)
                if i < len(forged_files):
                    pairs.append((
                        os.path.join(genuine_dir, genuine_files[i]),
                        os.path.join(forged_dir, forged_files[i]),
                        0
                    ))
    
    return pairs

if __name__ == "__main__":
    parser = ArgumentParser(description='ONNX SigNet Inference Script')
    parser.add_argument('--model_path', type=str, default='model.onnx',
                       help='Path to ONNX model file')
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
    parser.add_argument('--providers', nargs='+', 
                       default=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                       help='ONNX Runtime execution providers')
    parser.add_argument('--benchmark', action='store_true', default=True,
                       help='Run performance benchmark')
    parser.add_argument('--benchmark_iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--batch_test', action='store_true',
                       help='Test batch processing with sample images')
    parser.add_argument('--evaluate_dataset', type=str,
                       help='Path to dataset directory for evaluation')
    parser.add_argument('--output_json', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--model_info', action='store_true',
                       help='Display model information')
    
    args = parser.parse_args()
    
    # Initialize inferencer
    inferencer = ONNXSigNetInference(
        model_path=args.model_path,
        threshold=args.threshold,
        img_height=args.img_height,
        img_width=args.img_width,
        providers=args.providers
    )
    
    results = {}
    
    # Single pair prediction
    if args.image1 and args.image2:
        print(f"\n=== Single Pair Prediction ===")
        print(f"Image 1: {args.image1}")
        print(f"Image 2: {args.image2}")
        
        result = inferencer.predict_pair(args.image1, args.image2)
        
        print(f"\nResults:")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Prediction: {result['prediction']}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        
        results['single_prediction'] = result
    
    # Model info
    if args.model_info:
        print(f"\n=== Model Information ===")
        model_info = inferencer.get_model_info()
        for key, value in model_info.items():
            print(f"{key}: {value}")
        results['model_info'] = model_info
    
    # Benchmark
    if args.benchmark:
        print(f"\n=== Performance Benchmark ===")
        benchmark_results = inferencer.benchmark(args.benchmark_iterations)
        
        print(f"Average inference time: {benchmark_results['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
        print(f"Min/Max time: {benchmark_results['min_inference_time_ms']:.2f}/{benchmark_results['max_inference_time_ms']:.2f} ms")
        
        results['benchmark'] = benchmark_results
    
    # Batch test
    if args.batch_test and args.image1 and args.image2:
        print(f"\n=== Batch Processing Test ===")
        test_pairs = [(args.image1, args.image2)] * 5  # Test with same pair 5 times
        batch_results = inferencer.predict_batch(test_pairs)
        
        print(f"Processed {batch_results['summary']['total_pairs']} pairs")
        print(f"Average time per pair: {batch_results['summary']['avg_time_per_pair_ms']:.2f} ms")
        print(f"Genuine predictions: {batch_results['summary']['genuine_predictions']}")
        print(f"Forged predictions: {batch_results['summary']['forged_predictions']}")
        
        results['batch_test'] = batch_results
    
    # Dataset evaluation
    if args.evaluate_dataset:
        print(f"\n=== Dataset Evaluation ===")
        dataset_pairs = create_dataset_from_directory(args.evaluate_dataset)
        
        if dataset_pairs:
            eval_results = evaluate_on_dataset(inferencer, dataset_pairs)
            
            print(f"Overall Accuracy: {eval_results['overall_accuracy']:.4f}")
            print(f"Genuine Accuracy: {eval_results['genuine_accuracy']:.4f}")
            print(f"Forged Accuracy: {eval_results['forged_accuracy']:.4f}")
            print(f"False Accept Rate: {eval_results['false_accept_rate']:.4f}")
            print(f"False Reject Rate: {eval_results['false_reject_rate']:.4f}")
            
            results['evaluation'] = eval_results
        else:
            print("No dataset pairs found. Please check dataset structure.")
    
    # Save results to JSON
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    
    print("\n=== Inference Complete ===") 