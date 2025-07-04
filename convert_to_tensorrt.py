import tensorrt as trt
import onnx
import numpy as np
import os
import time
from argparse import ArgumentParser
import logging
import json
from typing import Optional, List, Tuple, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorRTConverter:
    def __init__(self, onnx_path: str, engine_path: str = None):
        """
        Initialize TensorRT converter
        
        Args:
            onnx_path (str): Path to ONNX model
            engine_path (str): Path to save TensorRT engine (optional)
        """
        self.onnx_path = onnx_path
        self.engine_path = engine_path or onnx_path.replace('.onnx', '.trt')
        
        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        
        # Verify ONNX model
        self._verify_onnx_model()
        
    def _verify_onnx_model(self):
        """Verify the ONNX model is valid"""
        try:
            model = onnx.load(self.onnx_path)
            onnx.checker.check_model(model)
            logger.info(f"ONNX model {self.onnx_path} is valid")
        except Exception as e:
            raise ValueError(f"Invalid ONNX model: {e}")
    
    def convert_to_tensorrt(self, 
                           max_batch_size: int = 1,
                           precision: str = 'fp16',
                           workspace_size: int = 1 << 30,  # 1GB
                           dynamic_shapes: bool = False,
                           optimization_level: int = 5,
                           min_shapes: Optional[Dict] = None,
                           opt_shapes: Optional[Dict] = None,
                           max_shapes: Optional[Dict] = None) -> str:

        logger.info(f"Converting ONNX model to TensorRT...")
        logger.info(f"Input: {self.onnx_path}")
        logger.info(f"Output: {self.engine_path}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Max batch size: {max_batch_size}")
        logger.info(f'Workspace size: {workspace_size / (1024**3):.1f} GB')
        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Parse ONNX model
        with open(self.onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Create builder config
        config = builder.create_builder_config()
        
        # Set workspace size (handle different TensorRT versions)
        try:
            # TensorRT 8.5+
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        except AttributeError:
            # TensorRT < 8.5
            config.max_workspace_size = workspace_size
        
        # Set optimization level
        config.builder_optimization_level = optimization_level
        
        # Set precision
        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            else:
                logger.warning("FP16 not supported on this platform, using FP32")
        elif precision == 'int8':
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # Note: INT8 calibration would be needed here for real use
                logger.info("INT8 precision enabled (requires calibration)")
            else:
                logger.warning("INT8 not supported on this platform, using FP32")
        
        # Handle dynamic shapes
        if dynamic_shapes:
            profile = builder.create_optimization_profile()
            
            # Get input information
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                input_name = input_tensor.name
                
                if min_shapes and input_name in min_shapes:
                    min_shape = min_shapes[input_name]
                    opt_shape = opt_shapes.get(input_name, min_shape) if opt_shapes else min_shape
                    max_shape = max_shapes.get(input_name, min_shape) if max_shapes else min_shape
                else:
                    # Default shapes for SigNet (batch, channels, height, width)
                    min_shape = (1, 1, 155, 220)
                    opt_shape = (1, 1, 155, 220)
                    max_shape = (max_batch_size, 1, 155, 220)
                
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.info(f"Input {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            
            config.add_optimization_profile(profile)
        else:
            # Set maximum batch size for static shapes (TensorRT < 8.4)
            try:
                builder.max_batch_size = max_batch_size
            except AttributeError:
                # In newer TensorRT versions, batch size is handled in network definition
                logger.info("Note: max_batch_size handled via network definition in this TensorRT version")
        
        # Build engine
        logger.info("Building TensorRT engine... This may take several minutes.")
        start_time = time.time()
        
        engine = builder.build_engine(network, config)
        
        build_time = time.time() - start_time
        logger.info(f"Engine built in {build_time:.1f} seconds")
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Serialize and save engine
        with open(self.engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved to: {self.engine_path}")
        
        # Print engine info
        self._print_engine_info(engine)
        
        return self.engine_path
    
    def _print_engine_info(self, engine):
        """Print information about the TensorRT engine"""
        logger.info("=== TensorRT Engine Information ===")
        
        # Handle different TensorRT versions for max_batch_size
        try:
            logger.info(f"Max batch size: {engine.max_batch_size}")
        except AttributeError:
            logger.info("Max batch size: Determined by input shapes (TensorRT 8.5+)")
            
        logger.info(f"Number of bindings: {engine.num_bindings}")
        
        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            binding_shape = engine.get_binding_shape(i)
            binding_dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            
            logger.info(f"Binding {i}: {binding_name}")
            logger.info(f"  Shape: {binding_shape}")
            logger.info(f"  Type: {binding_dtype}")
            logger.info(f"  Input: {is_input}")
    
    def benchmark_conversion(self, 
                           precisions: List[str] = ['fp32', 'fp16'],
                           batch_sizes: List[int] = [1, 4, 8],
                           workspace_sizes: List[int] = [1<<28, 1<<29, 1<<30]) -> Dict:
        """
        Benchmark different conversion configurations
        
        Args:
            precisions: List of precision modes to test
            batch_sizes: List of batch sizes to test
            workspace_sizes: List of workspace sizes to test
            
        Returns:
            Dict: Benchmark results
        """
        logger.info("Starting conversion benchmark...")
        results = []
        
        for precision in precisions:
            for batch_size in batch_sizes:
                for workspace_size in workspace_sizes:
                    try:
                        logger.info(f"Testing: precision={precision}, batch={batch_size}, workspace={workspace_size>>20}MB")
                        
                        # Create temporary engine path
                        temp_engine = f"temp_{precision}_b{batch_size}_w{workspace_size>>20}.trt"
                        
                        start_time = time.time()
                        
                        # Convert
                        self.engine_path = temp_engine
                        self.convert_to_tensorrt(
                            max_batch_size=batch_size,
                            precision=precision,
                            workspace_size=workspace_size
                        )
                        
                        build_time = time.time() - start_time
                        engine_size = os.path.getsize(temp_engine) / (1024**2)  # MB
                        
                        result = {
                            'precision': precision,
                            'batch_size': batch_size,
                            'workspace_size_mb': workspace_size >> 20,
                            'build_time_s': build_time,
                            'engine_size_mb': engine_size,
                            'success': True
                        }
                        
                        logger.info(f"  Build time: {build_time:.1f}s, Engine size: {engine_size:.1f}MB")
                        
                        # Clean up
                        if os.path.exists(temp_engine):
                            os.remove(temp_engine)
                            
                    except Exception as e:
                        result = {
                            'precision': precision,
                            'batch_size': batch_size,
                            'workspace_size_mb': workspace_size >> 20,
                            'error': str(e),
                            'success': False
                        }
                        logger.error(f"  Failed: {e}")
                    
                    results.append(result)
        
        return {
            'results': results,
            'summary': self._summarize_benchmark(results)
        }
    
    def _summarize_benchmark(self, results: List[Dict]) -> Dict:
        """Summarize benchmark results"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if not successful:
            return {'total': len(results), 'successful': 0, 'failed': len(failed)}
        
        build_times = [r['build_time_s'] for r in successful]
        engine_sizes = [r['engine_size_mb'] for r in successful]
        
        return {
            'total': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'avg_build_time_s': np.mean(build_times),
            'min_build_time_s': np.min(build_times),
            'max_build_time_s': np.max(build_times),
            'avg_engine_size_mb': np.mean(engine_sizes),
            'min_engine_size_mb': np.min(engine_sizes),
            'max_engine_size_mb': np.max(engine_sizes)
        }

def get_onnx_model_info(onnx_path: str) -> Dict:
    """Get information about an ONNX model"""
    model = onnx.load(onnx_path)
    
    inputs = []
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        inputs.append({
            'name': input_tensor.name,
            'shape': shape,
            'type': input_tensor.type.tensor_type.elem_type
        })
    
    outputs = []
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        outputs.append({
            'name': output_tensor.name,
            'shape': shape,
            'type': output_tensor.type.tensor_type.elem_type
        })
    
    return {
        'inputs': inputs,
        'outputs': outputs,
        'producer_name': model.producer_name,
        'producer_version': model.producer_version,
        'model_version': model.model_version
    }

def check_tensorrt_support():
    """Check TensorRT installation and GPU support"""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Check CUDA
        cuda.init()
        device_count = cuda.Device.count()
        logger.info(f"Found {device_count} CUDA devices")
        
        if device_count > 0:
            device = cuda.Device(0)
            logger.info(f"Primary GPU: {device.name()}")
            
            context = device.make_context()
            free_mem, total_mem = cuda.mem_get_info()
            context.pop()
            
            logger.info(f"GPU Memory: {free_mem/(1024**3):.1f}GB free / {total_mem/(1024**3):.1f}GB total")
        
        # Check TensorRT version and compatibility
        trt_version = trt.__version__
        logger.info(f"TensorRT version: {trt_version}")
        
        # Check for version-specific features
        major_version = int(trt_version.split('.')[0])
        minor_version = int(trt_version.split('.')[1]) if '.' in trt_version else 0
        
        if major_version >= 8 and minor_version >= 5:
            logger.info("TensorRT 8.5+ detected: Using new memory pool API")
        else:
            logger.info("TensorRT < 8.5 detected: Using legacy workspace API")
            
        # Test basic TensorRT functionality
        try:
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            config = builder.create_builder_config()
            
            # Test memory pool API
            try:
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024*1024*1024)
                logger.info("✓ Memory pool API available")
            except AttributeError:
                logger.info("✓ Legacy workspace API available")
                
            logger.info("✓ TensorRT builder functional")
            
        except Exception as e:
            logger.error(f"TensorRT builder test failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False
    except Exception as e:
        logger.error(f"GPU/TensorRT check failed: {e}")
        return False

if __name__ == "__main__":
    parser = ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('--onnx_path', type=str, default='signet.onnx',
                       help='Path to ONNX model file')
    parser.add_argument('--engine_path', type=str, default='signet.trt',
                       help='Path to save TensorRT engine')
    parser.add_argument('--max_batch_size', type=int, default=1,
                       help='Maximum batch size')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], default='fp16',
                       help='Precision mode')
    parser.add_argument('--workspace_size_mb', type=int, default=4096,
                       help='Workspace size in MB')
    parser.add_argument('--dynamic_shapes', action='store_true',
                       help='Enable dynamic shapes')
    parser.add_argument('--optimization_level', type=int, default=5, choices=range(6),
                       help='Optimization level (0-5)')
    parser.add_argument('--benchmark', action='store_true', default=True,
                       help='Run conversion benchmark')
    parser.add_argument('--model_info', action='store_true',
                       help='Display ONNX model information')
    parser.add_argument('--check_support', action='store_true',
                       help='Check TensorRT and GPU support')
    parser.add_argument('--output_json', type=str,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    results = {}
    
    # Check TensorRT support
    if args.check_support:
        logger.info("=== Checking TensorRT Support ===")
        support_ok = check_tensorrt_support()
        results['support_check'] = support_ok
        if not support_ok:
            logger.error("TensorRT support check failed. Please install required dependencies.")
            exit(1)
    
    # Display model info
    if args.model_info:
        logger.info("=== ONNX Model Information ===")
        try:
            model_info = get_onnx_model_info(args.onnx_path)
            for key, value in model_info.items():
                logger.info(f"{key}: {value}")
            results['model_info'] = model_info
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
    
    # Initialize converter
    try:
        converter = TensorRTConverter(args.onnx_path, args.engine_path)
    except Exception as e:
        logger.error(f"Failed to initialize converter: {e}")
        exit(1)
    
    # Run benchmark if requested
    if args.benchmark:
        logger.info("=== Running Conversion Benchmark ===")
        try:
            benchmark_results = converter.benchmark_conversion()
            logger.info(f"Benchmark completed:")
            logger.info(f"- Total configurations: {benchmark_results['summary']['total']}")
            logger.info(f"- Successful: {benchmark_results['summary']['successful']}")
            logger.info(f"- Failed: {benchmark_results['summary']['failed']}")
            
            if benchmark_results['summary']['successful'] > 0:
                logger.info(f"- Avg build time: {benchmark_results['summary']['avg_build_time_s']:.1f}s")
                logger.info(f"- Avg engine size: {benchmark_results['summary']['avg_engine_size_mb']:.1f}MB")
            
            results['benchmark'] = benchmark_results
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
    
    # Convert model
    logger.info("=== Converting ONNX to TensorRT ===")
    try:
        workspace_size = args.workspace_size_mb * 1024 * 1024  # Convert MB to bytes
        
        engine_path = converter.convert_to_tensorrt(
            max_batch_size=args.max_batch_size,
            precision=args.precision,
            workspace_size=workspace_size,
            dynamic_shapes=args.dynamic_shapes,
            optimization_level=args.optimization_level
        )
        
        engine_size_mb = os.path.getsize(engine_path) / (1024**2)
        
        conversion_result = {
            'engine_path': engine_path,
            'engine_size_mb': engine_size_mb,
            'max_batch_size': args.max_batch_size,
            'precision': args.precision,
            'workspace_size_mb': args.workspace_size_mb,
            'dynamic_shapes': args.dynamic_shapes,
            'optimization_level': args.optimization_level
        }
        
        logger.info(f"Conversion successful!")
        logger.info(f"Engine size: {engine_size_mb:.1f} MB")
        
        results['conversion'] = conversion_result
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        results['conversion'] = {'error': str(e)}
        exit(1)
    
    # Save results to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output_json}")
    
    logger.info("=== Conversion Complete ===")
    logger.info(f"TensorRT engine saved to: {args.engine_path}") 