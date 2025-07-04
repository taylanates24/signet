from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import os
import logging
from onnx_inference import ONNXSigNetInference
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global inference object
inferencer = None

def initialize_model(model_path='signet.onnx', threshold=0.5):
    """Initialize the ONNX model"""
    global inferencer
    try:
        inferencer = ONNXSigNetInference(
            model_path=model_path,
            threshold=threshold
        )
        logger.info(f"Model initialized successfully: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inferencer is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict signature verification
    
    Expected JSON payload:
    {
        "image1": "base64_encoded_image_or_file_path",
        "image2": "base64_encoded_image_or_file_path",
        "threshold": 0.5  // optional, overrides default
    }
    """
    try:
        if inferencer is None:
            return jsonify({
                'error': 'Model not initialized'
            }), 500
        
        data = request.get_json()
        
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({
                'error': 'Missing required fields: image1, image2'
            }), 400
        
        # Get images
        image1_data = data['image1']
        image2_data = data['image2']
        
        # Handle different input types
        if image1_data.startswith('data:') or (len(image1_data) > 100 and not os.path.exists(image1_data)):
            # Base64 encoded image
            image1 = decode_base64_image(image1_data)
            if image1 is None:
                return jsonify({'error': 'Invalid base64 image1'}), 400
        else:
            # File path
            if not os.path.exists(image1_data):
                return jsonify({'error': f'Image file not found: {image1_data}'}), 400
            image1 = image1_data
        
        if image2_data.startswith('data:') or (len(image2_data) > 100 and not os.path.exists(image2_data)):
            # Base64 encoded image
            image2 = decode_base64_image(image2_data)
            if image2 is None:
                return jsonify({'error': 'Invalid base64 image2'}), 400
        else:
            # File path
            if not os.path.exists(image2_data):
                return jsonify({'error': f'Image file not found: {image2_data}'}), 400
            image2 = image2_data
        
        # Set threshold if provided
        original_threshold = inferencer.threshold
        if 'threshold' in data:
            inferencer.set_threshold(data['threshold'])
        
        # Run prediction
        result = inferencer.predict_pair(image1, image2)
        
        # Restore original threshold
        if 'threshold' in data:
            inferencer.set_threshold(original_threshold)
        
        # Remove embeddings from response to reduce size (optional)
        if 'include_embeddings' not in data or not data['include_embeddings']:
            result.pop('embedding1', None)
            result.pop('embedding2', None)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple signature pairs
    
    Expected JSON payload:
    {
        "pairs": [
            {
                "image1": "base64_or_path",
                "image2": "base64_or_path"
            },
            ...
        ],
        "threshold": 0.5  // optional
    }
    """
    try:
        if inferencer is None:
            return jsonify({
                'error': 'Model not initialized'
            }), 500
        
        data = request.get_json()
        
        if not data or 'pairs' not in data:
            return jsonify({
                'error': 'Missing required field: pairs'
            }), 400
        
        pairs = data['pairs']
        if not isinstance(pairs, list) or len(pairs) == 0:
            return jsonify({
                'error': 'pairs must be a non-empty list'
            }), 400
        
        # Set threshold if provided
        original_threshold = inferencer.threshold
        if 'threshold' in data:
            inferencer.set_threshold(data['threshold'])
        
        # Process pairs
        processed_pairs = []
        for i, pair in enumerate(pairs):
            if 'image1' not in pair or 'image2' not in pair:
                return jsonify({
                    'error': f'Pair {i} missing image1 or image2'
                }), 400
            
            # Process image1
            if pair['image1'].startswith('data:') or (len(pair['image1']) > 100 and not os.path.exists(pair['image1'])):
                image1 = decode_base64_image(pair['image1'])
                if image1 is None:
                    return jsonify({'error': f'Invalid base64 image1 in pair {i}'}), 400
            else:
                if not os.path.exists(pair['image1']):
                    return jsonify({'error': f'Image file not found: {pair["image1"]} in pair {i}'}), 400
                image1 = pair['image1']
            
            # Process image2
            if pair['image2'].startswith('data:') or (len(pair['image2']) > 100 and not os.path.exists(pair['image2'])):
                image2 = decode_base64_image(pair['image2'])
                if image2 is None:
                    return jsonify({'error': f'Invalid base64 image2 in pair {i}'}), 400
            else:
                if not os.path.exists(pair['image2']):
                    return jsonify({'error': f'Image file not found: {pair["image2"]} in pair {i}'}), 400
                image2 = pair['image2']
            
            processed_pairs.append((image1, image2))
        
        # Run batch prediction
        results = inferencer.predict_batch(processed_pairs)
        
        # Restore original threshold
        if 'threshold' in data:
            inferencer.set_threshold(original_threshold)
        
        # Remove embeddings from response to reduce size (optional)
        if 'include_embeddings' not in data or not data['include_embeddings']:
            for result in results['results']:
                result.pop('embedding1', None)
                result.pop('embedding2', None)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if inferencer is None:
            return jsonify({
                'error': 'Model not initialized'
            }), 500
        
        info = inferencer.get_model_info()
        return jsonify({
            'success': True,
            'model_info': info
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': f'Failed to get model info: {str(e)}'
        }), 500

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """Set classification threshold"""
    try:
        if inferencer is None:
            return jsonify({
                'error': 'Model not initialized'
            }), 500
        
        data = request.get_json()
        
        if not data or 'threshold' not in data:
            return jsonify({
                'error': 'Missing required field: threshold'
            }), 400
        
        threshold = data['threshold']
        if not isinstance(threshold, (int, float)) or threshold < 0:
            return jsonify({
                'error': 'threshold must be a positive number'
            }), 400
        
        old_threshold = inferencer.threshold
        inferencer.set_threshold(threshold)
        
        return jsonify({
            'success': True,
            'old_threshold': old_threshold,
            'new_threshold': inferencer.threshold
        })
        
    except Exception as e:
        logger.error(f"Set threshold error: {str(e)}")
        return jsonify({
            'error': f'Failed to set threshold: {str(e)}'
        }), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Run performance benchmark"""
    try:
        if inferencer is None:
            return jsonify({
                'error': 'Model not initialized'
            }), 500
        
        data = request.get_json() or {}
        iterations = data.get('iterations', 100)
        
        if not isinstance(iterations, int) or iterations < 1:
            return jsonify({
                'error': 'iterations must be a positive integer'
            }), 400
        
        benchmark_results = inferencer.benchmark(iterations)
        
        return jsonify({
            'success': True,
            'benchmark': benchmark_results
        })
        
    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}")
        return jsonify({
            'error': f'Benchmark failed: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX SigNet API Server')
    parser.add_argument('--model_path', type=str, default='signet.onnx',
                       help='Path to ONNX model file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Default classification threshold')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize model
    if not initialize_model(args.model_path, args.threshold):
        logger.error("Failed to initialize model. Exiting.")
        exit(1)
    
    # Start server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug) 