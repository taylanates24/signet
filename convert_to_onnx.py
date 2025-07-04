import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from model import SigNet
from torchvision import transforms
from PIL import Image, ImageOps
import os
from argparse import ArgumentParser

def load_model(model_path, device='cpu'):
    """
    Load the trained SigNet model from checkpoint
    
    Args:
        model_path (str): Path to the model checkpoint
        device (str): Device to load the model on
        
    Returns:
        tuple: (model, args) where args contains training parameters
    """
    model = SigNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Get training args if available
    args = checkpoint.get('args', None)
    
    return model, args

def create_dummy_inputs(batch_size=1, img_height=155, img_width=220, device='cpu'):
    """
    Create dummy inputs for ONNX export
    
    Args:
        batch_size (int): Batch size for dummy inputs
        img_height (int): Image height
        img_width (int): Image width
        device (str): Device to create tensors on
        
    Returns:
        tuple: (dummy_x1, dummy_x2) dummy input tensors
    """
    dummy_x1 = torch.randn(batch_size, 1, img_height, img_width, device=device)
    dummy_x2 = torch.randn(batch_size, 1, img_height, img_width, device=device)
    return dummy_x1, dummy_x2

def convert_to_onnx(model_path, output_path, batch_size=1, img_height=155, img_width=220, 
                   opset_version=11, dynamic_axes=False, device='cpu'):
    """
    Convert SigNet PyTorch model to ONNX format
    
    Args:
        model_path (str): Path to the PyTorch model checkpoint
        output_path (str): Path to save the ONNX model
        batch_size (int): Batch size for the model
        img_height (int): Image height
        img_width (int): Image width
        opset_version (int): ONNX opset version
        dynamic_axes (bool): Whether to use dynamic axes for variable batch size
        device (str): Device to use for conversion
    """
    print(f"Loading model from: {model_path}")
    model, args = load_model(model_path, device)
    
    # Override dimensions with training args if available
    if args:
        if hasattr(args, 'img_height'):
            img_height = args.img_height
        if hasattr(args, 'img_width'):
            img_width = args.img_width
    
    print(f"Using image dimensions: {img_height}x{img_width}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Create dummy inputs
    dummy_x1, dummy_x2 = create_dummy_inputs(batch_size, img_height, img_width, device)
    
    # Define input and output names
    input_names = ['signature1', 'signature2']
    output_names = ['embedding1', 'embedding2']
    
    # Define dynamic axes if requested
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'signature1': {0: 'batch_size'},
            'signature2': {0: 'batch_size'},
            'embedding1': {0: 'batch_size'},
            'embedding2': {0: 'batch_size'}
        }
    
    print("Converting model to ONNX...")
    
    # Export the model
    torch.onnx.export(
        model,
        (dummy_x1, dummy_x2),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict
    )
    
    print(f"Model successfully converted to ONNX: {output_path}")
    return output_path

def verify_onnx_model(onnx_path, pytorch_model_path, img_height=155, img_width=220, device='cpu'):
    """
    Verify that the ONNX model produces the same outputs as the PyTorch model
    
    Args:
        onnx_path (str): Path to the ONNX model
        pytorch_model_path (str): Path to the PyTorch model
        img_height (int): Image height
        img_width (int): Image width
        device (str): Device to use for verification
    """
    print("Verifying ONNX model...")
    
    # Load PyTorch model
    pytorch_model, args = load_model(pytorch_model_path, device)
    
    # Override dimensions with training args if available
    if args:
        if hasattr(args, 'img_height'):
            img_height = args.img_height
        if hasattr(args, 'img_width'):
            img_width = args.img_width
    
    # Create test inputs
    test_x1, test_x2 = create_dummy_inputs(1, img_height, img_width, device)
    
    # Get PyTorch outputs
    with torch.no_grad():
        pytorch_out1, pytorch_out2 = pytorch_model(test_x1, test_x2)
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Prepare inputs for ONNX Runtime
    ort_inputs = {
        'signature1': test_x1.cpu().numpy(),
        'signature2': test_x2.cpu().numpy()
    }
    
    # Get ONNX outputs
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare outputs
    torch_out1_np = pytorch_out1.cpu().numpy()
    torch_out2_np = pytorch_out2.cpu().numpy()
    
    # Calculate differences
    diff1 = np.abs(torch_out1_np - ort_outputs[0]).max()
    diff2 = np.abs(torch_out2_np - ort_outputs[1]).max()
    
    print(f"Maximum difference for embedding1: {diff1}")
    print(f"Maximum difference for embedding2: {diff2}")
    
    # Check if differences are within acceptable tolerance
    tolerance = 1e-5
    if diff1 < tolerance and diff2 < tolerance:
        print("✓ ONNX model verification successful!")
        return True
    else:
        print("✗ ONNX model verification failed!")
        return False

def test_onnx_inference(onnx_path, image1_path=None, image2_path=None, 
                       img_height=155, img_width=220):
    """
    Test ONNX model inference with real signature images
    
    Args:
        onnx_path (str): Path to the ONNX model
        image1_path (str): Path to first signature image
        image2_path (str): Path to second signature image
        img_height (int): Image height
        img_width (int): Image width
    """
    print("Testing ONNX inference...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    if image1_path and image2_path and os.path.exists(image1_path) and os.path.exists(image2_path):
        # Use real images
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            ImageOps.invert,
            transforms.ToTensor(),
        ])
        
        img1 = Image.open(image1_path).convert('L')
        img2 = Image.open(image2_path).convert('L')
        
        img1_tensor = transform(img1).unsqueeze(0)  # Add batch dimension
        img2_tensor = transform(img2).unsqueeze(0)
        
        print(f"Using real images: {image1_path}, {image2_path}")
    else:
        # Use dummy inputs
        img1_tensor = torch.randn(1, 1, img_height, img_width)
        img2_tensor = torch.randn(1, 1, img_height, img_width)
        print("Using dummy inputs for testing")
    
    # Prepare inputs for ONNX Runtime
    ort_inputs = {
        'signature1': img1_tensor.numpy(),
        'signature2': img2_tensor.numpy()
    }
    
    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"Embedding 1 shape: {ort_outputs[0].shape}")
    print(f"Embedding 2 shape: {ort_outputs[1].shape}")
    
    # Calculate Euclidean distance between embeddings
    embedding1 = ort_outputs[0][0]  # Remove batch dimension
    embedding2 = ort_outputs[1][0]
    distance = np.linalg.norm(embedding1 - embedding2)
    
    print(f"Euclidean distance between embeddings: {distance:.4f}")
    
    return ort_outputs

def get_model_info(onnx_path):
    """
    Print information about the ONNX model
    
    Args:
        onnx_path (str): Path to the ONNX model
    """
    print("ONNX Model Information:")
    print("-" * 40)
    
    # Load the model
    model = onnx.load(onnx_path)
    
    # Check the model
    onnx.checker.check_model(model)
    print("✓ Model is valid")
    
    # Print input information
    print("\nInputs:")
    for input_tensor in model.graph.input:
        print(f"  - {input_tensor.name}: {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}")
    
    # Print output information
    print("\nOutputs:")
    for output_tensor in model.graph.output:
        print(f"  - {output_tensor.name}: {[dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]}")
    
    # Print model size
    model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
    print(f"\nModel size: {model_size:.2f} MB")

if __name__ == "__main__":
    parser = ArgumentParser(description='Convert SigNet PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, default='experiments/signet_exp_20250703_101313/checkpoints/20250703_101313_epoch_000_loss_0.4860_acc_1.0000.pt', 
                       help='Path to the PyTorch model checkpoint')
    parser.add_argument('--output_path', type=str, default='signet.onnx',
                       help='Path to save the ONNX model')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for the model (default: 1)')
    parser.add_argument('--img_height', type=int, default=155,
                       help='Image height (default: 155)')
    parser.add_argument('--img_width', type=int, default=220,
                       help='Image width (default: 220)')
    parser.add_argument('--opset_version', type=int, default=19,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--dynamic_axes', action='store_true', default=True,
                       help='Use dynamic axes for variable batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                       help='Device to use for conversion (default: cpu)')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify ONNX model against PyTorch model')
    parser.add_argument('--test_inference', action='store_true', default=True,
                       help='Test ONNX model inference')
    parser.add_argument('--image1', type=str, default='data/CEDAR/full_org/original_38_1.png',
                       help='Path to first signature image for testing')
    parser.add_argument('--image2', type=str, default='data/CEDAR/full_org/original_38_3.png',
                       help='Path to second signature image for testing')
    parser.add_argument('--model_info', action='store_true', default=True,
                       help='Display ONNX model information')
    
    args = parser.parse_args()
    
    # Convert model to ONNX
    onnx_path = convert_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        img_height=args.img_height,
        img_width=args.img_width,
        opset_version=args.opset_version,
        dynamic_axes=args.dynamic_axes,
        device=args.device
    )
    
    # Verify the model if requested
    if args.verify:
        verify_onnx_model(
            onnx_path=onnx_path,
            pytorch_model_path=args.model_path,
            img_height=args.img_height,
            img_width=args.img_width,
            device=args.device
        )
    
    # Test inference if requested
    if args.test_inference:
        test_onnx_inference(
            onnx_path=onnx_path,
            image1_path=args.image1,
            image2_path=args.image2,
            img_height=args.img_height,
            img_width=args.img_width
        )
    
    # Display model info if requested
    if args.model_info:
        get_model_info(onnx_path)
    
    print(f"\nConversion complete! ONNX model saved to: {onnx_path}") 