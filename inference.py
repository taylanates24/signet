from model import SigNet
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import os
from argparse import ArgumentParser
from data import get_data_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SigNetInference:
    def __init__(self, model_path, threshold=0.5):
        """
        Initialize SigNet inference class
        
        Args:
            model_path (str): Path to the trained model checkpoint
            threshold (float): Distance threshold for genuine/forged classification
        """
        self.threshold = threshold
        self.device = device
        
        # Load model
        self.model = SigNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # Get image transform settings from checkpoint if available
        if 'args' in checkpoint:
            args = checkpoint['args']
            self.img_height = args.img_height if hasattr(args, 'img_height') else 155
            self.img_width = args.img_width if hasattr(args, 'img_width') else 220
        else:
            self.img_height = 155
            self.img_width = 220
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            ImageOps.invert,
            transforms.ToTensor(),
        ])
        
        print(f"Model loaded from: {model_path}")
        print(f"Using threshold: {self.threshold}")
        print(f"Image dimensions: {self.img_height}x{self.img_width}")
        print(f"Device: {self.device}")

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for inference
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('L')  # Convert to grayscale
        else:
            image = image_path  # Assume it's already a PIL image
            
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)  # Add batch dimension

    def predict_pair(self, image1_path, image2_path):
        """
        Predict if two signatures are from the same person
        
        Args:
            image1_path (str): Path to first signature image
            image2_path (str): Path to second signature image
            
        Returns:
            dict: Dictionary containing prediction results
        """
        with torch.no_grad():
            # Preprocess images
            img1 = self.preprocess_image(image1_path)
            img2 = self.preprocess_image(image2_path)
            
            # Get embeddings
            emb1, emb2 = self.model(img1, img2)
            
            # Calculate Euclidean distance
            distance = F.pairwise_distance(emb1, emb2, p=2).item()
            
            # Make prediction based on threshold
            is_genuine = distance < self.threshold 

            
            return {
                'distance': distance,
                'threshold': self.threshold,
                'prediction': 'genuine' if is_genuine else 'forged'
            }

    def predict_batch(self, image_pairs):
        """
        Predict for a batch of image pairs
        
        Args:
            image_pairs (list): List of tuples (image1_path, image2_path)
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for img1_path, img2_path in image_pairs:
            result = self.predict_pair(img1_path, img2_path)
            result['pair'] = (img1_path, img2_path)
            results.append(result)
        return results

    def set_threshold(self, new_threshold):
        """Update the threshold for predictions"""
        self.threshold = new_threshold
        print(f"Threshold updated to: {self.threshold}")

@torch.no_grad()
def evaluate_with_threshold(model, dataloader, threshold, device='cuda'):
    """
    Evaluate model performance on a dataset using a specific threshold
    
    Args:
        model: Trained SigNet model
        dataloader: DataLoader for evaluation
        threshold (float): Distance threshold for classification
        device (str): Device to run evaluation on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    genuine_correct = 0
    genuine_total = 0
    forged_correct = 0
    forged_total = 0
    
    all_distances = []
    all_labels = []
    
    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        # Get embeddings
        emb1, emb2 = model(x1, x2)
        
        # Calculate distances
        distances = F.pairwise_distance(emb1, emb2, p=2)
        
        # Make predictions (1 = genuine, 0 = forged)
        predictions = (distances < threshold).long()
        
        # Calculate metrics
        correct = (predictions == y).sum().item()
        correct_predictions += correct
        total_samples += len(y)
        
        # Separate metrics for genuine and forged
        genuine_mask = (y == 1)
        forged_mask = (y == 0)
        
        if genuine_mask.sum() > 0:
            genuine_correct += (predictions[genuine_mask] == y[genuine_mask]).sum().item()
            genuine_total += genuine_mask.sum().item()
            
        if forged_mask.sum() > 0:
            forged_correct += (predictions[forged_mask] == y[forged_mask]).sum().item()
            forged_total += forged_mask.sum().item()
        
        # Store for detailed analysis
        all_distances.extend(distances.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    
    # Calculate metrics
    overall_accuracy = correct_predictions / total_samples
    genuine_accuracy = genuine_correct / genuine_total if genuine_total > 0 else 0
    forged_accuracy = forged_correct / forged_total if forged_total > 0 else 0
    
    # Calculate False Accept Rate (FAR) and False Reject Rate (FRR)
    false_accepts = forged_total - forged_correct  # Forged classified as genuine
    false_rejects = genuine_total - genuine_correct  # Genuine classified as forged
    
    far = false_accepts / forged_total if forged_total > 0 else 0
    frr = false_rejects / genuine_total if genuine_total > 0 else 0
    
    return {
        'threshold': threshold,
        'overall_accuracy': overall_accuracy,
        'genuine_accuracy': genuine_accuracy,
        'forged_accuracy': forged_accuracy,
        'false_accept_rate': far,
        'false_reject_rate': frr,
        'total_samples': total_samples,
        'genuine_samples': genuine_total,
        'forged_samples': forged_total,
        'distances': all_distances,
        'labels': all_labels
    }

def find_optimal_threshold(model, dataloader, threshold_range=(0.1, 2.0), num_thresholds=100, device='cuda'):
    """
    Find optimal threshold by testing multiple values
    
    Args:
        model: Trained SigNet model
        dataloader: DataLoader for evaluation
        threshold_range (tuple): Range of thresholds to test (min, max)
        num_thresholds (int): Number of thresholds to test
        device (str): Device to run evaluation on
        
    Returns:
        dict: Results for all tested thresholds
    """
    print(f"Testing {num_thresholds} thresholds from {threshold_range[0]} to {threshold_range[1]}")
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    results = []
    
    # Get all distances and labels first
    model.eval()
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            emb1, emb2 = model(x1, x2)
            distances = F.pairwise_distance(emb1, emb2, p=2)
            all_distances.extend(distances.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    
    # Test each threshold
    for threshold in thresholds:
        predictions = (all_distances < threshold).astype(int)
        accuracy = (predictions == all_labels).mean()
        
        # Calculate FAR and FRR
        genuine_mask = all_labels == 1
        forged_mask = all_labels == 0
        
        if genuine_mask.sum() > 0:
            frr = (predictions[genuine_mask] != all_labels[genuine_mask]).mean()
        else:
            frr = 0
            
        if forged_mask.sum() > 0:
            far = (predictions[forged_mask] != all_labels[forged_mask]).mean()
        else:
            far = 0
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'far': far,
            'frr': frr
        })
    
    # Find best threshold (highest accuracy)
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print(f"Best threshold: {best_result['threshold']:.4f}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    print(f"FAR: {best_result['far']:.4f}, FRR: {best_result['frr']:.4f}")
    
    return {
        'all_results': results,
        'best_threshold': best_result['threshold'],
        'best_accuracy': best_result['accuracy'],
        'best_result': best_result
    }

if __name__ == "__main__":
    parser = ArgumentParser(description='SigNet Inference Script')
    parser.add_argument('--model_path', type=str, default='experiments/signet_exp_20250703_101313/checkpoints/20250703_101313_epoch_000_loss_0.4860_acc_1.0000.pt', help='Path to trained model checkpoint')
    parser.add_argument('--image1', type=str, default='data/CEDAR/full_org/original_38_1.png', help='Path to first signature image')
    parser.add_argument('--image2', type=str, default='data/CEDAR/full_org/original_38_3.png', help='Path to second signature image')
    parser.add_argument('--threshold', type=float, default=0.0349, help='Distance threshold for classification')
    parser.add_argument('--dataset', type=str, help='Dataset directory for evaluation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--find_optimal_threshold', action='store_true', help='Find optimal threshold on dataset')
    parser.add_argument('--threshold_min', type=float, default=0.1, help='Minimum threshold to test')
    parser.add_argument('--threshold_max', type=float, default=2.0, help='Maximum threshold to test')
    parser.add_argument('--num_thresholds', type=int, default=100, help='Number of thresholds to test')
    
    args = parser.parse_args()
    
    # Load model for inference
    inferencer = SigNetInference(args.model_path, args.threshold)
    
    if args.image1 and args.image2:
        # Single pair prediction
        print(f"\nPredicting signature pair:")
        print(f"Image 1: {args.image1}")
        print(f"Image 2: {args.image2}")
        
        result = inferencer.predict_pair(args.image1, args.image2)
        
        print(f"\nResults:")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Prediction: {result['prediction']}")
        print(f"Threshold: {result['threshold']:.4f}")
        
    if args.dataset:
        # Dataset evaluation
        print(f"\nEvaluating on dataset: {args.dataset}")
        
        # Get image dimensions from model
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'args' in checkpoint:
            model_args = checkpoint['args']
            img_height = model_args.img_height if hasattr(model_args, 'img_height') else 155
            img_width = model_args.img_width if hasattr(model_args, 'img_width') else 220
        else:
            img_height, img_width = 155, 220
        
        # Create data transform
        image_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            ImageOps.invert,
            transforms.ToTensor(),
        ])
        
        # Load test data
        testloader = get_data_loader(is_train=False, batch_size=args.batch_size, 
                                   image_transform=image_transform, dataset_dir=args.dataset)
        
        if args.find_optimal_threshold:
            # Find optimal threshold
            print("Finding optimal threshold...")
            optimal_results = find_optimal_threshold(
                inferencer.model, testloader, 
                threshold_range=(args.threshold_min, args.threshold_max),
                num_thresholds=args.num_thresholds,
                device=device
            )
            
            # Update inferencer with optimal threshold
            inferencer.set_threshold(optimal_results['best_threshold'])
        
        # Evaluate with current threshold
        print(f"Evaluating with threshold: {inferencer.threshold}")
        eval_results = evaluate_with_threshold(inferencer.model, testloader, inferencer.threshold, device)
        
        print(f"\nEvaluation Results:")
        print(f"Overall Accuracy: {eval_results['overall_accuracy']:.4f}")
        print(f"Genuine Accuracy: {eval_results['genuine_accuracy']:.4f}")
        print(f"Forged Accuracy: {eval_results['forged_accuracy']:.4f}")
        print(f"False Accept Rate (FAR): {eval_results['false_accept_rate']:.4f}")
        print(f"False Reject Rate (FRR): {eval_results['false_reject_rate']:.4f}")
        print(f"Total Samples: {eval_results['total_samples']}")
        print(f"Genuine Samples: {eval_results['genuine_samples']}")
        print(f"Forged Samples: {eval_results['forged_samples']}") 