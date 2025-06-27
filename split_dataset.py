import os
import glob
import random
import shutil
from pathlib import Path
from typing import List, Tuple
import argparse
from tqdm import tqdm

def get_all_images(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Get all genuine and forged signature image paths from the CEDAR dataset.
    
    Args:
        data_dir: Path to the CEDAR dataset directory
        
    Returns:
        Tuple of (genuine_images, forged_images) lists
    """
    genuine_dir = os.path.join(data_dir, 'signatures', 'full_org')
    forged_dir = os.path.join(data_dir, 'signatures', 'full_forg')
    
    # Get all genuine signature images
    genuine_pattern = os.path.join(genuine_dir, 'original_*.png')
    genuine_images = glob.glob(genuine_pattern)
    
    # Get all forged signature images  
    forged_pattern = os.path.join(forged_dir, 'forgeries_*.png')
    forged_images = glob.glob(forged_pattern)
    
    return genuine_images, forged_images

def create_split_directories(output_dir: str):
    """
    Create train and validation directories with subdirectories for genuine and forged signatures.
    
    Args:
        output_dir: Base directory for the split dataset
    """
    splits = ['train', 'val']
    categories = ['genuine', 'forged']
    
    for split in splits:
        for category in categories:
            dir_path = os.path.join(output_dir, split, category)
            os.makedirs(dir_path, exist_ok=True)

def copy_images(image_paths: List[str], dest_dir: str, desc: str = "Copying images"):
    """
    Copy images to destination directory with progress bar.
    
    Args:
        image_paths: List of source image paths
        dest_dir: Destination directory
        desc: Description for the progress bar
    """
    for img_path in tqdm(image_paths, desc=desc, unit="files"):
        filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(img_path, dest_path)

def split_cedar_dataset(
    data_dir: str = 'data/CEDAR', 
    output_dir: str = 'data/CEDAR_split',
    val_ratio: float = 0.2, 
    random_seed: int = 42
):
    """
    Split the CEDAR dataset randomly into train and validation sets.
    
    Args:
        data_dir: Path to the original CEDAR dataset
        output_dir: Path where the split dataset will be saved
        val_ratio: Ratio of validation set (0.0 to 1.0)
        random_seed: Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    print(f"Loading images from {data_dir}...")
    genuine_images, forged_images = get_all_images(data_dir)
    
    print(f"Found {len(genuine_images)} genuine signatures")
    print(f"Found {len(forged_images)} forged signatures")
    
    # Shuffle the lists randomly
    random.shuffle(genuine_images)
    random.shuffle(forged_images)
    
    # Calculate split indices
    genuine_val_size = int(len(genuine_images) * val_ratio)
    forged_val_size = int(len(forged_images) * val_ratio)
    
    # Split genuine signatures
    genuine_val = genuine_images[:genuine_val_size]
    genuine_train = genuine_images[genuine_val_size:]
    
    # Split forged signatures
    forged_val = forged_images[:forged_val_size]
    forged_train = forged_images[forged_val_size:]
    
    print(f"\nSplit summary:")
    print(f"Training set - Genuine: {len(genuine_train)}, Forged: {len(forged_train)}")
    print(f"Validation set - Genuine: {len(genuine_val)}, Forged: {len(forged_val)}")
    
    # Create output directories
    print(f"\nCreating directories in {output_dir}...")
    create_split_directories(output_dir)
    
    # Copy images to their respective directories
    print("\nCopying images with progress tracking...")
    copy_images(genuine_train, os.path.join(output_dir, 'train', 'genuine'), 
                desc="Copying training genuine signatures")
    copy_images(forged_train, os.path.join(output_dir, 'train', 'forged'),
                desc="Copying training forged signatures")
    copy_images(genuine_val, os.path.join(output_dir, 'val', 'genuine'),
                desc="Copying validation genuine signatures")
    copy_images(forged_val, os.path.join(output_dir, 'val', 'forged'),
                desc="Copying validation forged signatures")
    
    print("\nDataset split completed successfully!")
    print(f"Split dataset saved to: {output_dir}")
    
    # Print final statistics
    print(f"\nFinal statistics:")
    print(f"├── train/")
    print(f"│   ├── genuine/ ({len(genuine_train)} images)")
    print(f"│   └── forged/ ({len(forged_train)} images)")
    print(f"└── val/")
    print(f"    ├── genuine/ ({len(genuine_val)} images)")
    print(f"    └── forged/ ({len(forged_val)} images)")

def main():
    parser = argparse.ArgumentParser(description='Split CEDAR dataset into train and validation sets')
    parser.add_argument('--data_dir', type=str, default='data/CEDAR',
                        help='Path to the original CEDAR dataset (default: data/CEDAR)')
    parser.add_argument('--output_dir', type=str, default='data/CEDAR_split',
                        help='Path where the split dataset will be saved (default: data/CEDAR_split)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        return
    
    if not (0.0 < args.val_ratio < 1.0):
        print(f"Error: Validation ratio must be between 0.0 and 1.0, got {args.val_ratio}")
        return
    
    # Run the splitting
    split_cedar_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir, 
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main() 