import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def create_subset(file_path, subset_size, output_dir, stratify=False):
    """
    Creates a subset of a dataset from a CSV file.

    Args:
        file_path (str): Path to the input CSV file.
        subset_size (float): Proportion of the dataset to include in the subset (0.0 to 1.0).
        output_dir (str): Directory to save the output subset CSV file.
        stratify (bool): Whether to perform stratified sampling based on the label column.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path, header=None)
    
    # Assuming the label is in the 3rd column (index 2)
    labels = df.iloc[:, 2] if stratify and df.shape[1] > 2 else None

    # Create subset
    subset_df, _ = train_test_split(df, train_size=subset_size, stratify=labels, random_state=42)

    # Save subset to new CSV file
    base_filename = os.path.basename(file_path)
    name, ext = os.path.splitext(base_filename)
    output_filename = f"{name}_subset_{subset_size}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    subset_df.to_csv(output_path, index=False, header=False)
    print(f"Subset of size {len(subset_df)} created at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset of a dataset CSV.")
    parser.add_argument('--file_path', type=str, default='data/CEDAR/test.csv', help="Path to the input CSV file (e.g., 'data/CEDAR/train.csv').")
    parser.add_argument('--subset_size', type=float, default=0.4, help="Proportion of the dataset to include in the subset (e.g., 0.1 for 10%).")
    parser.add_argument('--output_dir', type=str, default='data/CEDAR', help="Directory to save the output subset CSV file.")
    parser.add_argument('--stratify', action='store_true', help="Perform stratified sampling based on the label column (column 3).")
    
    args = parser.parse_args()

    create_subset(args.file_path, args.subset_size, args.output_dir, args.stratify) 