"""
Script to process the dataset by creating a label column based on Priority and Impact,
then removing those columns from the dataset.

Label: 1 if (Priority >= 2 AND Impact >= 2), else 0
"""

import pandas as pd
import os

# Define file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'Datasetxlsx.xlsx - Data Paper.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'labeled_dataset.csv')

def main():
    print("Loading dataset...")
    # Read the CSV file
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create the label column
    print("\nCreating label column...")
    df['label'] = ((df['Priority'] >= 2) & (df['Impact'] >= 2)).astype(int)
    
    # Check label distribution
    label_counts = df['label'].value_counts()
    print(f"\nLabel distribution:")
    print(f"  Label 0: {label_counts.get(0, 0)} ({label_counts.get(0, 0) / len(df) * 100:.2f}%)")
    print(f"  Label 1: {label_counts.get(1, 0)} ({label_counts.get(1, 0) / len(df) * 100:.2f}%)")
    
    # Remove Priority and Impact columns
    print("\nRemoving Priority and Impact columns...")
    df = df.drop(columns=['Priority', 'Impact'])
    
    print(f"Processed dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Save to new CSV
    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("✓ Processing complete!")
    print(f"\nFirst 5 rows of processed dataset:")
    print(df.head())

if __name__ == "__main__":
    main()
