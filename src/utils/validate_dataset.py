"""
Script untuk validasi dataset - mengecek kolom apa saja yang ada dan jumlahnya.
Validation script to check what columns exist and count them.
"""

import pandas as pd
import os

# Define file path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATASET_FILE = os.path.join(DATA_DIR, 'labeled_dataset.csv')

def validate_dataset():
    print("=" * 70)
    print("VALIDASI DATASET / DATASET VALIDATION")
    print("=" * 70)
    print(f"\nFile: {DATASET_FILE}\n")
    
    # Read the dataset
    df = pd.read_csv(DATASET_FILE)
    
    # 1. Jumlah kolom / Column count
    num_columns = len(df.columns)
    print(f"📊 JUMLAH KOLOM / NUMBER OF COLUMNS: {num_columns}")
    print()
    
    # 2. Daftar kolom / List of columns
    print("📋 DAFTAR KOLOM / COLUMN LIST:")
    print("-" * 70)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    print()
    
    # 3. Info dataset
    print("ℹ️  INFORMASI DATASET / DATASET INFO:")
    print("-" * 70)
    print(f"Jumlah baris / Number of rows    : {len(df):,}")
    print(f"Jumlah kolom / Number of columns : {num_columns}")
    print()
    
    # 4. Tipe data tiap kolom / Data types
    print("🔢 TIPE DATA / DATA TYPES:")
    print("-" * 70)
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f"{col:30s} | {str(dtype):10s} | Non-null: {non_null:6,} | Null: {null_count:6,}")
    print()
    
    # 5. Statistik dasar untuk kolom numerik / Basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("📈 KOLOM NUMERIK / NUMERIC COLUMNS:")
        print("-" * 70)
        for col in numeric_cols:
            unique_vals = df[col].nunique()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col:30s} | Unique: {unique_vals:6,} | Min: {min_val:10} | Max: {max_val:10}")
        print()
    
    # 6. Untuk kolom label (jika ada) / For label column (if exists)
    if 'label' in df.columns:
        print("🏷️  DISTRIBUSI LABEL / LABEL DISTRIBUTION:")
        print("-" * 70)
        label_counts = df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"Label {label}: {count:6,} ({percentage:5.2f}%)")
        print()
    
    print("=" * 70)
    print("✅ VALIDASI SELESAI / VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    validate_dataset()
