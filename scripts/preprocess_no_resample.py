"""
Preprocessing Script - Without Time Resampling
Produces dataset with only Timestamp, Impact, Priority, and Incident
"""

import numpy as np
import pandas as pd
import os
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join('..',PROJECT_DIR, 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'raw_dataset.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'preprocessed_no_resample.csv')

def load_data():
    """Load preprocessed dataset"""
    print("=" * 60)
    print("📂 Loading preprocessed data...")
    print("=" * 60)
    
    df = pd.read_csv(INPUT_FILE)
    print(f"   Original Shape: {df.shape}")
    
    return df

def visualize_distribution(df):
    """Print incident distribution based on the new logic"""
    print("\n" + "=" * 60)
    print("📊 Incident Distribution")
    print("=" * 60)
    
    counts = df['incident'].value_counts()
    print(f"   Normal (0): {counts.get(0, 0):,} ({counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   Incident (1): {counts.get(1, 0):,} ({counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"   Ratio: 1 : {counts.get(0, 0)/max(counts.get(1, 0), 1):.1f}")

def prepare_output(df):
    """Filter by restoration and create incident column"""
    print("\n" + "=" * 60)
    print("🔧 Preparing Output (Filtering & Feature Engineering)...")
    print("=" * 60)
    
    # 1. Konversi Timestamp ke datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 2. Hanya keep column yang typenya = restoration
    df = df[df['Type'].str.contains('restoration', case=False, na=False)]
    print(f"   Shape after filtering 'restoration': {df.shape}")
    
    # 3. Konversi Priority dan Impact ke numeric (untuk memastikan operasi >= 2 berjalan)
    df['Priority'] = pd.to_numeric(df['Priority'], errors='coerce')
    df['Impact'] = pd.to_numeric(df['Impact'], errors='coerce')
    
    # 4. Buat kolom incident: 1 apabila impact >= 2 && priority >= 2
    df['incident'] = ((df['Impact'] >= 2) & (df['Priority'] >= 2)).astype(int)
    
    # 5. Hanya keep kolom yang diminta: Timestamp, Impact, Priority, incident
    columns_to_keep = ['Timestamp', 'Impact', 'Priority', 'incident']
    df = df[columns_to_keep]
    
    print(f"   Final columns: {df.columns.tolist()}")
    print(f"   Output shape: {df.shape}")
    print("   ✅ Applied restoration filter and created 'incident' label")
    
    return df

def main():
    # 1. Load data
    df = load_data()
    
    # 2. Prepare output (filter & logic)
    df = prepare_output(df)
    
    # 3. Show distribution
    visualize_distribution(df)
    
    # 4. Save output
    print("\n" + "=" * 60)
    print("💾 Saving Dataset (No Resample)...")
    print("=" * 60)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"   ✅ Saved to: {OUTPUT_FILE}")
    print(f"   Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    main()