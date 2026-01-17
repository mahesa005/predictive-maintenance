"""
Preprocessing script for LSTM training.
Prepares labeled_dataset.csv for sequence-based learning.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'labeled_dataset.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'preprocessed')

# Hyperparameters
SEQUENCE_LENGTH = 5  # Number of rows to look back

def main():
    print("=" * 60)
    print("PREPROCESSING FOR LSTM")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 2. Select features (drop non-informative columns)
    print("\n[2/5] Selecting features...")
    # Drop: NO (ID), Timestamp, Resolved Time (datetime strings)
    feature_cols = ['Service Type', 'Service Name', 'Type', 'Status', 
                    'SLA (minutes)', 'Month']
    label_col = 'label'
    
    df_features = df[feature_cols].copy()
    y = df[label_col].values
    
    # 3. Handle missing values and encode
    print("\n[3/5] Encoding categorical features...")
    encoders = {}
    categorical_cols = ['Service Type', 'Service Name', 'Type', 'Status', 'Month']
    
    for col in categorical_cols:
        df_features[col] = df_features[col].fillna('Unknown')
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col].astype(str))
        encoders[col] = le
    
    # Fill numerical NaN with median
    df_features['SLA (minutes)'] = df_features['SLA (minutes)'].fillna(
        df_features['SLA (minutes)'].median()
    )
    
    # 4. Scale features
    print("\n[4/5] Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_features.values)
    
    print(f"Feature shape: {X_scaled.shape}")
    
    # 5. Create sequences
    print("\n[5/5] Creating sequences...")
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_scaled) - SEQUENCE_LENGTH):
        X_sequences.append(X_scaled[i:i + SEQUENCE_LENGTH])
        y_sequences.append(y[i + SEQUENCE_LENGTH])  # Predict next label
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"Sequences shape: {X_sequences.shape}")
    print(f"Labels shape: {y_sequences.shape}")
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'X_sequences.npy'), X_sequences)
    np.save(os.path.join(OUTPUT_DIR, 'y_sequences.npy'), y_sequences)
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
