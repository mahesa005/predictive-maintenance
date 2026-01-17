"""
Training script for LSTM model using CuPy (GPU).
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model.lstm_cupy import LSTMModelGPU, GPU_AVAILABLE

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')
OUTPUT_FILE = os.path.join(DATA_DIR, 'predictions_gpu.csv')

# Hyperparameters
HIDDEN_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01
TEST_SPLIT = 0.2

def main():
    print("=" * 60)
    print("LSTM TRAINING (GPU)" if GPU_AVAILABLE else "LSTM TRAINING (CPU Fallback)")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading preprocessed data...")
    X = np.load(os.path.join(PREPROCESSED_DIR, 'X_sequences.npy'))
    y = np.load(os.path.join(PREPROCESSED_DIR, 'y_sequences.npy'))
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Train/Test split
    print("\n[2/4] Splitting data...")
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train
    print("\n[3/4] Training...")
    input_size = X.shape[2]
    model = LSTMModelGPU(input_size=input_size, hidden_size=HIDDEN_SIZE)
    model.train(X_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Predict
    print("\n[4/4] Generating predictions...")
    probabilities = model.predict(X_test)
    predictions = [1 if p >= 0.5 else 0 for p in probabilities]
    
    accuracy = np.mean(np.array(predictions) == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Save
    pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'probability': probabilities
    }).to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
