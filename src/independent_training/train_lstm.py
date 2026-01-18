"""
Training script for LSTM model.
Loads preprocessed data, trains model, outputs predictions.csv
"""

import numpy as np
import pandas as pd
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from model.lstm import LSTMModel

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')
OUTPUT_FILE = os.path.join(DATA_DIR, 'predictions.csv')

# Hyperparameters
HIDDEN_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01
TEST_SPLIT = 0.2

def main():
    print("=" * 60)
    print("LSTM TRAINING")
    print("=" * 60)
    
    # 1. Load preprocessed data
    print("\n[1/4] Loading preprocessed data...")
    X = np.load(os.path.join(PREPROCESSED_DIR, 'X_sequences.npy'))
    y = np.load(os.path.join(PREPROCESSED_DIR, 'y_sequences.npy'))
    
    print(f"X shape: {X.shape}")  # (samples, sequence_length, features)
    print(f"y shape: {y.shape}")
    
    # 2. Train/Test split
    print("\n[2/4] Splitting data...")
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # 3. Initialize and train model
    print("\n[3/4] Training LSTM model...")
    input_size = X.shape[2]  # Number of features
    model = LSTMModel(input_size=input_size, hidden_size=HIDDEN_SIZE)
    
    model.train(X_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # 4. Generate predictions on test set
    print("\n[4/4] Generating predictions...")
    predictions = []
    probabilities = []
    
    for i in range(len(X_test)):
        h = np.zeros((HIDDEN_SIZE, 1))
        c = np.zeros((HIDDEN_SIZE, 1))
        
        for t in range(len(X_test[i])):
            x_t = X_test[i][t].reshape(-1, 1)
            h, c, y_pred, _ = model.forward_step(x_t, h, c)
        
        prob = float(np.squeeze(y_pred))
        pred = 1 if prob >= 0.5 else 0
        probabilities.append(prob)
        predictions.append(pred)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'probability': probabilities
    })
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print(f"Predictions saved to: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
