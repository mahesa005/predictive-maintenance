"""
Training script with Validation Loss Tracking & Early Stopping.
Detects overfitting automatically.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model.lstm_cupy_optimized import LSTMModelGPUOptimized, GPU_AVAILABLE

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')
OUTPUT_FILE = os.path.join(DATA_DIR, 'predictions_with_validation.csv')

# Hyperparameters
HIDDEN_SIZE = 128
EPOCHS = 250
LEARNING_RATE = 0.001
BATCH_SIZE = 128
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1  # 10% of training data for validation
EARLY_STOPPING_PATIENCE = 15  # Stop if val loss doesn't improve for N epochs


def evaluate_loss(model, X, y, batch_size=128):
    """Calculate loss on a dataset without training."""
    try:
        import cupy as cp
    except ImportError:
        import numpy as cp
    
    X_gpu = cp.asarray(X, dtype=cp.float32)
    y_gpu = cp.asarray(y, dtype=cp.float32)
    
    n_samples = len(X_gpu)
    n_batches = (n_samples + batch_size - 1) // batch_size
    total_loss = 0
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)
        
        X_batch = X_gpu[start:end]
        y_batch = y_gpu[start:end]
        
        y_pred, _, _, _ = model.forward_batch(X_batch)
        
        loss = -cp.mean(y_batch * cp.log(y_pred + 1e-9) + 
                       (1 - y_batch) * cp.log(1 - y_pred + 1e-9))
        total_loss += float(loss)
    
    return total_loss / n_batches


def train_with_validation(model, X_train, y_train, X_val, y_val, 
                          epochs, batch_size, lr, patience):
    """
    Training loop with validation loss tracking and early stopping.
    """
    try:
        import cupy as cp
    except ImportError:
        import numpy as cp
    
    X_gpu = cp.asarray(X_train, dtype=cp.float32)
    y_gpu = cp.asarray(y_train, dtype=cp.float32)
    
    n_samples = len(X_gpu)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"\nTraining: {n_samples} samples, {n_batches} batches/epoch")
    print(f"Validation: {len(X_val)} samples")
    print(f"Early stopping patience: {patience} epochs\n")
    print("-" * 70)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Val Loss':^12} | {'Status':^20}")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = cp.random.permutation(n_samples)
        X_shuffled = X_gpu[indices]
        y_shuffled = y_gpu[indices]
        
        epoch_loss = 0
        
        # Training loop
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward
            y_pred, caches, _, _ = model.forward_batch(X_batch)
            
            # Loss
            loss = -cp.mean(y_batch * cp.log(y_pred + 1e-9) + 
                           (1 - y_batch) * cp.log(1 - y_pred + 1e-9))
            epoch_loss += float(loss)
            
            # Backward
            grads = model.backward_batch(y_pred, y_batch, caches)
            
            # Update
            model.update_adam(grads, lr=lr)
        
        # Calculate losses
        train_loss = epoch_loss / n_batches
        val_loss = evaluate_loss(model, X_val, y_val, batch_size)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            status = "✓ Best"
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status = "⚠ EARLY STOP"
            elif val_loss > train_loss * 1.1:
                status = "⚠ Overfitting?"
            else:
                status = ""
        
        # Print progress
        print(f"{epoch+1:^8} | {train_loss:^12.4f} | {val_loss:^12.4f} | {status:^20}")
        
        # Early stopping
        if patience_counter >= patience:
            print("-" * 70)
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            print(f"   Best epoch: {best_epoch} with val_loss: {best_val_loss:.4f}")
            break
        
        if GPU_AVAILABLE:
            cp.cuda.Stream.null.synchronize()
    
    print("-" * 70)
    return history, best_epoch, best_val_loss


def main():
    print("=" * 70)
    print("LSTM TRAINING WITH VALIDATION & EARLY STOPPING")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading preprocessed data...")
    X = np.load(os.path.join(PREPROCESSED_DIR, 'X_sequences.npy'))
    y = np.load(os.path.join(PREPROCESSED_DIR, 'y_sequences.npy'))
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split: Train / Validation / Test
    print("\n[2/5] Splitting data...")
    test_idx = int(len(X) * (1 - TEST_SPLIT))
    X_trainval, X_test = X[:test_idx], X[test_idx:]
    y_trainval, y_test = y[:test_idx], y[test_idx:]
    
    val_idx = int(len(X_trainval) * (1 - VAL_SPLIT))
    X_train, X_val = X_trainval[:val_idx], X_trainval[val_idx:]
    y_train, y_val = y_trainval[:val_idx], y_trainval[val_idx:]
    
    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    # Create model
    print("\n[3/5] Creating model...")
    input_size = X.shape[2]
    model = LSTMModelGPUOptimized(input_size=input_size, hidden_size=HIDDEN_SIZE)
    
    print(f"Model: LSTM (input={input_size}, hidden={HIDDEN_SIZE})")
    
    # Train with validation
    print("\n[4/5] Training...")
    history, best_epoch, best_val_loss = train_with_validation(
        model, X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        patience=EARLY_STOPPING_PATIENCE
    )
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE)
    predictions = (probabilities >= 0.5).astype(int)
    
    accuracy = np.mean(predictions == y_test)
    
    # Calculate precision, recall, F1
    tp = np.sum((predictions == 1) & (y_test == 1))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))
    tn = np.sum((predictions == 0) & (y_test == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print(f"\n📈 Training Summary:")
    print(f"   Best Epoch:     {best_epoch}")
    print(f"   Best Val Loss:  {best_val_loss:.4f}")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Val Loss:   {history['val_loss'][-1]:.4f}")
    
    # Overfitting analysis
    gap = history['val_loss'][-1] - history['train_loss'][-1]
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   Train-Val Gap: {gap:.4f}")
    if gap > 0.1:
        print("   Status: ⚠️  OVERFITTING detected (gap > 0.1)")
    elif gap > 0.05:
        print("   Status: ⚡ Slight overfitting (gap > 0.05)")
    else:
        print("   Status: ✅ Good fit (gap < 0.05)")
    
    # Save results
    pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'probability': probabilities
    }).to_csv(OUTPUT_FILE, index=False)
    
    # Save history
    history_file = os.path.join(DATA_DIR, 'training_history.csv')
    pd.DataFrame(history).to_csv(history_file, index=False)
    
    print(f"\n✅ Results saved to: {OUTPUT_FILE}")
    print(f"✅ History saved to: {history_file}")


if __name__ == "__main__":
    main()
