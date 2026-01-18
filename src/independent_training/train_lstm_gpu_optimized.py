"""
Training script for Optimized LSTM model using CuPy (GPU).
Uses batched operations for ~10-50x speedup.
Includes train/validation split and visualization for overfitting analysis.
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from model.lstm_cupy_optimized import LSTMModelGPUOptimized, GPU_AVAILABLE

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')
OUTPUT_FILE = os.path.join(DATA_DIR, 'predictions_gpu_optimized.csv')
PLOT_FILE = os.path.join(DATA_DIR, 'training_history.png')

# Hyperparameters
HIDDEN_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001  # Lower LR for Adam
BATCH_SIZE = 128       # Mini-batch size
VAL_SPLIT = 0.2        # 80/20 train/validation split
RANDOM_STATE = 42      # For reproducibility


def plot_training_history(history, save_path):
    """
    Plot training history with two subplots:
    1. Training Loss vs Validation Loss
    2. Validation Accuracy over epochs
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss comparison
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (BCE)', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for overfitting detection
    if history['val_loss']:
        final_train = history['train_loss'][-1]
        final_val = history['val_loss'][-1]
        gap = final_val - final_train
        
        if gap > 0.1:
            ax1.annotate('⚠️ Overfitting Detected', 
                        xy=(len(epochs), final_val), 
                        xytext=(len(epochs)*0.7, max(history['val_loss'])*0.9),
                        fontsize=10, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 2: Accuracy over epochs
    ax2 = axes[1]
    if history['val_acc']:
        ax2.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
    if history['train_acc']:
        ax2.plot(epochs, history['train_acc'], 'b--', label='Training Accuracy', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy over Epochs', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add horizontal line at 0.5 (random baseline)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random Baseline')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Training history plot saved to: {save_path}")


def main():
    print("=" * 70)
    print("OPTIMIZED LSTM TRAINING WITH VALIDATION" + (" (GPU)" if GPU_AVAILABLE else " (CPU)"))
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading preprocessed data...")
    X = np.load(os.path.join(PREPROCESSED_DIR, 'X_sequences.npy'))
    y = np.load(os.path.join(PREPROCESSED_DIR, 'y_sequences.npy'))
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution: 0={np.sum(y==0)} | 1={np.sum(y==1)}")
    
    # Train/Validation split using sklearn
    print(f"\n[2/5] Splitting data (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=VAL_SPLIT, 
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class balance
    )
    
    print(f"Training:   {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Train class distribution: 0={np.sum(y_train==0)} | 1={np.sum(y_train==1)}")
    print(f"Val class distribution:   0={np.sum(y_val==0)} | 1={np.sum(y_val==1)}")
    
    # Initialize model
    print("\n[3/5] Initializing model...")
    input_size = X.shape[2]
    seq_len = X.shape[1]
    print(f"Input size: {input_size}, Sequence length: {seq_len}, Hidden size: {HIDDEN_SIZE}")
    
    model = LSTMModelGPUOptimized(input_size=input_size, hidden_size=HIDDEN_SIZE)
    
    # Train with validation
    print("\n[4/5] Training (Batched with Validation)...")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    print()
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        print_every=10
    )
    
    # Plot training history
    plot_training_history(history, PLOT_FILE)
    
    # Final evaluation
    print("\n[5/5] Final Evaluation...")
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Generate predictions on validation set
    probabilities = model.predict(X_val, batch_size=BATCH_SIZE)
    predictions = (probabilities >= 0.5).astype(int)
    
    # Detailed metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_val, predictions, target_names=['Class 0', 'Class 1']))
    
    cm = confusion_matrix(y_val, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # Diagnose underfitting vs overfitting
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    gap = final_val_loss - final_train_loss
    
    print(f"Final Training Loss:   {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Gap (Val - Train):     {gap:.4f}")
    
    if val_acc < 0.6 and gap < 0.1:
        print("\n⚠️ DIAGNOSIS: UNDERFITTING")
        print("   - Both train and validation performance are poor")
        print("   - Model is too simple to capture patterns")
        print("   - Suggestions:")
        print("     * Increase hidden_size (currently: {})".format(HIDDEN_SIZE))
        print("     * Train for more epochs")
        print("     * Check data quality and preprocessing")
        print("     * Add more features if available")
    elif gap > 0.1:
        print("\n⚠️ DIAGNOSIS: OVERFITTING")
        print("   - Training loss is much lower than validation loss")
        print("   - Model memorizes training data instead of generalizing")
        print("   - Suggestions:")
        print("     * Add dropout regularization")
        print("     * Reduce hidden_size")
        print("     * Use early stopping")
        print("     * Collect more training data")
    else:
        print("\n✅ Model appears to be well-fitted")
        print("   - If accuracy is still low, consider:")
        print("     * Feature engineering")
        print("     * Hyperparameter tuning")
        print("     * Trying different architectures")
    
    # Save predictions
    pd.DataFrame({
        'actual': y_val,
        'predicted': predictions,
        'probability': probabilities
    }).to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Predictions saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
