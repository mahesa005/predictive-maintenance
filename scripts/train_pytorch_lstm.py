"""
PyTorch LSTM Training Script for Predictive Maintenance
Comparison script to the custom CuPy implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class LSTMModel(nn.Module):
    """PyTorch LSTM for binary classification."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()


def create_sequences(X, y, seq_length=5):
    """Create sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)


def load_and_prepare_data(data_path, seq_length=5, test_size=0.2, val_size=0.1):
    """Load and prepare data for training."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    print(f"Class 1 ratio: {y.mean():.4f}")
    
    # Create sequences
    X_seq, y_seq = create_sequences(X, y, seq_length)
    print(f"Sequences created: {len(X_seq)} (seq_length={seq_length})")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, random_state=42, stratify=y_seq
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs=50, patience=7, device='cpu'):
    """Train the model with early stopping."""
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs (patience={patience})")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = (outputs >= 0.35).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            all_probs.extend(outputs.cpu().numpy())
            preds = (outputs >= 0.35).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Failure', 'Failure']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Additional metrics
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"\nROC-AUC Score: {auc:.4f}")
    except:
        print("\nROC-AUC: Could not compute")
    
    print(f"\nPrediction Statistics:")
    print(f"  Min prob: {all_probs.min():.4f}")
    print(f"  Max prob: {all_probs.max():.4f}")
    print(f"  Mean prob: {all_probs.mean():.4f}")
    print(f"  Predicted 1s: {all_preds.sum()} / {len(all_preds)}")
    
    return all_preds, all_probs, all_labels


def plot_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['val_acc'], label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1 Score
    axes[2].plot(history['val_f1'], label='Val F1', color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def main():
    # ========== HYPERPARAMETERS ==========
    DATA_PATH = 'data/processed_dataset.csv'
    SEQ_LENGTH = 48
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    EPOCHS = 100
    PATIENCE = 10
    POS_WEIGHT = 2.2  # Weight for positive class (failure)
    
    print("="*60)
    print("PyTorch LSTM - Predictive Maintenance")
    print("="*60)
    print(f"\nHyperparameters:")
    print(f"  SEQ_LENGTH:    {SEQ_LENGTH}")
    print(f"  HIDDEN_SIZE:   {HIDDEN_SIZE}")
    print(f"  NUM_LAYERS:    {NUM_LAYERS}")
    print(f"  DROPOUT:       {DROPOUT}")
    print(f"  BATCH_SIZE:    {BATCH_SIZE}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  POS_WEIGHT:    {POS_WEIGHT}")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        DATA_PATH, seq_length=SEQ_LENGTH
    )
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    input_size = X_train.shape[2]
    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss with class weight
    pos_weight = torch.tensor([POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use BCELoss since we have sigmoid in forward
    criterion = nn.BCELoss(reduction='none')
    
    def weighted_bce_loss(pred, target):
        loss = criterion(pred, target)
        weight = torch.where(target == 1, torch.tensor(POS_WEIGHT).to(device), torch.tensor(1.0).to(device))
        return (loss * weight).mean()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    
    # Train
    history = train_model(
        model, train_loader, val_loader,
        weighted_bce_loss, optimizer,
        epochs=EPOCHS, patience=PATIENCE, device=device
    )
    
    # Evaluate
    all_preds, all_probs, all_labels = evaluate_model(model, test_loader, device)
    
    # Plot
    plot_history(history, save_path='data/pytorch_training_history.png')
    
    # Save predictions
    results_df = pd.DataFrame({
        'actual': all_labels,
        'predicted': all_preds,
        'probability': all_probs
    })
    results_df.to_csv('data/predictions_pytorch.csv', index=False)
    print("\nPredictions saved to: data/predictions_pytorch.csv")
    
    # Save model
    torch.save(model.state_dict(), 'data/pytorch_lstm_model.pth')
    print("Model saved to: data/pytorch_lstm_model.pth")


if __name__ == '__main__':
    main()
