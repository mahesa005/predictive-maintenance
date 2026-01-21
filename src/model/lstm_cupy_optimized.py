"""
Optimized CuPy LSTM with Batched Operations.
~10-50x faster than naive implementation.
Includes validation tracking for overfitting/underfitting analysis.
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ CuPy detected - Using GPU acceleration")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("⚠️ CuPy not found - Falling back to NumPy (CPU)")

import numpy as np


class LSTMModelGPUOptimized:
    """
    Batched LSTM for GPU - processes multiple samples simultaneously.
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cw = cw
        
        z_dim = hidden_size + input_size
        scale = 0.1
        
        # Xavier-like initialization
        self.params = {
            'Wf': cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale,
            'Wi': cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale,
            'Wc': cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale,
            'Wo': cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale,
            'Wy': cp.random.randn(output_size, hidden_size).astype(cp.float32) * scale,
            'bf': cp.zeros((hidden_size, 1), dtype=cp.float32),
            'bi': cp.zeros((hidden_size, 1), dtype=cp.float32),
            'bc': cp.zeros((hidden_size, 1), dtype=cp.float32),
            'bo': cp.zeros((hidden_size, 1), dtype=cp.float32),
            'by': cp.zeros((output_size, 1), dtype=cp.float32),
        }
        
        # Adam optimizer states
        self.m = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-cp.clip(x, -500, 500)))

    def forward_batch(self, X_batch):
        """
        Batched forward pass.
        X_batch: (batch_size, seq_len, input_size)
        Returns: predictions (batch_size,), caches
        """
        batch_size, seq_len, _ = X_batch.shape
        
        # Initialize hidden states for batch: (hidden_size, batch_size)
        h = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        caches = []
        
        for t in range(seq_len):
            # x_t: (input_size, batch_size)
            x_t = X_batch[:, t, :].T
            
            # Concatenate h and x: (hidden_size + input_size, batch_size)
            z = cp.vstack((h, x_t))
            
            # Gates (all vectorized across batch)
            f = self.sigmoid(cp.dot(self.params['Wf'], z) + self.params['bf'])
            i = self.sigmoid(cp.dot(self.params['Wi'], z) + self.params['bi'])
            c_bar = cp.tanh(cp.dot(self.params['Wc'], z) + self.params['bc'])
            o = self.sigmoid(cp.dot(self.params['Wo'], z) + self.params['bo'])
            
            # Calculate new cell and hidden state value
            c_new = f * c + i * c_bar
            h_new = o * cp.tanh(c_new)
            
            # Save cache for backprop
            caches.append((z, f, i, c_bar, c_new, o, h_new, c, h))
            
            # Update hidden and cell state
            h, c = h_new, c_new
        
        # Output layer: (output_size, batch_size)
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], h) + self.params['by'])
        
        return y_pred.flatten(), caches, h, c

    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass using BPTT.
        """
        batch_size = len(y_true)
        seq_len = len(caches)
        
        # Initialize gradients
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # Output gradient: (1, batch_size)
        # Weighted Cross Entropy: Class 1 is 2.2x more important
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        
        # Adjust gradient for class 1 using class weight
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        # Get final hidden state from last cache
        h_final = caches[-1][6]  # h_new from last timestep
        
        # Calculate gradients for output layer weight/neuron layer + bias
        grads['Wy'] = cp.dot(dy, h_final.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size

        # IMPORTANT NOTE:
        # All gradient values are divided by batch_size to average the gradients across the batches
        # (Since each batch consists of different samples of the dataset, therefore having different y_true
        # directly producing different gradients from every other batches).
        
        # Backprop through hidden state
        dh_next = cp.dot(self.params['Wy'].T, dy)
        dc_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, c, o, h, c_prev, h_prev = caches[t]
            
            dh = dh_next
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo'] += cp.dot(da_o, z.T) / batch_size 
            grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size

            # Cell state
            dc = dh * o * (1 - cp.tanh(c)**2) + dc_next
            
            # Candidate
            dc_bar = dc * i
            da_c = dc_bar * (1 - c_bar**2)
            grads['Wc'] += cp.dot(da_c, z.T) / batch_size
            grads['bc'] += cp.sum(da_c, axis=1, keepdims=True) / batch_size
            
            # Input gate
            di = dc * c_bar
            da_i = di * i * (1 - i)
            grads['Wi'] += cp.dot(da_i, z.T) / batch_size
            grads['bi'] += cp.sum(da_i, axis=1, keepdims=True) / batch_size
            
            # Forget gate
            df = dc * c_prev
            da_f = df * f * (1 - f)
            grads['Wf'] += cp.dot(da_f, z.T) / batch_size
            grads['bf'] += cp.sum(da_f, axis=1, keepdims=True) / batch_size
            
            # Gradients for next timestep
            dz = (cp.dot(self.params['Wf'].T, da_f) +
                  cp.dot(self.params['Wi'].T, da_i) +
                  cp.dot(self.params['Wc'].T, da_c) +
                  cp.dot(self.params['Wo'].T, da_o))
            
            dh_next = dz[:self.hidden_size, :]
            dc_next = f * dc
        
        # Gradient clipping
        for k in grads:
            grads[k] = cp.clip(grads[k], -5, 5) # Prevent exploding gradients, forces the gradients to be in the range of [-5, 5]
        
        return grads

    def update_adam(self, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, lambda_l2=0.01):
        self.t += 1
        for k in self.params:
            # Add L2 regularization to gradients (only for weight matrices)
            actual_grad = grads[k]
            if k.startswith('W'):
                actual_grad += lambda_l2 * self.params[k]
                
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * actual_grad
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (actual_grad ** 2)
            
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            
            self.params[k] -= lr * m_hat / (cp.sqrt(v_hat) + eps)

    def evaluate(self, X, y, batch_size=256):
        """
        Evaluate model on given dataset (forward pass only, no backprop).
        
        Args:
            X: Input sequences (n_samples, seq_len, input_size)
            y: True labels (n_samples,)
            batch_size: Batch size for evaluation
            
        Returns:
            avg_loss: Average binary cross-entropy loss
            accuracy: Classification accuracy
        """
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        
        n_samples = len(X_gpu)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        correct = 0
        eps = 1e-9  # Prevent log(0)
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_gpu[start:end]
            y_batch = y_gpu[start:end]
            
            # Forward pass only
            y_pred, _, _, _ = self.forward_batch(X_batch)
            
            # Binary cross-entropy loss with epsilon
            batch_loss = -cp.mean(
                y_batch * cp.log(y_pred + eps) + 
                (1 - y_batch) * cp.log(1 - y_pred + eps)
            )
            total_loss += float(batch_loss) * len(y_batch)
            
            # Accuracy - lowered threshold for imbalanced data
            predictions = (y_pred >= 0.3).astype(cp.float32)
            correct += int(cp.sum(predictions == y_batch))
        
        avg_loss = total_loss / n_samples
        accuracy = correct / n_samples
        
        return avg_loss, accuracy

    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=10, batch_size=64, lr=0.001, print_every=1, 
              patience=7, lambda_l2=0.01):
        """
        Batched training loop with optional validation tracking.
        
        Args:
            X_train: Training sequences (n_samples, seq_len, input_size)
            y_train: Training labels (n_samples,)
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            lr: Learning rate
            print_every: Print progress every N epochs
            
        Returns:
            history: Dictionary with 'train_loss', 'val_loss', 'val_acc'
        """
        X_gpu = cp.asarray(X_train, dtype=cp.float32)
        y_gpu = cp.asarray(y_train, dtype=cp.float32)
        
        n_samples = len(X_gpu)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'train_acc': []
        }
        
        has_validation = X_val is not None and y_val is not None
        
        # Initialize Early Stopping
        best_val_loss = float('inf')
        wait = 0
        best_params = None
        
        print(f"Training: {n_samples} samples, {n_batches} batches/epoch, batch_size={batch_size}")
        if has_validation:
            print(f"Validation: {len(X_val)} samples | Early stopping patience: {patience}")
        print("-" * 70)
        
        eps = 1e-9  # Prevent log(0)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = cp.random.permutation(n_samples)
            X_shuffled = X_gpu[indices]  # Shuffled input sequences
            y_shuffled = y_gpu[indices]  # Shuffled labels
            
            epoch_loss = 0 # Initialize epoch loss
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward
                y_pred, caches, _, _ = self.forward_batch(X_batch)
                
                # Loss (BCE) with epsilon to prevent log(0)
                loss = -cp.mean(
                    y_batch * cp.log(y_pred + eps) + 
                    (1 - y_batch) * cp.log(1 - y_pred + eps)
                )
                epoch_loss += float(loss)
                
                # Backward
                grads = self.backward_batch(y_pred, y_batch, caches, self.cw)
                
                # Update (Adam)
                self.update_adam(grads, lr=lr, lambda_l2=lambda_l2)
            
            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)
            
            # Compute training accuracy
            _, train_acc = self.evaluate(X_train, y_train, batch_size)
            history['train_acc'].append(train_acc)
            
            # Validation evaluation
            if has_validation:
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    best_params = {k: v.copy() for k, v in self.params.items()}  # Save best weights
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                        self.params = best_params  # Restore best weights
                        break
            
            if GPU_AVAILABLE:
                cp.cuda.Stream.null.synchronize()
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                if has_validation:
                    # Determine if overfitting or underfitting
                    gap = val_loss - avg_train_loss
                    status = ""
                    if gap > 0.1:
                        status = " [OVERFITTING]"
                    elif avg_train_loss > 0.5 and val_loss > 0.5:
                        status = " [UNDERFITTING]"
                    
                    print(f"Epoch {epoch+1:>4}/{epochs} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f}{status}")
                else:
                    print(f"Epoch {epoch+1:>4}/{epochs} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f}")
        
        print("-" * 70)
        if has_validation:
            print(f"Final - Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f} | "
                  f"Val Acc: {history['val_acc'][-1]:.4f}")
        
        return history

    def predict(self, X, batch_size=256):
        """Batched prediction."""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        n_samples = len(X_gpu)
        predictions = []
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_gpu[start:end]
            
            y_pred, _, _, _ = self.forward_batch(X_batch)
            
            if GPU_AVAILABLE:
                predictions.extend(cp.asnumpy(y_pred).tolist())
            else:
                predictions.extend(y_pred.tolist())
        
        return np.array(predictions)
