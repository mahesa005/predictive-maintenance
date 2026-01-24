"""
Nested LSTM (Cell-in-Cell) Model with GPU Optimization (Adam Optimizer).
Extends LSTMModelGPUOptimized with inner gates for hierarchical cell state updates.

Nested LSTM Concept:
Instead of linear cell state update, the cell state is treated as an inner LSTM system.

Mathematical formulation:
    Standard outer gates:
        f_t = σ(W_f z + b_f)
        i_t = σ(W_i z + b_i)
        c̃_t = tanh(W_c z + b_c)
        o_t = σ(W_o z + b_o)
    
    Inner gates:
        f_inner = σ(W_fi z + b_fi)
        i_inner = σ(W_ii z + b_ii)
    
    Hierarchical cell state update:
        c_temp = f_t ⊙ c_prev + i_t ⊙ c̃_t
        c_t = f_inner ⊙ c_prev + i_inner ⊙ tanh(c_temp)
    
    Hidden state:
        h_t = o_t ⊙ tanh(c_t)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_cupy_optimized import LSTMModelGPUOptimized


class NestedLSTMModelGPUOptimized(LSTMModelGPUOptimized):
    """
    Nested LSTM with inner gates for hierarchical cell state updates.
    
    The cell state update is treated as a nested LSTM operation,
    allowing for more complex memory dynamics.
    
    Inherits from LSTMModelGPUOptimized:
    - sigmoid(), evaluate(), train(), predict() methods
    
    Overrides:
    - __init__(): Adds inner gate parameters
    - forward_batch(): Implements nested cell update
    - backward_batch(): Computes gradients for all gates including inner gates
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        """
        Initialize Nested LSTM with inner gates.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units (default 1 for binary classification)
            cw: Class weight for positive class (default 1)
        """
        # Initialize parent class (sets up outer gate params and Adam states)
        super().__init__(input_size, hidden_size, output_size, cw)
        
        z_dim = hidden_size + input_size
        scale = 0.1
        
        # Add inner gate parameters for nested cell update
        # f_inner: Inner forget gate
        self.params['Wf_i'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['bf_i'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        
        # i_inner: Inner input gate
        self.params['Wi_i'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['bi_i'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        
        # Add Adam optimizer states for inner gate parameters
        self.m['Wf_i'] = cp.zeros_like(self.params['Wf_i'])
        self.m['bf_i'] = cp.zeros_like(self.params['bf_i'])
        self.m['Wi_i'] = cp.zeros_like(self.params['Wi_i'])
        self.m['bi_i'] = cp.zeros_like(self.params['bi_i'])
        
        self.v['Wf_i'] = cp.zeros_like(self.params['Wf_i'])
        self.v['bf_i'] = cp.zeros_like(self.params['bf_i'])
        self.v['Wi_i'] = cp.zeros_like(self.params['Wi_i'])
        self.v['bi_i'] = cp.zeros_like(self.params['bi_i'])
    
    def forward_batch(self, X_batch):
        """
        Batched forward pass with nested cell state update.
        
        Args:
            X_batch: Input batch (batch_size, seq_len, input_size)
            
        Returns:
            y_pred: Predictions (batch_size,)
            caches: List of cache tuples for backprop
            h: Final hidden state
            c: Final cell state
        """
        batch_size, seq_len, _ = X_batch.shape
        
        # Initialize hidden states for batch
        h = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        caches = []
        
        for t in range(seq_len):
            # x_t: (input_size, batch_size)
            x_t = X_batch[:, t, :].T
            
            # Concatenate h and x
            z = cp.vstack((h, x_t))
            
            # Store c_prev for gradient computation
            c_prev = c
            
            # ===== Outer Gates (Standard LSTM) =====
            f = self.sigmoid(cp.dot(self.params['Wf'], z) + self.params['bf'])
            i = self.sigmoid(cp.dot(self.params['Wi'], z) + self.params['bi'])
            c_bar = cp.tanh(cp.dot(self.params['Wc'], z) + self.params['bc'])
            o = self.sigmoid(cp.dot(self.params['Wo'], z) + self.params['bo'])
            
            # ===== Inner Gates (Nested LSTM) =====
            f_inner = self.sigmoid(cp.dot(self.params['Wf_i'], z) + self.params['bf_i'])
            i_inner = self.sigmoid(cp.dot(self.params['Wi_i'], z) + self.params['bi_i'])
            
            # ===== Hierarchical Cell State Update =====
            # Step 1: Standard cell update (temporary)
            c_temp = f * c_prev + i * c_bar
            
            # Step 2: Inner nested update
            c_new = f_inner * c_prev + i_inner * cp.tanh(c_temp)
            
            # Hidden state (uses outer output gate)
            h_new = o * cp.tanh(c_new)
            
            # Save cache for backprop (includes inner gates and c_temp)
            caches.append((z, f, i, c_bar, o, f_inner, i_inner, c_temp, c_new, h_new, c_prev, h))
            
            # Update states
            h, c = h_new, c_new
        
        # Output layer
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], h) + self.params['by'])
        
        return y_pred.flatten(), caches, h, c
    
    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass with inner gate gradient computation.
        
        Chain rule is more complex because c_prev appears in multiple places:
        - In outer forget gate: f * c_prev
        - In inner forget gate: f_inner * c_prev
        
        Args:
            y_pred: Predictions from forward pass
            y_true: True labels
            caches: Caches from forward pass
            cw: Class weight for positive class
            
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        batch_size = len(y_true)
        seq_len = len(caches)
        
        # Initialize gradients (includes inner gate parameters)
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # Output gradient with class weighting
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        # Get final hidden state from last cache
        h_final = caches[-1][9]  # h_new from last timestep
        
        # Output layer gradients
        grads['Wy'] = cp.dot(dy, h_final.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size
        
        # Backprop through hidden state
        dh_next = cp.dot(self.params['Wy'].T, dy)
        dc_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, o, f_inner, i_inner, c_temp, c, h, c_prev, h_prev = caches[t]
            
            dh = dh_next
            
            # ===== Output Gate (Outer) =====
            # h = o * tanh(c)
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo'] += cp.dot(da_o, z.T) / batch_size
            grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # ===== Cell State Gradient =====
            # Gradient through h = o * tanh(c)
            dc = dh * o * (1 - cp.tanh(c)**2) + dc_next
            
            # ===== Inner Gates Backprop =====
            # c = f_inner * c_prev + i_inner * tanh(c_temp)
            
            # Inner input gate gradient
            tanh_c_temp = cp.tanh(c_temp)
            di_inner = dc * tanh_c_temp
            da_i_inner = di_inner * i_inner * (1 - i_inner)
            grads['Wi_i'] += cp.dot(da_i_inner, z.T) / batch_size
            grads['bi_i'] += cp.sum(da_i_inner, axis=1, keepdims=True) / batch_size
            
            # Inner forget gate gradient
            df_inner = dc * c_prev
            da_f_inner = df_inner * f_inner * (1 - f_inner)
            grads['Wf_i'] += cp.dot(da_f_inner, z.T) / batch_size
            grads['bf_i'] += cp.sum(da_f_inner, axis=1, keepdims=True) / batch_size
            
            # Gradient through tanh(c_temp)
            dc_temp = dc * i_inner * (1 - tanh_c_temp**2)
            
            # ===== Outer Gates Backprop (through c_temp) =====
            # c_temp = f * c_prev + i * c_bar
            
            # Candidate gradient
            dc_bar = dc_temp * i
            da_c = dc_bar * (1 - c_bar**2)
            grads['Wc'] += cp.dot(da_c, z.T) / batch_size
            grads['bc'] += cp.sum(da_c, axis=1, keepdims=True) / batch_size
            
            # Outer input gate gradient
            di = dc_temp * c_bar
            da_i = di * i * (1 - i)
            grads['Wi'] += cp.dot(da_i, z.T) / batch_size
            grads['bi'] += cp.sum(da_i, axis=1, keepdims=True) / batch_size
            
            # Outer forget gate gradient
            df = dc_temp * c_prev
            da_f = df * f * (1 - f)
            grads['Wf'] += cp.dot(da_f, z.T) / batch_size
            grads['bf'] += cp.sum(da_f, axis=1, keepdims=True) / batch_size
            
            # ===== Gradients for next timestep =====
            # Gradient through z (for all gates)
            dz = (cp.dot(self.params['Wf'].T, da_f) +
                  cp.dot(self.params['Wi'].T, da_i) +
                  cp.dot(self.params['Wc'].T, da_c) +
                  cp.dot(self.params['Wo'].T, da_o) +
                  cp.dot(self.params['Wf_i'].T, da_f_inner) +
                  cp.dot(self.params['Wi_i'].T, da_i_inner))
            
            dh_next = dz[:self.hidden_size, :]
            
            # dc_prev has contributions from multiple paths:
            # 1. Through outer forget: f * c_prev (in c_temp)
            # 2. Through inner forget: f_inner * c_prev
            dc_next = (dc_temp * f +      # From outer forget in c_temp
                      dc * f_inner)        # From inner forget
        
        # Gradient clipping
        for k in grads:
            grads[k] = cp.clip(grads[k], -5, 5)
        
        return grads
    
    def update_adam(self, grads, lr=0.001, beta1=0.9, beta2=0.999, 
                    eps=1e-8, lambda_l2=0.01):
        """
        Adam optimizer update for all parameters including inner gates.
        """
        self.t += 1
        for k in self.params:
            # Add L2 regularization (for W parameters)
            actual_grad = grads[k]
            if k.startswith('W'):
                actual_grad = actual_grad + lambda_l2 * self.params[k]
            
            # Adam update
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * actual_grad
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (actual_grad ** 2)
            
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            
            self.params[k] -= lr * m_hat / (cp.sqrt(v_hat) + eps)
