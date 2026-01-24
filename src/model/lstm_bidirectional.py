"""
Bi-directional LSTM Model with GPU Optimization (Adam Optimizer).
Extends LSTMModelGPUOptimized with bidirectional processing for enhanced sequence understanding.

Architecture:
- Forward LSTM: Processes sequence from t=0 to T-1
- Backward LSTM: Processes sequence from t=T-1 to 0
- Output: Concatenation of final hidden states from both directions

Mathematical formulation (for each direction d ∈ {f, b}):
    f_t^d = σ(W_f^d z + b_f^d)
    i_t^d = σ(W_i^d z + b_i^d)
    c_t^d = f_t^d ⊙ c_{prev}^d + i_t^d ⊙ tanh(W_c^d z + b_c^d)
    o_t^d = σ(W_o^d z + b_o^d)
    h_t^d = o_t^d ⊙ tanh(c_t^d)
    
    y = σ(W_y [h_T^f; h_0^b] + b_y)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_cupy_optimized import LSTMModelGPUOptimized


class BiLSTMModelGPUOptimized(LSTMModelGPUOptimized):
    """
    Bi-directional LSTM with Adam optimizer and GPU acceleration.
    
    Processes sequences in both forward and backward directions,
    capturing context from both past and future timesteps.
    
    Inherits from LSTMModelGPUOptimized:
    - sigmoid(), evaluate(), train(), predict() methods
    
    Overrides:
    - __init__(): Creates separate forward/backward parameters
    - forward_batch(): Processes both directions
    - backward_batch(): Computes gradients for both directions
    - update_adam(): Updates all parameters
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        """
        Initialize Bi-directional LSTM with Adam optimizer.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units (per direction)
            output_size: Number of output units (default 1 for binary classification)
            cw: Class weight for positive class (default 1)
        """
        # Store dimensions (don't call parent __init__ to avoid duplicate params)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cw = cw
        
        z_dim = hidden_size + input_size
        scale = 0.1
        
        # Initialize parameters for both directions
        self.params = {}
        
        # Forward LSTM parameters (suffix _f)
        self.params['Wf_f'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['Wi_f'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['Wc_f'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['Wo_f'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['bf_f'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        self.params['bi_f'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        self.params['bc_f'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        self.params['bo_f'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        
        # Backward LSTM parameters (suffix _b)
        self.params['Wf_b'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['Wi_b'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['Wc_b'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['Wo_b'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['bf_b'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        self.params['bi_b'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        self.params['bc_b'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        self.params['bo_b'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        
        # Output layer - receives concatenated hidden states (hidden_size * 2)
        self.params['Wy'] = cp.random.randn(output_size, hidden_size * 2).astype(cp.float32) * scale
        self.params['by'] = cp.zeros((output_size, 1), dtype=cp.float32)
        
        # Adam optimizer states for ALL parameters
        self.m = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
    
    def _lstm_cell_forward(self, x_t, h_prev, c_prev, direction='f'):
        """
        Single LSTM cell forward pass for specified direction.
        
        Args:
            x_t: Input at timestep t (input_size, batch_size)
            h_prev: Previous hidden state (hidden_size, batch_size)
            c_prev: Previous cell state (hidden_size, batch_size)
            direction: 'f' for forward, 'b' for backward
            
        Returns:
            h_new, c_new, cache
        """
        # Concatenate h and x
        z = cp.vstack((h_prev, x_t))
        
        # Get parameters for this direction
        Wf = self.params[f'Wf_{direction}']
        Wi = self.params[f'Wi_{direction}']
        Wc = self.params[f'Wc_{direction}']
        Wo = self.params[f'Wo_{direction}']
        bf = self.params[f'bf_{direction}']
        bi = self.params[f'bi_{direction}']
        bc = self.params[f'bc_{direction}']
        bo = self.params[f'bo_{direction}']
        
        # Gates
        f = self.sigmoid(cp.dot(Wf, z) + bf)
        i = self.sigmoid(cp.dot(Wi, z) + bi)
        c_bar = cp.tanh(cp.dot(Wc, z) + bc)
        o = self.sigmoid(cp.dot(Wo, z) + bo)
        
        # Cell and hidden state
        c_new = f * c_prev + i * c_bar
        h_new = o * cp.tanh(c_new)
        
        # Cache for backprop
        cache = (z, f, i, c_bar, c_new, o, h_new, c_prev, h_prev)
        
        return h_new, c_new, cache
    
    def forward_batch(self, X_batch):
        """
        Batched forward pass with bidirectional processing.
        
        Args:
            X_batch: Input batch (batch_size, seq_len, input_size)
            
        Returns:
            y_pred: Predictions (batch_size,)
            caches: Tuple of (forward_caches, backward_caches)
            h_combined: Combined final hidden state
            c_combined: Tuple of final cell states
        """
        batch_size, seq_len, _ = X_batch.shape
        
        # Initialize hidden states for both directions
        h_f = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c_f = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        h_b = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c_b = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        forward_caches = []
        backward_caches = []
        
        # Forward direction: t = 0 to T-1
        for t in range(seq_len):
            x_t = X_batch[:, t, :].T  # (input_size, batch_size)
            h_f, c_f, cache = self._lstm_cell_forward(x_t, h_f, c_f, direction='f')
            forward_caches.append(cache)
        
        # Backward direction: t = T-1 to 0
        for t in range(seq_len - 1, -1, -1):
            x_t = X_batch[:, t, :].T  # (input_size, batch_size)
            h_b, c_b, cache = self._lstm_cell_forward(x_t, h_b, c_b, direction='b')
            backward_caches.append(cache)
        
        # Note: backward_caches[0] corresponds to t=T-1, backward_caches[-1] to t=0
        # Reverse to align with forward indices
        backward_caches = backward_caches[::-1]
        
        # Concatenate final hidden states: h_f[-1] and h_b[0] (after reversal)
        # h_f is the state after processing t=T-1 (forward)
        # h_b is the state after processing t=0 (backward, which is the "last" backward step)
        h_combined = cp.vstack((h_f, h_b))  # (hidden_size * 2, batch_size)
        
        # Output layer
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], h_combined) + self.params['by'])
        
        return y_pred.flatten(), (forward_caches, backward_caches), h_combined, (c_f, c_b)
    
    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass for bidirectional LSTM.
        
        Computes gradients for both forward and backward LSTM cells.
        
        Args:
            y_pred: Predictions from forward pass
            y_true: True labels
            caches: Tuple of (forward_caches, backward_caches)
            cw: Class weight for positive class
            
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        batch_size = len(y_true)
        forward_caches, backward_caches = caches
        seq_len = len(forward_caches)
        
        # Initialize gradients
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # Output gradient with class weighting
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        # Get final hidden states
        h_f_final = forward_caches[-1][6]  # h_new from last forward timestep
        h_b_final = backward_caches[0][6]   # h_new from first backward timestep (t=0)
        h_combined = cp.vstack((h_f_final, h_b_final))
        
        # Output layer gradients
        grads['Wy'] = cp.dot(dy, h_combined.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size
        
        # Split gradient to forward and backward paths
        dh_combined = cp.dot(self.params['Wy'].T, dy)  # (hidden_size * 2, batch_size)
        dh_f_next = dh_combined[:self.hidden_size, :]
        dh_b_next = dh_combined[self.hidden_size:, :]
        
        # BPTT for Forward LSTM (t = T-1 to 0)
        dc_f_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, c, o, h, c_prev, h_prev = forward_caches[t]
            
            dh = dh_f_next
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo_f'] += cp.dot(da_o, z.T) / batch_size
            grads['bo_f'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # Cell state
            dc = dh * o * (1 - cp.tanh(c)**2) + dc_f_next
            
            # Candidate
            dc_bar = dc * i
            da_c = dc_bar * (1 - c_bar**2)
            grads['Wc_f'] += cp.dot(da_c, z.T) / batch_size
            grads['bc_f'] += cp.sum(da_c, axis=1, keepdims=True) / batch_size
            
            # Input gate
            di = dc * c_bar
            da_i = di * i * (1 - i)
            grads['Wi_f'] += cp.dot(da_i, z.T) / batch_size
            grads['bi_f'] += cp.sum(da_i, axis=1, keepdims=True) / batch_size
            
            # Forget gate
            df = dc * c_prev
            da_f = df * f * (1 - f)
            grads['Wf_f'] += cp.dot(da_f, z.T) / batch_size
            grads['bf_f'] += cp.sum(da_f, axis=1, keepdims=True) / batch_size
            
            # Gradients for next timestep
            dz = (cp.dot(self.params['Wf_f'].T, da_f) +
                  cp.dot(self.params['Wi_f'].T, da_i) +
                  cp.dot(self.params['Wc_f'].T, da_c) +
                  cp.dot(self.params['Wo_f'].T, da_o))
            
            dh_f_next = dz[:self.hidden_size, :]
            dc_f_next = f * dc
        
        # BPTT for Backward LSTM (t = 0 to T-1, processing backward_caches in reverse)
        dc_b_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        # backward_caches is already aligned: [0] = t=0, [-1] = t=T-1
        # We backprop through the backward LSTM in the order it was computed
        # which means from t=0 to t=T-1 (the reverse of forward)
        for t in range(seq_len):
            z, f, i, c_bar, c, o, h, c_prev, h_prev = backward_caches[t]
            
            # Only the first timestep (t=0) receives gradient from output
            if t == 0:
                dh = dh_b_next
            else:
                dh = dh_b_next
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo_b'] += cp.dot(da_o, z.T) / batch_size
            grads['bo_b'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # Cell state
            dc = dh * o * (1 - cp.tanh(c)**2) + dc_b_next
            
            # Candidate
            dc_bar = dc * i
            da_c = dc_bar * (1 - c_bar**2)
            grads['Wc_b'] += cp.dot(da_c, z.T) / batch_size
            grads['bc_b'] += cp.sum(da_c, axis=1, keepdims=True) / batch_size
            
            # Input gate
            di = dc * c_bar
            da_i = di * i * (1 - i)
            grads['Wi_b'] += cp.dot(da_i, z.T) / batch_size
            grads['bi_b'] += cp.sum(da_i, axis=1, keepdims=True) / batch_size
            
            # Forget gate
            df = dc * c_prev
            da_f = df * f * (1 - f)
            grads['Wf_b'] += cp.dot(da_f, z.T) / batch_size
            grads['bf_b'] += cp.sum(da_f, axis=1, keepdims=True) / batch_size
            
            # Gradients for next timestep
            dz = (cp.dot(self.params['Wf_b'].T, da_f) +
                  cp.dot(self.params['Wi_b'].T, da_i) +
                  cp.dot(self.params['Wc_b'].T, da_c) +
                  cp.dot(self.params['Wo_b'].T, da_o))
            
            dh_b_next = dz[:self.hidden_size, :]
            dc_b_next = f * dc
        
        # Gradient clipping
        for k in grads:
            grads[k] = cp.clip(grads[k], -5, 5)
        
        return grads
    
    def update_adam(self, grads, lr=0.001, beta1=0.9, beta2=0.999, 
                    eps=1e-8, lambda_l2=0.01):
        """
        Adam optimizer update for all bidirectional parameters.
        
        Updates all parameters for both forward and backward LSTMs.
        """
        self.t += 1
        for k in self.params:
            # Add L2 regularization (for W parameters only)
            actual_grad = grads[k]
            if k.startswith('W'):
                actual_grad = actual_grad + lambda_l2 * self.params[k]
            
            # Adam update
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * actual_grad
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (actual_grad ** 2)
            
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            
            self.params[k] -= lr * m_hat / (cp.sqrt(v_hat) + eps)
