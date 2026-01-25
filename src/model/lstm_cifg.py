"""
CIFG (Coupled Input and Forget Gate) LSTM Model with GPU Optimization.

This variant couples the input gate (i) with the forget gate (f) using the relationship:
    i_t = 1 - f_t

This reduces parameter count by ~25% compared to standard LSTM while maintaining
competitive performance. The coupled gates enforce a constraint that the total
contribution of old and new information sums to 1.

Reference: Greff et al., "LSTM: A Search Space Odyssey" (2017)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_cupy_optimized import LSTMModelGPUOptimized


class CIFGLSTMModelGPUOptimized(LSTMModelGPUOptimized):
    """
    Coupled Input and Forget Gate (CIFG) LSTM for GPU.
    
    Inherits from LSTMModelGPUOptimized and modifies:
    - __init__: Removes Wi and bi parameters (~25% reduction)
    - forward_batch: Computes i_t = 1 - f_t instead of separate input gate
    - backward_batch: Redirects input gate gradients to forget gate (negated)
    
    This enforces that when the model "forgets" more (high f_t), it also
    "accepts" less new information (low i_t), and vice versa.
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        # Call parent constructor first
        super().__init__(input_size, hidden_size, output_size, cw)
        
        # CIFG: Remove Wi and bi parameters
        del self.params['Wi']
        del self.params['bi']
        
        # Reinitialize Adam optimizer states without Wi and bi
        self.m = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in self.params.items()}

    def forward_batch(self, X_batch):
        """
        Batched forward pass with CIFG modification.
        X_batch: (batch_size, seq_len, input_size)
        
        Key modification: i_t = 1 - f_t (coupled gates)
        Cell state: c_t = f_t * c_{t-1} + (1 - f_t) * c_bar_t
        
        Returns: predictions (batch_size,), caches, h, c
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
            
            # Gates (CIFG modification)
            f = self.sigmoid(cp.dot(self.params['Wf'], z) + self.params['bf'])
            i = 1 - f  # CIFG: Input gate coupled with forget gate
            c_bar = cp.tanh(cp.dot(self.params['Wc'], z) + self.params['bc'])
            o = self.sigmoid(cp.dot(self.params['Wo'], z) + self.params['bo'])
            
            # Calculate new cell state using CIFG formula:
            # c_t = f_t * c_{t-1} + (1 - f_t) * c_bar_t
            c_new = f * c + i * c_bar  # Note: i = 1 - f
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
        Batched backward pass using BPTT with CIFG modification.
        
        Key modification: Since i_t = 1 - f_t, gradients that would flow
        to the input gate (di) now flow to the forget gate with negative sign:
        
        df_combined = df - di (because d(1-f)/df = -1)
        """
        batch_size = len(y_true)
        seq_len = len(caches)
        
        # Initialize gradients (no Wi, bi in CIFG)
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # Output gradient: (1, batch_size)
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        
        # Adjust gradient for class 1 using class weight
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        # Get final hidden state from last cache
        h_final = caches[-1][6]  # h_new from last timestep
        
        # Calculate gradients for output layer
        grads['Wy'] = cp.dot(dy, h_final.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size
        
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
            dc_bar = dc * i  # Note: i = 1 - f
            da_c = dc_bar * (1 - c_bar**2)
            grads['Wc'] += cp.dot(da_c, z.T) / batch_size
            grads['bc'] += cp.sum(da_c, axis=1, keepdims=True) / batch_size
            
            # CIFG: Combined gradient for forget gate
            # df = dc * c_prev (original forget gate gradient)
            # di = dc * c_bar (gradient that would go to input gate)
            # Since i = 1 - f, di/df = -1, so: df_combined = df - di
            df = dc * c_prev
            di = dc * c_bar
            df_combined = df - di  # Equivalent to df + (-1) * di
            
            da_f = df_combined * f * (1 - f)
            grads['Wf'] += cp.dot(da_f, z.T) / batch_size
            grads['bf'] += cp.sum(da_f, axis=1, keepdims=True) / batch_size
            
            # Gradients for next timestep (no Wi contribution in CIFG)
            dz = (cp.dot(self.params['Wf'].T, da_f) +
                  cp.dot(self.params['Wc'].T, da_c) +
                  cp.dot(self.params['Wo'].T, da_o))
            
            dh_next = dz[:self.hidden_size, :]
            dc_next = f * dc
        
        # Gradient clipping
        for k in grads:
            grads[k] = cp.clip(grads[k], -5, 5)
        
        return grads
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(v.size for v in self.params.values())
    
    def compare_with_standard_lstm(self):
        """Compare parameter count with standard LSTM."""
        z_dim = self.hidden_size + self.input_size
        
        # Standard LSTM parameters
        standard_total = (
            4 * self.hidden_size * z_dim +  # Wf, Wi, Wc, Wo
            self.output_size * self.hidden_size +  # Wy
            4 * self.hidden_size +  # bf, bi, bc, bo
            self.output_size  # by
        )
        
        cifg_total = self.count_parameters()
        reduction = (standard_total - cifg_total) / standard_total * 100
        
        return {
            'standard_lstm_params': standard_total,
            'cifg_lstm_params': cifg_total,
            'parameters_removed': standard_total - cifg_total,
            'reduction_percentage': reduction
        }
