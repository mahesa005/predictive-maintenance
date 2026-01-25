"""
Residual LSTM Model with GPU Optimization.

This variant adds skip connections (residual connections) that connect the input
directly to the output, preventing signal degradation on long time windows.

Formula: h_t = h_lstm + projection(x_t)

Reference: He et al., "Deep Residual Learning for Image Recognition" (2016)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_cupy_optimized import LSTMModelGPUOptimized


class ResidualLSTMModelGPUOptimized(LSTMModelGPUOptimized):
    """
    Residual LSTM for GPU with skip connections.
    
    Inherits from LSTMModelGPUOptimized and modifies:
    - __init__: Adds W_skip projection matrix if input_size != hidden_size
    - forward_batch: Adds skip connection h_t = h_lstm + projection(x_t)
    - backward_batch: Gradient flows through both LSTM and skip paths
    
    Residual connections provide:
    - Better gradient flow through identity path
    - Prevention of signal degradation on long sequences
    - Deeper effective model capacity
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        # Call parent constructor first
        super().__init__(input_size, hidden_size, output_size, cw)
        
        self.use_projection = (input_size != hidden_size)
        
        # Add projection matrix if dimensions don't match
        if self.use_projection:
            scale = 0.1
            self.params['W_skip'] = cp.random.randn(hidden_size, input_size).astype(cp.float32) * scale
            self.params['b_skip'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        
        # Reinitialize Adam optimizer states to include new parameters
        self.m = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in self.params.items()}

    def forward_batch(self, X_batch):
        """
        Batched forward pass with residual/skip connections.
        X_batch: (batch_size, seq_len, input_size)
        
        Key modification: h_t = h_lstm + projection(x_t)
        
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
            
            # Standard LSTM gates
            f = self.sigmoid(cp.dot(self.params['Wf'], z) + self.params['bf'])
            i = self.sigmoid(cp.dot(self.params['Wi'], z) + self.params['bi'])
            c_bar = cp.tanh(cp.dot(self.params['Wc'], z) + self.params['bc'])
            o = self.sigmoid(cp.dot(self.params['Wo'], z) + self.params['bo'])
            
            # Calculate new cell state
            c_new = f * c + i * c_bar
            
            # LSTM hidden state (before residual)
            h_lstm = o * cp.tanh(c_new)
            
            # Skip connection: project x_t to hidden_size if needed
            if self.use_projection:
                x_skip = cp.dot(self.params['W_skip'], x_t) + self.params['b_skip']
            else:
                x_skip = x_t
            
            # Residual connection: h_t = h_lstm + projection(x_t)
            h_new = h_lstm + x_skip
            
            # Save cache for backprop (include x_t and x_skip for residual gradient)
            caches.append((z, f, i, c_bar, c_new, o, h_lstm, h_new, c, h, x_t, x_skip))
            
            # Update hidden and cell state
            h, c = h_new, c_new
        
        # Output layer: (output_size, batch_size)
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], h) + self.params['by'])
        
        return y_pred.flatten(), caches, h, c

    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass using BPTT with residual gradient flow.
        
        Gradient flows through both:
        1. LSTM path (standard BPTT)
        2. Skip connection path (direct to x_t and W_skip)
        """
        batch_size = len(y_true)
        seq_len = len(caches)
        
        # Initialize gradients
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # Output gradient: (1, batch_size)
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        
        # Adjust gradient for class 1 using class weight
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        # Get final hidden state from last cache
        h_final = caches[-1][7]  # h_new (with residual) from last timestep
        
        # Calculate gradients for output layer
        grads['Wy'] = cp.dot(dy, h_final.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size
        
        # Backprop through hidden state
        dh_next = cp.dot(self.params['Wy'].T, dy)
        dc_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, c, o, h_lstm, h_new, c_prev, h_prev, x_t, x_skip = caches[t]
            
            # dh is gradient w.r.t h_new (after residual)
            dh = dh_next
            
            # Gradient splits into two paths:
            # 1. LSTM path: dh_lstm = dh (gradient flows to h_lstm)
            # 2. Skip path: dx_skip = dh (gradient flows to skip connection)
            
            dh_lstm = dh  # Gradient to LSTM hidden state
            dx_skip = dh  # Gradient to skip connection
            
            # === Skip connection gradients ===
            if self.use_projection:
                # dx_skip flows through W_skip: x_skip = W_skip @ x_t + b_skip
                grads['W_skip'] += cp.dot(dx_skip, x_t.T) / batch_size
                grads['b_skip'] += cp.sum(dx_skip, axis=1, keepdims=True) / batch_size
                # Gradient to x_t from skip path
                dx_t_skip = cp.dot(self.params['W_skip'].T, dx_skip)
            else:
                # If no projection, gradient flows directly to x_t
                dx_t_skip = dx_skip
            
            # === LSTM path gradients ===
            # Output gate
            do = dh_lstm * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo'] += cp.dot(da_o, z.T) / batch_size 
            grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size

            # Cell state
            dc = dh_lstm * o * (1 - cp.tanh(c)**2) + dc_next
            
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
            
            # Gradients for next timestep through LSTM path
            dz = (cp.dot(self.params['Wf'].T, da_f) +
                  cp.dot(self.params['Wi'].T, da_i) +
                  cp.dot(self.params['Wc'].T, da_c) +
                  cp.dot(self.params['Wo'].T, da_o))
            
            # dz contains gradients for [h, x]
            # dh_next from LSTM path
            dh_next_lstm = dz[:self.hidden_size, :]
            # dx_t from LSTM path (through z concatenation)
            dx_t_lstm = dz[self.hidden_size:, :]
            
            # Total gradient to h_prev: only from LSTM path
            dh_next = dh_next_lstm
            
            # Note: dx_t_skip and dx_t_lstm both contribute to the gradient
            # w.r.t x_t, but since x_t is an input (not a parameter), we don't
            # need to accumulate these gradients explicitly. The gradient flows
            # back through the sequence correctly via dh_next and dc_next.
            
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
        
        residual_total = self.count_parameters()
        
        if self.use_projection:
            increase = (residual_total - standard_total) / standard_total * 100
            return {
                'standard_lstm_params': standard_total,
                'residual_lstm_params': residual_total,
                'parameters_added': residual_total - standard_total,
                'increase_percentage': increase,
                'uses_projection': True
            }
        else:
            return {
                'standard_lstm_params': standard_total,
                'residual_lstm_params': residual_total,
                'parameters_added': 0,
                'increase_percentage': 0.0,
                'uses_projection': False
            }
