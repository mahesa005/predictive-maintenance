"""
Layer Normalization LSTM Model with GPU Optimization.

This variant applies Layer Normalization before activation functions for improved
convergence speed and stability against extreme input fluctuations.

LayerNorm formula: LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β

Reference: Ba et al., "Layer Normalization" (2016)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_cupy_optimized import LSTMModelGPUOptimized


class LayerNormLSTMModelGPUOptimized(LSTMModelGPUOptimized):
    """
    Layer Normalization LSTM for GPU.
    
    Inherits from LSTMModelGPUOptimized and modifies:
    - __init__: Adds gamma and beta parameters for each gate (f, i, c, o)
    - forward_batch: Applies LayerNorm before activation functions
    - backward_batch: Implements LayerNorm gradient computation
    
    LayerNorm provides:
    - Faster convergence
    - Stability against extreme input fluctuations
    - Better gradient flow through normalization
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1, eps=1e-5):
        # Call parent constructor first
        super().__init__(input_size, hidden_size, output_size, cw)
        
        self.eps = eps  # Epsilon for numerical stability
        
        # Add LayerNorm parameters (gamma and beta) for each gate
        # gamma: scale parameter, beta: shift parameter
        for gate in ['f', 'i', 'c', 'o']:
            self.params[f'gamma_{gate}'] = cp.ones((hidden_size, 1), dtype=cp.float32)
            self.params[f'beta_{gate}'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        
        # Reinitialize Adam optimizer states to include new parameters
        self.m = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in self.params.items()}

    def layer_norm(self, x, gamma, beta):
        """
        Apply Layer Normalization along hidden dimension (axis=0).
        
        Args:
            x: Input tensor (hidden_size, batch_size)
            gamma: Scale parameter (hidden_size, 1)
            beta: Shift parameter (hidden_size, 1)
            
        Returns:
            x_norm: Normalized tensor (hidden_size, batch_size)
            cache: (x, x_centered, std, gamma) for backward pass
        """
        # Compute mean and variance along hidden dimension
        mu = cp.mean(x, axis=0, keepdims=True)  # (1, batch_size)
        x_centered = x - mu
        var = cp.mean(x_centered ** 2, axis=0, keepdims=True)  # (1, batch_size)
        std = cp.sqrt(var + self.eps)  # (1, batch_size)
        
        # Normalize
        x_hat = x_centered / std  # (hidden_size, batch_size)
        
        # Scale and shift
        x_norm = gamma * x_hat + beta  # (hidden_size, batch_size)
        
        cache = (x, x_centered, std, x_hat, gamma)
        return x_norm, cache

    def layer_norm_backward(self, dx_norm, cache):
        """
        Backward pass for Layer Normalization.
        
        Args:
            dx_norm: Gradient w.r.t normalized output (hidden_size, batch_size)
            cache: Cached values from forward pass
            
        Returns:
            dx: Gradient w.r.t input (hidden_size, batch_size)
            dgamma: Gradient w.r.t gamma (hidden_size, 1)
            dbeta: Gradient w.r.t beta (hidden_size, 1)
        """
        x, x_centered, std, x_hat, gamma = cache
        hidden_size = x.shape[0]
        
        # Gradient w.r.t gamma and beta
        dgamma = cp.sum(dx_norm * x_hat, axis=1, keepdims=True)  # (hidden_size, 1)
        dbeta = cp.sum(dx_norm, axis=1, keepdims=True)  # (hidden_size, 1)
        
        # Gradient w.r.t x_hat
        dx_hat = dx_norm * gamma  # (hidden_size, batch_size)
        
        # Gradient w.r.t variance
        dvar = cp.sum(dx_hat * x_centered * (-0.5) * (std ** -3), axis=0, keepdims=True)
        
        # Gradient w.r.t mean
        dmu = cp.sum(dx_hat * (-1 / std), axis=0, keepdims=True) + \
              dvar * cp.mean(-2 * x_centered, axis=0, keepdims=True)
        
        # Gradient w.r.t input
        dx = dx_hat / std + dvar * (2 * x_centered / hidden_size) + dmu / hidden_size
        
        return dx, dgamma, dbeta

    def forward_batch(self, X_batch):
        """
        Batched forward pass with Layer Normalization.
        X_batch: (batch_size, seq_len, input_size)
        
        Key modification: Apply LayerNorm before activation functions
        
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
            
            # Pre-activation values (before LayerNorm and activation)
            pre_f = cp.dot(self.params['Wf'], z) + self.params['bf']
            pre_i = cp.dot(self.params['Wi'], z) + self.params['bi']
            pre_c = cp.dot(self.params['Wc'], z) + self.params['bc']
            pre_o = cp.dot(self.params['Wo'], z) + self.params['bo']
            
            # Apply Layer Normalization before activation
            ln_f, cache_f = self.layer_norm(pre_f, self.params['gamma_f'], self.params['beta_f'])
            ln_i, cache_i = self.layer_norm(pre_i, self.params['gamma_i'], self.params['beta_i'])
            ln_c, cache_c = self.layer_norm(pre_c, self.params['gamma_c'], self.params['beta_c'])
            ln_o, cache_o = self.layer_norm(pre_o, self.params['gamma_o'], self.params['beta_o'])
            
            # Apply activation functions
            f = self.sigmoid(ln_f)
            i = self.sigmoid(ln_i)
            c_bar = cp.tanh(ln_c)
            o = self.sigmoid(ln_o)
            
            # Calculate new cell and hidden state
            c_new = f * c + i * c_bar
            h_new = o * cp.tanh(c_new)
            
            # Save cache for backprop (include LayerNorm caches)
            caches.append((z, f, i, c_bar, c_new, o, h_new, c, h,
                          cache_f, cache_i, cache_c, cache_o,
                          ln_f, ln_i, ln_c, ln_o))
            
            # Update hidden and cell state
            h, c = h_new, c_new
        
        # Output layer: (output_size, batch_size)
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], h) + self.params['by'])
        
        return y_pred.flatten(), caches, h, c

    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass using BPTT with LayerNorm gradient computation.
        
        Includes gradients for:
        - Standard LSTM parameters (Wf, Wi, Wc, Wo, bf, bi, bc, bo)
        - LayerNorm parameters (gamma_f, gamma_i, gamma_c, gamma_o, beta_f, beta_i, beta_c, beta_o)
        """
        batch_size = len(y_true)
        seq_len = len(caches)
        
        # Initialize gradients (including LayerNorm parameters)
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
            (z, f, i, c_bar, c, o, h, c_prev, h_prev,
             cache_f, cache_i, cache_c, cache_o,
             ln_f, ln_i, ln_c, ln_o) = caches[t]
            
            dh = dh_next
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)  # Gradient after sigmoid
            
            # LayerNorm backward for output gate
            dln_o, dgamma_o, dbeta_o = self.layer_norm_backward(da_o, cache_o)
            grads['gamma_o'] += dgamma_o / batch_size
            grads['beta_o'] += dbeta_o / batch_size
            grads['Wo'] += cp.dot(dln_o, z.T) / batch_size 
            grads['bo'] += cp.sum(dln_o, axis=1, keepdims=True) / batch_size

            # Cell state
            dc = dh * o * (1 - cp.tanh(c)**2) + dc_next
            
            # Candidate
            dc_bar = dc * i
            da_c = dc_bar * (1 - c_bar**2)  # Gradient after tanh
            
            # LayerNorm backward for candidate
            dln_c, dgamma_c, dbeta_c = self.layer_norm_backward(da_c, cache_c)
            grads['gamma_c'] += dgamma_c / batch_size
            grads['beta_c'] += dbeta_c / batch_size
            grads['Wc'] += cp.dot(dln_c, z.T) / batch_size
            grads['bc'] += cp.sum(dln_c, axis=1, keepdims=True) / batch_size
            
            # Input gate
            di = dc * c_bar
            da_i = di * i * (1 - i)  # Gradient after sigmoid
            
            # LayerNorm backward for input gate
            dln_i, dgamma_i, dbeta_i = self.layer_norm_backward(da_i, cache_i)
            grads['gamma_i'] += dgamma_i / batch_size
            grads['beta_i'] += dbeta_i / batch_size
            grads['Wi'] += cp.dot(dln_i, z.T) / batch_size
            grads['bi'] += cp.sum(dln_i, axis=1, keepdims=True) / batch_size
            
            # Forget gate
            df = dc * c_prev
            da_f = df * f * (1 - f)  # Gradient after sigmoid
            
            # LayerNorm backward for forget gate
            dln_f, dgamma_f, dbeta_f = self.layer_norm_backward(da_f, cache_f)
            grads['gamma_f'] += dgamma_f / batch_size
            grads['beta_f'] += dbeta_f / batch_size
            grads['Wf'] += cp.dot(dln_f, z.T) / batch_size
            grads['bf'] += cp.sum(dln_f, axis=1, keepdims=True) / batch_size
            
            # Gradients for next timestep
            dz = (cp.dot(self.params['Wf'].T, dln_f) +
                  cp.dot(self.params['Wi'].T, dln_i) +
                  cp.dot(self.params['Wc'].T, dln_c) +
                  cp.dot(self.params['Wo'].T, dln_o))
            
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
        
        ln_total = self.count_parameters()
        increase = (ln_total - standard_total) / standard_total * 100
        
        return {
            'standard_lstm_params': standard_total,
            'layernorm_lstm_params': ln_total,
            'parameters_added': ln_total - standard_total,
            'increase_percentage': increase
        }
