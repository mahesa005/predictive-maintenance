"""
GLU (Gated Linear Unit) LSTM Model with GPU Optimization.

This variant replaces the tanh activation in the candidate cell state with
a Gated Linear Unit (GLU) mechanism for improved gradient flow and stability
on noisy data.

GLU formula: output = A ⊙ σ(B)
where A and B are linear projections of the input.

Reference: Dauphin et al., "Language Modeling with Gated Convolutional Networks" (2017)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_cupy_optimized import LSTMModelGPUOptimized


class GLULSTMModelGPUOptimized(LSTMModelGPUOptimized):
    """
    GLU (Gated Linear Unit) LSTM for GPU.
    
    Inherits from LSTMModelGPUOptimized and modifies:
    - __init__: Adds W_glu and b_glu parameters for GLU gate
    - forward_batch: Replaces tanh(Wc·z + bc) with GLU mechanism
    - backward_batch: Implements GLU gradient computation
    
    GLU provides:
    - Better gradient flow (linear path through A)
    - Learnable activation through gate B
    - Improved stability on noisy data
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        # Call parent constructor first
        super().__init__(input_size, hidden_size, output_size, cw)
        
        z_dim = hidden_size + input_size
        scale = 0.1
        
        # Add GLU gate parameters (W_glu and b_glu)
        # These are used with Wc, bc to form the GLU: A ⊙ σ(B)
        # A = Wc·z + bc (reuse existing candidate weights)
        # B = W_glu·z + b_glu (new gate weights)
        self.params['W_glu'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * scale
        self.params['b_glu'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        
        # Reinitialize Adam optimizer states to include new parameters
        self.m = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in self.params.items()}

    def forward_batch(self, X_batch):
        """
        Batched forward pass with GLU candidate state.
        X_batch: (batch_size, seq_len, input_size)
        
        Key modification: 
        Standard: c̃_t = tanh(Wc·z + bc)
        GLU:      c̃_t = A ⊙ σ(B) where A = Wc·z + bc, B = W_glu·z + b_glu
        
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
            
            # Standard gates
            f = self.sigmoid(cp.dot(self.params['Wf'], z) + self.params['bf'])
            i = self.sigmoid(cp.dot(self.params['Wi'], z) + self.params['bi'])
            o = self.sigmoid(cp.dot(self.params['Wo'], z) + self.params['bo'])
            
            # GLU candidate state: c̃_t = A ⊙ σ(B)
            A = cp.dot(self.params['Wc'], z) + self.params['bc']  # Linear projection
            B = cp.dot(self.params['W_glu'], z) + self.params['b_glu']  # Gate projection
            gate_B = self.sigmoid(B)  # GLU gate
            c_bar = A * gate_B  # GLU output (replaces tanh)
            
            # Calculate new cell and hidden state
            c_new = f * c + i * c_bar
            h_new = o * cp.tanh(c_new)
            
            # Save cache for backprop (include A, B, gate_B for GLU gradients)
            caches.append((z, f, i, c_bar, c_new, o, h_new, c, h, A, B, gate_B))
            
            # Update hidden and cell state
            h, c = h_new, c_new
        
        # Output layer: (output_size, batch_size)
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], h) + self.params['by'])
        
        return y_pred.flatten(), caches, h, c

    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass using BPTT with GLU gradient computation.
        
        GLU backward:
        c̃ = A ⊙ σ(B)
        dc̃/dA = σ(B)
        dc̃/dB = A ⊙ σ(B) ⊙ (1 - σ(B))
        """
        batch_size = len(y_true)
        seq_len = len(caches)
        
        # Initialize gradients (including GLU parameters)
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
            z, f, i, c_bar, c, o, h, c_prev, h_prev, A, B, gate_B = caches[t]
            
            dh = dh_next
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo'] += cp.dot(da_o, z.T) / batch_size 
            grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size

            # Cell state
            dc = dh * o * (1 - cp.tanh(c)**2) + dc_next
            
            # GLU Candidate gradient
            # c̃ = A ⊙ σ(B)
            # dc̃/dA = σ(B) = gate_B
            # dc̃/dB = A ⊙ σ(B) ⊙ (1 - σ(B)) = A ⊙ gate_B ⊙ (1 - gate_B)
            dc_bar = dc * i
            
            # Gradient for A (linear projection)
            dA = dc_bar * gate_B
            grads['Wc'] += cp.dot(dA, z.T) / batch_size
            grads['bc'] += cp.sum(dA, axis=1, keepdims=True) / batch_size
            
            # Gradient for B (GLU gate)
            dB = dc_bar * A * gate_B * (1 - gate_B)
            grads['W_glu'] += cp.dot(dB, z.T) / batch_size
            grads['b_glu'] += cp.sum(dB, axis=1, keepdims=True) / batch_size
            
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
            
            # Gradients for next timestep (include GLU contributions)
            dz = (cp.dot(self.params['Wf'].T, da_f) +
                  cp.dot(self.params['Wi'].T, da_i) +
                  cp.dot(self.params['Wc'].T, dA) +      # A gradient
                  cp.dot(self.params['W_glu'].T, dB) +   # B gradient (GLU)
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
        
        glu_total = self.count_parameters()
        increase = (glu_total - standard_total) / standard_total * 100
        
        return {
            'standard_lstm_params': standard_total,
            'glu_lstm_params': glu_total,
            'parameters_added': glu_total - standard_total,
            'increase_percentage': increase
        }
