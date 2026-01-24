"""
Peephole LSTM Model with GPU Optimization (Adam Optimizer).
Extends LSTMModelGPUOptimized with peephole connections for improved memory capacity.

Peephole Connections:
- V_f, V_i: Use c_{prev} (cell state from previous timestep)
- V_o: Uses c_t (current cell state)

Mathematical formulation:
    f_t = σ(W_f z + V_f ⊙ c_{prev} + b_f)
    i_t = σ(W_i z + V_i ⊙ c_{prev} + b_i)
    c_t = f_t ⊙ c_{prev} + i_t ⊙ tanh(W_c z + b_c)
    o_t = σ(W_o z + V_o ⊙ c_t + b_o)
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


class PeepholeLSTMModelGPUOptimized(LSTMModelGPUOptimized):
    """
    Peephole LSTM with Adam optimizer and GPU acceleration.
    
    Peephole connections allow gates to directly observe the cell state,
    improving the model's ability to learn long-term dependencies.
    
    Inherits from LSTMModelGPUOptimized:
    - sigmoid(), evaluate(), train(), predict() methods
    - Base parameters (W, b) initialization
    
    Overrides:
    - __init__(): Adds peephole parameters (Vf, Vi, Vo)
    - forward_batch(): Adds peephole connections to gates
    - backward_batch(): Computes gradients for peephole parameters
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        """
        Initialize Peephole LSTM with Adam optimizer.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units (default 1 for binary classification)
            cw: Class weight for positive class (default 1)
        """
        # Initialize parent class (sets up W, b parameters and Adam states)
        super().__init__(input_size, hidden_size, output_size, cw)
        
        # Add peephole connection parameters
        # Small scale (0.01) to prevent peephole from dominating gates initially
        peephole_scale = 0.01
        self.params['Vf'] = cp.random.randn(hidden_size, 1).astype(cp.float32) * peephole_scale
        self.params['Vi'] = cp.random.randn(hidden_size, 1).astype(cp.float32) * peephole_scale
        self.params['Vo'] = cp.random.randn(hidden_size, 1).astype(cp.float32) * peephole_scale
        
        # Add Adam optimizer states for peephole parameters
        self.m['Vf'] = cp.zeros_like(self.params['Vf'])
        self.m['Vi'] = cp.zeros_like(self.params['Vi'])
        self.m['Vo'] = cp.zeros_like(self.params['Vo'])
        self.v['Vf'] = cp.zeros_like(self.params['Vf'])
        self.v['Vi'] = cp.zeros_like(self.params['Vi'])
        self.v['Vo'] = cp.zeros_like(self.params['Vo'])
    
    def forward_batch(self, X_batch):
        """
        Batched forward pass with peephole connections.
        
        Args:
            X_batch: Input batch (batch_size, seq_len, input_size)
            
        Returns:
            y_pred: Predictions (batch_size,)
            caches: List of cache tuples for backprop
            h: Final hidden state
            c: Final cell state
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
            
            # Store c_prev for gradient computation
            c_prev = c
            
            # Gates with peephole connections
            # f_t: forget gate with peephole from c_prev
            # V_f shape: (hidden_size, 1), c_prev shape: (hidden_size, batch_size)
            # Broadcasting: V_f * c_prev -> (hidden_size, batch_size)
            f = self.sigmoid(
                cp.dot(self.params['Wf'], z) + 
                self.params['Vf'] * c_prev +  # Peephole from c_prev
                self.params['bf']
            )
            
            # i_t: input gate with peephole from c_prev
            i = self.sigmoid(
                cp.dot(self.params['Wi'], z) + 
                self.params['Vi'] * c_prev +  # Peephole from c_prev
                self.params['bi']
            )
            
            # c_bar: candidate cell state (no peephole)
            c_bar = cp.tanh(cp.dot(self.params['Wc'], z) + self.params['bc'])
            
            # New cell state
            c_new = f * c_prev + i * c_bar
            
            # o_t: output gate with peephole from c_t (current cell state!)
            o = self.sigmoid(
                cp.dot(self.params['Wo'], z) + 
                self.params['Vo'] * c_new +  # Peephole from c_t (current!)
                self.params['bo']
            )
            
            # New hidden state
            h_new = o * cp.tanh(c_new)
            
            # Save cache for backprop (includes c_prev for peephole gradients)
            caches.append((z, f, i, c_bar, c_new, o, h_new, c_prev, h))
            
            # Update states
            h, c = h_new, c_new
        
        # Output layer
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], h) + self.params['by'])
        
        return y_pred.flatten(), caches, h, c
    
    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass with peephole gradient computation.
        
        Gradient derivation for peephole connections:
        
        For V_o (uses c_t):
            dV_o = Σ_t (da_o * c_t)
            where da_o = do * o * (1 - o)
        
        For V_f and V_i (use c_prev):
            dV_f = Σ_t (da_f * c_prev)
            dV_i = Σ_t (da_i * c_prev)
        
        Note: V gradients are summed across batch and divided by batch_size.
        Since V shape is (hidden_size, 1), we sum over the batch dimension.
        
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
        
        # Initialize gradients (includes V parameters)
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # Output gradient with class weighting
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        # Get final hidden state from last cache
        h_final = caches[-1][6]  # h_new from last timestep
        
        # Output layer gradients
        grads['Wy'] = cp.dot(dy, h_final.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size
        
        # Backprop through hidden state
        dh_next = cp.dot(self.params['Wy'].T, dy)
        dc_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, c, o, h, c_prev, h_prev = caches[t]
            
            dh = dh_next
            
            # ===== Output Gate =====
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo'] += cp.dot(da_o, z.T) / batch_size
            grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # Peephole gradient for V_o: uses c_t (current cell state)
            # da_o shape: (hidden_size, batch_size)
            # c shape: (hidden_size, batch_size)
            # dV_o = sum over batch of (da_o * c) -> (hidden_size, 1)
            grads['Vo'] += cp.sum(da_o * c, axis=1, keepdims=True) / batch_size
            
            # ===== Cell State =====
            # Gradient through h = o * tanh(c)
            dc = dh * o * (1 - cp.tanh(c)**2) + dc_next
            
            # Additional gradient through output gate peephole: o depends on c_t
            # o = sigmoid(W_o z + V_o * c + b_o)
            # So dc has additional term: da_o * V_o
            dc += da_o * self.params['Vo']
            
            # ===== Candidate Cell =====
            dc_bar = dc * i
            da_c = dc_bar * (1 - c_bar**2)
            grads['Wc'] += cp.dot(da_c, z.T) / batch_size
            grads['bc'] += cp.sum(da_c, axis=1, keepdims=True) / batch_size
            
            # ===== Input Gate =====
            di = dc * c_bar
            da_i = di * i * (1 - i)
            grads['Wi'] += cp.dot(da_i, z.T) / batch_size
            grads['bi'] += cp.sum(da_i, axis=1, keepdims=True) / batch_size
            
            # Peephole gradient for V_i: uses c_prev
            grads['Vi'] += cp.sum(da_i * c_prev, axis=1, keepdims=True) / batch_size
            
            # ===== Forget Gate =====
            df = dc * c_prev
            da_f = df * f * (1 - f)
            grads['Wf'] += cp.dot(da_f, z.T) / batch_size
            grads['bf'] += cp.sum(da_f, axis=1, keepdims=True) / batch_size
            
            # Peephole gradient for V_f: uses c_prev
            grads['Vf'] += cp.sum(da_f * c_prev, axis=1, keepdims=True) / batch_size
            
            # ===== Gradients for next timestep =====
            dz = (cp.dot(self.params['Wf'].T, da_f) +
                  cp.dot(self.params['Wi'].T, da_i) +
                  cp.dot(self.params['Wc'].T, da_c) +
                  cp.dot(self.params['Wo'].T, da_o))
            
            dh_next = dz[:self.hidden_size, :]
            
            # dc_next includes peephole contributions from f and i gates
            # f = sigmoid(W_f z + V_f * c_prev + b_f)
            # i = sigmoid(W_i z + V_i * c_prev + b_i)
            dc_next = f * dc + da_f * self.params['Vf'] + da_i * self.params['Vi']
        
        # Gradient clipping
        for k in grads:
            grads[k] = cp.clip(grads[k], -5, 5)
        
        return grads
    
    def update_adam(self, grads, lr=0.001, beta1=0.9, beta2=0.999, 
                    eps=1e-8, lambda_l2=0.01):
        """
        Adam optimizer update (overridden to include L2 for V parameters).
        
        Updates all parameters including peephole weights (Vf, Vi, Vo).
        """
        self.t += 1
        for k in self.params:
            # Add L2 regularization (for W and V parameters)
            actual_grad = grads[k]
            if k.startswith('W') or k.startswith('V'):
                actual_grad = actual_grad + lambda_l2 * self.params[k]
            
            # Adam update
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * actual_grad
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (actual_grad ** 2)
            
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            
            self.params[k] -= lr * m_hat / (cp.sqrt(v_hat) + eps)
