"""
Attention LSTM Model with GPU Optimization.

This variant adds an Attention mechanism on top of the standard LSTM.
Instead of using only the final hidden state h_T for prediction, it computes
a weighted sum (context vector) of all hidden states h_1, h_2, ..., h_T.

The attention weights α_t are computed using:
    e_t = v_att^T * tanh(W_att * h_t)
    α_t = softmax(e) over time dimension

Context vector: c = Σ α_t * h_t

This allows the model to focus on the most relevant timesteps for prediction.

Reference: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_cupy_optimized import LSTMModelGPUOptimized


class AttentionLSTMModelGPUOptimized(LSTMModelGPUOptimized):
    """
    LSTM with Attention mechanism for GPU.
    
    Inherits from LSTMModelGPUOptimized and adds:
    - __init__: Adds W_att (H x H) and v_att (H x 1) parameters
    - forward_batch: Collects all h_t, computes attention scores, context vector
    - backward_batch: Backprops through attention layer to all h_t and attention params
    
    The context vector aggregates information from all timesteps weighted by
    their relevance scores, allowing the model to attend to important events
    in the sequence.
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        # Call parent constructor first
        super().__init__(input_size, hidden_size, output_size, cw)
        
        # Xavier/Glorot initialization for attention parameters
        # W_att: (hidden_size, hidden_size) - transforms h_t for scoring
        # v_att: (hidden_size, 1) - projects tanh output to scalar score
        scale_att = np.sqrt(2.0 / (hidden_size + hidden_size))
        
        self.params['W_att'] = cp.random.randn(hidden_size, hidden_size).astype(cp.float32) * scale_att
        self.params['v_att'] = cp.random.randn(hidden_size, 1).astype(cp.float32) * scale_att
        
        # Re-initialize Wy to take context vector (same size as hidden)
        # Already correct: Wy is (output_size, hidden_size)
        
        # Reinitialize Adam optimizer states to include attention params
        self.m['W_att'] = cp.zeros_like(self.params['W_att'])
        self.m['v_att'] = cp.zeros_like(self.params['v_att'])
        self.v['W_att'] = cp.zeros_like(self.params['W_att'])
        self.v['v_att'] = cp.zeros_like(self.params['v_att'])

    def forward_batch(self, X_batch):
        """
        Batched forward pass with Attention mechanism.
        X_batch: (batch_size, seq_len, input_size)
        
        Process:
        1. Run standard LSTM to get all hidden states h_1, ..., h_T
        2. Compute attention scores: e_t = v_att^T * tanh(W_att * h_t)
        3. Normalize scores with softmax: α_t = softmax(e)
        4. Compute context vector: c = Σ α_t * h_t
        5. Pass context vector to output layer
        
        Returns: predictions (batch_size,), caches, h_final, c_final
        """
        batch_size, seq_len, _ = X_batch.shape
        
        # Initialize hidden states for batch: (hidden_size, batch_size)
        h = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        lstm_caches = []
        all_h = []  # Store all hidden states for attention
        
        # Standard LSTM forward pass
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
            lstm_caches.append((z, f, i, c_bar, c_new, o, h_new, c, h))
            
            # Store hidden state for attention
            all_h.append(h_new.copy())
            
            # Update hidden and cell state
            h, c = h_new, c_new
        
        # Stack all hidden states: (seq_len, hidden_size, batch_size)
        H = cp.stack(all_h, axis=0)
        
        # ========== ATTENTION MECHANISM ==========
        # Compute attention scores for each timestep
        # e_t = v_att^T * tanh(W_att * h_t)
        
        # Pre-attention transformation: (hidden_size, batch_size) for each t
        # S_t = tanh(W_att @ h_t)  -> (seq_len, hidden_size, batch_size)
        S = cp.tanh(cp.tensordot(self.params['W_att'], H, axes=([1], [1])))
        # Result: (hidden_size, seq_len, batch_size)
        S = cp.transpose(S, (1, 0, 2))  # -> (seq_len, hidden_size, batch_size)
        
        # Compute scalar scores: e_t = v_att^T @ S_t
        # v_att: (hidden_size, 1), S: (seq_len, hidden_size, batch_size)
        # e: (seq_len, batch_size)
        e = cp.tensordot(self.params['v_att'].flatten(), S, axes=([0], [1]))
        # Result: (seq_len, batch_size)
        
        # Softmax over time dimension for attention weights
        # α_t = softmax(e_t) for numerical stability, subtract max
        e_max = cp.max(e, axis=0, keepdims=True)
        e_exp = cp.exp(e - e_max)
        alpha = e_exp / (cp.sum(e_exp, axis=0, keepdims=True) + 1e-9)
        # alpha: (seq_len, batch_size)
        
        # Compute context vector: c = Σ α_t * h_t
        # H: (seq_len, hidden_size, batch_size)
        # alpha: (seq_len, batch_size) -> expand to (seq_len, 1, batch_size)
        alpha_expanded = alpha[:, cp.newaxis, :]
        # context: (hidden_size, batch_size)
        context = cp.sum(H * alpha_expanded, axis=0)
        
        # ========== OUTPUT LAYER ==========
        # Use context vector instead of final hidden state
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], context) + self.params['by'])
        
        # Store attention cache for backprop
        attention_cache = {
            'H': H,           # (seq_len, hidden_size, batch_size)
            'S': S,           # (seq_len, hidden_size, batch_size) - tanh outputs
            'e': e,           # (seq_len, batch_size) - raw scores
            'alpha': alpha,   # (seq_len, batch_size) - attention weights
            'context': context  # (hidden_size, batch_size)
        }
        
        # Combine caches
        caches = (lstm_caches, attention_cache)
        
        return y_pred.flatten(), caches, h, c

    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass with Attention mechanism using BPTT.
        
        Gradient flow:
        1. dy -> context vector
        2. context -> attention weights (alpha) and hidden states (H)
        3. alpha -> scores (e) -> S -> W_att, v_att
        4. H -> all h_t -> standard LSTM backprop
        """
        batch_size = len(y_true)
        lstm_caches, attention_cache = caches
        seq_len = len(lstm_caches)
        
        # Unpack attention cache
        H = attention_cache['H']           # (seq_len, hidden_size, batch_size)
        S = attention_cache['S']           # (seq_len, hidden_size, batch_size)
        e = attention_cache['e']           # (seq_len, batch_size)
        alpha = attention_cache['alpha']   # (seq_len, batch_size)
        context = attention_cache['context']  # (hidden_size, batch_size)
        
        # Initialize gradients
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # ========== OUTPUT LAYER GRADIENTS ==========
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        
        # Adjust gradient for class 1 using class weight
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        # Gradients for output layer (using context vector)
        grads['Wy'] = cp.dot(dy, context.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size
        
        # ========== ATTENTION MECHANISM GRADIENTS ==========
        # d_context: gradient w.r.t. context vector
        # context = Wy^T @ dy, shape: (hidden_size, batch_size)
        d_context = cp.dot(self.params['Wy'].T, dy)  # (hidden_size, batch_size)
        
        # context = Σ α_t * h_t
        # d_alpha_t = d_context · h_t (dot product over hidden dim)
        # d_h_t (from attention) = α_t * d_context
        
        # d_alpha: (seq_len, batch_size)
        # For each t: d_alpha[t] = sum over hidden_size of (d_context * H[t])
        d_alpha = cp.sum(d_context[cp.newaxis, :, :] * H, axis=1)  # (seq_len, batch_size)
        
        # d_H from attention: (seq_len, hidden_size, batch_size)
        alpha_expanded = alpha[:, cp.newaxis, :]  # (seq_len, 1, batch_size)
        d_H_att = alpha_expanded * d_context[cp.newaxis, :, :]  # (seq_len, hidden_size, batch_size)
        
        # ========== SOFTMAX BACKWARD ==========
        # alpha = softmax(e)
        # d_e = alpha * (d_alpha - sum_t(alpha_t * d_alpha_t))
        # Softmax jacobian: d_e_i = α_i * (d_alpha_i - Σ_j α_j * d_alpha_j)
        sum_alpha_dalpha = cp.sum(alpha * d_alpha, axis=0, keepdims=True)  # (1, batch_size)
        d_e = alpha * (d_alpha - sum_alpha_dalpha)  # (seq_len, batch_size)
        
        # ========== ATTENTION SCORE BACKWARD ==========
        # e_t = v_att^T @ S_t
        # d_v_att = Σ_t S_t @ d_e_t^T (summed over batch and time)
        # d_S_t = v_att @ d_e_t^T
        
        # d_v_att: (hidden_size, 1)
        # S: (seq_len, hidden_size, batch_size), d_e: (seq_len, batch_size)
        # Sum over seq_len and batch_size
        d_v_att = cp.sum(S * d_e[:, cp.newaxis, :], axis=(0, 2), keepdims=False)
        grads['v_att'] = d_v_att.reshape(-1, 1) / batch_size
        
        # d_S: (seq_len, hidden_size, batch_size)
        # v_att: (hidden_size, 1), d_e: (seq_len, batch_size)
        d_S = self.params['v_att'] * d_e[:, cp.newaxis, :]  # broadcast: (seq_len, hidden_size, batch_size)
        
        # ========== TANH BACKWARD ==========
        # S = tanh(W_att @ H)
        # d_pre_S = d_S * (1 - S^2)
        d_pre_S = d_S * (1 - S ** 2)  # (seq_len, hidden_size, batch_size)
        
        # d_W_att = Σ_t d_pre_S[t] @ H[t]^T
        # H: (seq_len, hidden_size, batch_size)
        # d_pre_S: (seq_len, hidden_size, batch_size)
        # For each t: d_W_att += d_pre_S[t] @ H[t].T
        # Summing over seq_len: matmul with transpose
        d_W_att = cp.zeros_like(self.params['W_att'])
        for t in range(seq_len):
            d_W_att += cp.dot(d_pre_S[t], H[t].T)
        grads['W_att'] = d_W_att / batch_size
        
        # d_H from W_att: W_att^T @ d_pre_S
        d_H_Watt = cp.zeros_like(H)
        for t in range(seq_len):
            d_H_Watt[t] = cp.dot(self.params['W_att'].T, d_pre_S[t])
        
        # Total gradient to hidden states from attention
        d_H_total = d_H_att + d_H_Watt  # (seq_len, hidden_size, batch_size)
        
        # ========== LSTM BACKWARD (BPTT) ==========
        # Now backprop through LSTM with gradients coming from attention
        
        dh_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        dc_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, c, o, h, c_prev, h_prev = lstm_caches[t]
            
            # Gradient from attention mechanism for this timestep
            dh_from_att = d_H_total[t]  # (hidden_size, batch_size)
            
            # Total gradient to h_t
            dh = dh_next + dh_from_att
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo'] += cp.dot(da_o, z.T) / batch_size
            grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # Cell state
            dc = dh * o * (1 - cp.tanh(c) ** 2) + dc_next
            
            # Candidate
            dc_bar = dc * i
            da_c = dc_bar * (1 - c_bar ** 2)
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
            grads[k] = cp.clip(grads[k], -5, 5)
        
        return grads

    def get_attention_weights(self, X_batch):
        """
        Get attention weights for visualization.
        
        Args:
            X_batch: Input sequences (batch_size, seq_len, input_size)
            
        Returns:
            alpha: Attention weights (batch_size, seq_len)
        """
        X_gpu = cp.asarray(X_batch, dtype=cp.float32)
        _, caches, _, _ = self.forward_batch(X_gpu)
        
        _, attention_cache = caches
        alpha = attention_cache['alpha']  # (seq_len, batch_size)
        
        if GPU_AVAILABLE:
            return cp.asnumpy(alpha.T)  # (batch_size, seq_len)
        return alpha.T
    
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
        
        attention_total = self.count_parameters()
        additional = attention_total - standard_total
        increase = additional / standard_total * 100
        
        return {
            'standard_lstm_params': standard_total,
            'attention_lstm_params': attention_total,
            'additional_params': additional,
            'increase_percentage': increase
        }
