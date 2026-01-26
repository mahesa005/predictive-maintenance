"""
Bi-directional LSTM + Attention Model with GPU Optimization.

This variant combines bidirectional processing with global attention:
1. BiLSTM: Processes sequence in both forward and backward directions
2. Attention: Computes weighted context from concatenated hidden states

Key Features:
- Dual Context: Attention operates on concatenated [h_t^f; h_t^b] (2H dimensions)
- W_att size: (2H × 2H) to handle combined hidden states
- Sees importance of each timestep from both past AND future perspectives

References:
- Schuster & Paliwal, "Bidirectional Recurrent Neural Networks" (1997)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_bidirectional import BiLSTMModelGPUOptimized


class BiDirectionalAttentionLSTMModelGPUOptimized(BiLSTMModelGPUOptimized):
    """
    Bidirectional LSTM with Global Attention mechanism for GPU.
    
    Inherits from BiLSTMModelGPUOptimized and adds:
    - __init__: Adds W_att (2H × 2H) and v_att (2H × 1) parameters
    - forward_batch: BiLSTM + attention on concatenated [h_f; h_b] per timestep
    - backward_batch: Combined BiLSTM gradients + attention backprop
    
    The attention mechanism operates on the concatenated hidden states from
    both directions, allowing the model to weigh the importance of each 
    timestep considering both past and future context.
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        # Call parent BiLSTM constructor
        super().__init__(input_size, hidden_size, output_size, cw)
        
        # Attention operates on concatenated hidden states (2 * hidden_size)
        combined_size = 2 * hidden_size
        
        # Xavier/Glorot initialization for attention parameters
        # W_att: (2H, 2H) - transforms concatenated [h_f; h_b] for scoring
        # v_att: (2H, 1) - projects tanh output to scalar score
        scale_att = np.sqrt(2.0 / (combined_size + combined_size))
        
        self.params['W_att'] = cp.random.randn(combined_size, combined_size).astype(cp.float32) * scale_att
        self.params['v_att'] = cp.random.randn(combined_size, 1).astype(cp.float32) * scale_att
        
        # Add Adam optimizer states for attention params
        self.m['W_att'] = cp.zeros_like(self.params['W_att'])
        self.m['v_att'] = cp.zeros_like(self.params['v_att'])
        self.v['W_att'] = cp.zeros_like(self.params['W_att'])
        self.v['v_att'] = cp.zeros_like(self.params['v_att'])
        
        # Store combined size for later use
        self.combined_size = combined_size

    def forward_batch(self, X_batch):
        """
        Batched forward pass with BiLSTM + Attention.
        X_batch: (batch_size, seq_len, input_size)
        
        Process:
        1. Run Forward LSTM (t=0 to T-1) to get h_f for each timestep
        2. Run Backward LSTM (t=T-1 to 0) to get h_b for each timestep
        3. Concatenate [h_f_t; h_b_t] at each timestep -> H_combined
        4. Compute attention on H_combined: e_t = v_att^T * tanh(W_att * [h_f_t; h_b_t])
        5. Context vector: c = Σ α_t * [h_f_t; h_b_t]
        6. Output: ŷ = σ(Wy · c + by)
        
        Returns: predictions (batch_size,), caches, h_combined_final, c_states
        """
        batch_size, seq_len, _ = X_batch.shape
        
        # Initialize hidden states for both directions
        h_f = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c_f = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        h_b = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c_b = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        forward_caches = []
        backward_caches = []
        all_h_forward = []
        all_h_backward = []
        
        # Forward direction: t = 0 to T-1
        for t in range(seq_len):
            x_t = X_batch[:, t, :].T  # (input_size, batch_size)
            h_f, c_f, cache = self._lstm_cell_forward(x_t, h_f, c_f, direction='f')
            forward_caches.append(cache)
            all_h_forward.append(h_f.copy())
        
        # Backward direction: t = T-1 to 0
        for t in range(seq_len - 1, -1, -1):
            x_t = X_batch[:, t, :].T  # (input_size, batch_size)
            h_b, c_b, cache = self._lstm_cell_forward(x_t, h_b, c_b, direction='b')
            backward_caches.append(cache)
            all_h_backward.append(h_b.copy())
        
        # Reverse backward caches and hidden states to align with forward indices
        backward_caches = backward_caches[::-1]
        all_h_backward = all_h_backward[::-1]
        
        # Stack all hidden states: (seq_len, hidden_size, batch_size)
        H_forward = cp.stack(all_h_forward, axis=0)
        H_backward = cp.stack(all_h_backward, axis=0)
        
        # Concatenate forward and backward hidden states at each timestep
        # H_combined: (seq_len, 2*hidden_size, batch_size)
        H_combined = cp.concatenate([H_forward, H_backward], axis=1)
        
        # ========== ATTENTION MECHANISM ==========
        # Attention operates on concatenated [h_f_t; h_b_t]
        # e_t = v_att^T * tanh(W_att * [h_f_t; h_b_t])
        
        # Pre-attention transformation: S_t = tanh(W_att @ h_combined_t)
        # W_att: (2H, 2H), H_combined: (seq_len, 2H, batch_size)
        S = cp.tanh(cp.tensordot(self.params['W_att'], H_combined, axes=([1], [1])))
        # Result: (2H, seq_len, batch_size) -> transpose to (seq_len, 2H, batch_size)
        S = cp.transpose(S, (1, 0, 2))
        
        # Compute scalar scores: e_t = v_att^T @ S_t
        # v_att: (2H, 1), S: (seq_len, 2H, batch_size)
        e = cp.tensordot(self.params['v_att'].flatten(), S, axes=([0], [1]))
        # e: (seq_len, batch_size)
        
        # Softmax over time dimension for attention weights
        e_max = cp.max(e, axis=0, keepdims=True)
        e_exp = cp.exp(e - e_max)
        alpha = e_exp / (cp.sum(e_exp, axis=0, keepdims=True) + 1e-9)
        # alpha: (seq_len, batch_size)
        
        # Compute context vector: c = Σ α_t * [h_f_t; h_b_t]
        # H_combined: (seq_len, 2H, batch_size)
        alpha_expanded = alpha[:, cp.newaxis, :]  # (seq_len, 1, batch_size)
        context = cp.sum(H_combined * alpha_expanded, axis=0)  # (2H, batch_size)
        
        # ========== OUTPUT LAYER ==========
        # Use context vector (already 2H dimension, same as Wy expects)
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], context) + self.params['by'])
        
        # Store attention cache for backprop
        attention_cache = {
            'H_forward': H_forward,      # (seq_len, H, batch_size)
            'H_backward': H_backward,    # (seq_len, H, batch_size)
            'H_combined': H_combined,    # (seq_len, 2H, batch_size)
            'S': S,                      # (seq_len, 2H, batch_size)
            'e': e,                      # (seq_len, batch_size)
            'alpha': alpha,              # (seq_len, batch_size)
            'context': context           # (2H, batch_size)
        }
        
        # Combine all caches
        caches = (forward_caches, backward_caches, attention_cache)
        
        return y_pred.flatten(), caches, context, (c_f, c_b)

    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass with BiLSTM + Attention.
        
        Gradient flow:
        1. dy -> context vector
        2. context -> attention weights (alpha) and H_combined
        3. alpha -> scores (e) -> S -> W_att, v_att
        4. H_combined -> H_forward and H_backward -> separate BiLSTM BPTT
        """
        batch_size = len(y_true)
        forward_caches, backward_caches, attention_cache = caches
        seq_len = len(forward_caches)
        
        # Unpack attention cache
        H_forward = attention_cache['H_forward']
        H_backward = attention_cache['H_backward']
        H_combined = attention_cache['H_combined']
        S = attention_cache['S']
        e = attention_cache['e']
        alpha = attention_cache['alpha']
        context = attention_cache['context']
        
        # Initialize gradients
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # ========== OUTPUT LAYER GRADIENTS ==========
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        # Gradients for output layer (using context vector)
        grads['Wy'] = cp.dot(dy, context.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size
        
        # ========== ATTENTION MECHANISM GRADIENTS ==========
        # d_context: (2H, batch_size)
        d_context = cp.dot(self.params['Wy'].T, dy)
        
        # context = Σ α_t * H_combined_t
        # d_alpha: (seq_len, batch_size)
        d_alpha = cp.sum(d_context[cp.newaxis, :, :] * H_combined, axis=1)
        
        # d_H_combined from attention: (seq_len, 2H, batch_size)
        alpha_expanded = alpha[:, cp.newaxis, :]
        d_H_combined_att = alpha_expanded * d_context[cp.newaxis, :, :]
        
        # ========== SOFTMAX BACKWARD ==========
        sum_alpha_dalpha = cp.sum(alpha * d_alpha, axis=0, keepdims=True)
        d_e = alpha * (d_alpha - sum_alpha_dalpha)  # (seq_len, batch_size)
        
        # ========== ATTENTION SCORE BACKWARD ==========
        # d_v_att: (2H, 1)
        d_v_att = cp.sum(S * d_e[:, cp.newaxis, :], axis=(0, 2), keepdims=False)
        grads['v_att'] = d_v_att.reshape(-1, 1) / batch_size
        
        # d_S: (seq_len, 2H, batch_size)
        d_S = self.params['v_att'] * d_e[:, cp.newaxis, :]
        
        # ========== TANH BACKWARD ==========
        d_pre_S = d_S * (1 - S ** 2)
        
        # d_W_att = Σ_t d_pre_S[t] @ H_combined[t]^T
        d_W_att = cp.zeros_like(self.params['W_att'])
        for t in range(seq_len):
            d_W_att += cp.dot(d_pre_S[t], H_combined[t].T)
        grads['W_att'] = d_W_att / batch_size
        
        # d_H_combined from W_att: W_att^T @ d_pre_S
        d_H_combined_Watt = cp.zeros_like(H_combined)
        for t in range(seq_len):
            d_H_combined_Watt[t] = cp.dot(self.params['W_att'].T, d_pre_S[t])
        
        # Total gradient to combined hidden states
        d_H_combined_total = d_H_combined_att + d_H_combined_Watt
        
        # Split gradients for forward and backward LSTMs
        # d_H_combined_total: (seq_len, 2H, batch_size)
        d_H_forward = d_H_combined_total[:, :self.hidden_size, :]   # (seq_len, H, batch)
        d_H_backward = d_H_combined_total[:, self.hidden_size:, :]  # (seq_len, H, batch)
        
        # ========== FORWARD LSTM BPTT ==========
        dh_f_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        dc_f_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, c, o, h, c_prev, h_prev = forward_caches[t]
            
            # Gradient from attention for this timestep
            dh_from_att = d_H_forward[t]
            dh = dh_f_next + dh_from_att
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo_f'] += cp.dot(da_o, z.T) / batch_size
            grads['bo_f'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # Cell state
            dc = dh * o * (1 - cp.tanh(c) ** 2) + dc_f_next
            
            # Candidate
            dc_bar = dc * i
            da_c = dc_bar * (1 - c_bar ** 2)
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
        
        # ========== BACKWARD LSTM BPTT ==========
        dh_b_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        dc_b_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        # backward_caches is aligned: [0] = t=0, [-1] = t=T-1
        # BPTT processes from t=0 to t=T-1 (following backward LSTM computation order)
        for t in range(seq_len):
            z, f, i, c_bar, c, o, h, c_prev, h_prev = backward_caches[t]
            
            # Gradient from attention for this timestep
            dh_from_att = d_H_backward[t]
            dh = dh_b_next + dh_from_att
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo_b'] += cp.dot(da_o, z.T) / batch_size
            grads['bo_b'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # Cell state
            dc = dh * o * (1 - cp.tanh(c) ** 2) + dc_b_next
            
            # Candidate
            dc_bar = dc * i
            da_c = dc_bar * (1 - c_bar ** 2)
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
        
        _, _, attention_cache = caches
        alpha = attention_cache['alpha']  # (seq_len, batch_size)
        
        if GPU_AVAILABLE:
            return cp.asnumpy(alpha.T)  # (batch_size, seq_len)
        return alpha.T

    def get_bidirectional_hidden_states(self, X_batch):
        """
        Get forward and backward hidden states for analysis.
        
        Args:
            X_batch: Input sequences (batch_size, seq_len, input_size)
            
        Returns:
            H_forward: Forward hidden states (batch_size, seq_len, hidden_size)
            H_backward: Backward hidden states (batch_size, seq_len, hidden_size)
        """
        X_gpu = cp.asarray(X_batch, dtype=cp.float32)
        _, caches, _, _ = self.forward_batch(X_gpu)
        
        _, _, attention_cache = caches
        H_forward = attention_cache['H_forward']    # (seq_len, H, batch)
        H_backward = attention_cache['H_backward']  # (seq_len, H, batch)
        
        # Transpose to (batch, seq_len, H)
        if GPU_AVAILABLE:
            return cp.asnumpy(H_forward.transpose(2, 0, 1)), cp.asnumpy(H_backward.transpose(2, 0, 1))
        return H_forward.transpose(2, 0, 1), H_backward.transpose(2, 0, 1)

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(v.size for v in self.params.values())
    
    def compare_with_variants(self):
        """Compare parameter count with other LSTM variants."""
        z_dim = self.hidden_size + self.input_size
        
        # Standard LSTM
        standard_total = (
            4 * self.hidden_size * z_dim +
            self.output_size * self.hidden_size +
            4 * self.hidden_size +
            self.output_size
        )
        
        # Bidirectional (without attention)
        bilstm_total = (
            2 * 4 * self.hidden_size * z_dim +     # 2 directions × 4 gates
            self.output_size * 2 * self.hidden_size +  # Wy (2H input)
            2 * 4 * self.hidden_size +              # 2 directions × 4 biases
            self.output_size
        )
        
        # BiLSTM + Attention (current model)
        current_total = self.count_parameters()
        
        # Attention overhead
        attention_overhead = (2 * self.hidden_size) ** 2 + 2 * self.hidden_size
        
        return {
            'standard_lstm_params': standard_total,
            'bilstm_only_params': bilstm_total,
            'bilstm_attention_params': current_total,
            'attention_overhead': attention_overhead,
            'increase_vs_bilstm': current_total - bilstm_total,
            'increase_vs_standard': current_total - standard_total
        }
