"""
CIFG + Attention LSTM Model with GPU Optimization.

This variant combines two powerful modifications:
1. CIFG (Coupled Input and Forget Gate): i_t = 1 - f_t (~25% param reduction)
2. Global Attention: Context vector from weighted sum of all hidden states

The combination provides:
- Memory efficiency from CIFG (fewer parameters to store/compute)
- Long-range dependency capturing from Attention (focus on relevant timesteps)

Ideal for long sequences (win=60+) where both efficiency and 
interpretability are important.

References:
- Greff et al., "LSTM: A Search Space Odyssey" (2017)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_cifg import CIFGLSTMModelGPUOptimized


class CIFGAttentionLSTMModelGPUOptimized(CIFGLSTMModelGPUOptimized):
    """
    CIFG LSTM with Global Attention mechanism for GPU.
    
    Inherits from CIFGLSTMModelGPUOptimized and adds:
    - __init__: Adds W_att (H x H) and v_att (H x 1) parameters
    - forward_batch: CIFG gates + collects all h_t + attention context vector
    - backward_batch: Combined CIFG gradient coupling + attention backprop
    
    Benefits:
    - ~25% fewer LSTM parameters (from CIFG)
    - Attention for long-range dependencies and interpretability
    - Efficient for predictive maintenance on long sensor windows
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        # Call parent CIFG constructor (already removes Wi, bi)
        super().__init__(input_size, hidden_size, output_size, cw)
        
        # Xavier/Glorot initialization for attention parameters
        # W_att: (hidden_size, hidden_size) - transforms h_t for scoring
        # v_att: (hidden_size, 1) - projects tanh output to scalar score
        scale_att = np.sqrt(2.0 / (hidden_size + hidden_size))
        
        self.params['W_att'] = cp.random.randn(hidden_size, hidden_size).astype(cp.float32) * scale_att
        self.params['v_att'] = cp.random.randn(hidden_size, 1).astype(cp.float32) * scale_att
        
        # Add Adam optimizer states for attention params
        self.m['W_att'] = cp.zeros_like(self.params['W_att'])
        self.m['v_att'] = cp.zeros_like(self.params['v_att'])
        self.v['W_att'] = cp.zeros_like(self.params['W_att'])
        self.v['v_att'] = cp.zeros_like(self.params['v_att'])

    def forward_batch(self, X_batch):
        """
        Batched forward pass with CIFG + Attention.
        X_batch: (batch_size, seq_len, input_size)
        
        Process:
        1. Run CIFG-LSTM (i_t = 1 - f_t) to get all hidden states h_1, ..., h_T
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
        
        # CIFG-LSTM forward pass
        for t in range(seq_len):
            # x_t: (input_size, batch_size)
            x_t = X_batch[:, t, :].T
            
            # Concatenate h and x: (hidden_size + input_size, batch_size)
            z = cp.vstack((h, x_t))
            
            # Gates (CIFG modification: i = 1 - f)
            f = self.sigmoid(cp.dot(self.params['Wf'], z) + self.params['bf'])
            i = 1 - f  # CIFG: Input gate coupled with forget gate
            c_bar = cp.tanh(cp.dot(self.params['Wc'], z) + self.params['bc'])
            o = self.sigmoid(cp.dot(self.params['Wo'], z) + self.params['bo'])
            
            # Calculate new cell state using CIFG formula
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
        
        # Pre-attention transformation: S_t = tanh(W_att @ h_t)
        S = cp.tanh(cp.tensordot(self.params['W_att'], H, axes=([1], [1])))
        # Result: (hidden_size, seq_len, batch_size)
        S = cp.transpose(S, (1, 0, 2))  # -> (seq_len, hidden_size, batch_size)
        
        # Compute scalar scores: e_t = v_att^T @ S_t
        # e: (seq_len, batch_size)
        e = cp.tensordot(self.params['v_att'].flatten(), S, axes=([0], [1]))
        
        # Softmax over time dimension for attention weights
        e_max = cp.max(e, axis=0, keepdims=True)
        e_exp = cp.exp(e - e_max)
        alpha = e_exp / (cp.sum(e_exp, axis=0, keepdims=True) + 1e-9)
        # alpha: (seq_len, batch_size)
        
        # Compute context vector: c = Σ α_t * h_t
        # H: (seq_len, hidden_size, batch_size)
        alpha_expanded = alpha[:, cp.newaxis, :]  # (seq_len, 1, batch_size)
        context = cp.sum(H * alpha_expanded, axis=0)  # (hidden_size, batch_size)
        
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
        Batched backward pass combining CIFG gradients with Attention mechanism.
        
        Gradient flow:
        1. dy -> context vector
        2. context -> attention weights (alpha) and hidden states (H)
        3. alpha -> scores (e) -> S -> W_att, v_att
        4. H -> all h_t -> CIFG-LSTM backprop (with df_combined = df - di)
        
        Key CIFG modification: Since i_t = 1 - f_t, gradients that would flow
        to input gate are redirected to forget gate with negative sign.
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
        
        # Initialize gradients (no Wi, bi in CIFG)
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
        d_context = cp.dot(self.params['Wy'].T, dy)  # (hidden_size, batch_size)
        
        # context = Σ α_t * h_t
        # d_alpha: (seq_len, batch_size)
        d_alpha = cp.sum(d_context[cp.newaxis, :, :] * H, axis=1)
        
        # d_H from attention: (seq_len, hidden_size, batch_size)
        alpha_expanded = alpha[:, cp.newaxis, :]
        d_H_att = alpha_expanded * d_context[cp.newaxis, :, :]
        
        # ========== SOFTMAX BACKWARD ==========
        # d_e = alpha * (d_alpha - sum_t(alpha_t * d_alpha_t))
        sum_alpha_dalpha = cp.sum(alpha * d_alpha, axis=0, keepdims=True)
        d_e = alpha * (d_alpha - sum_alpha_dalpha)  # (seq_len, batch_size)
        
        # ========== ATTENTION SCORE BACKWARD ==========
        # d_v_att: (hidden_size, 1)
        d_v_att = cp.sum(S * d_e[:, cp.newaxis, :], axis=(0, 2), keepdims=False)
        grads['v_att'] = d_v_att.reshape(-1, 1) / batch_size
        
        # d_S: (seq_len, hidden_size, batch_size)
        d_S = self.params['v_att'] * d_e[:, cp.newaxis, :]
        
        # ========== TANH BACKWARD ==========
        # S = tanh(W_att @ H)
        d_pre_S = d_S * (1 - S ** 2)
        
        # d_W_att = Σ_t d_pre_S[t] @ H[t]^T
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
        
        # ========== CIFG-LSTM BACKWARD (BPTT) ==========
        # Now backprop through CIFG-LSTM with gradients coming from attention
        
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
            dc_bar = dc * i  # Note: i = 1 - f
            da_c = dc_bar * (1 - c_bar ** 2)
            grads['Wc'] += cp.dot(da_c, z.T) / batch_size
            grads['bc'] += cp.sum(da_c, axis=1, keepdims=True) / batch_size
            
            # CIFG: Combined gradient for forget gate
            # df = dc * c_prev (original forget gate gradient)
            # di = dc * c_bar (gradient that would go to input gate)
            # Since i = 1 - f, df/di = -1, so: df_combined = df - di
            df = dc * c_prev
            di = dc * c_bar
            df_combined = df - di  # CIFG gradient coupling
            
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
        """Compare parameter count with standard LSTM and variants."""
        z_dim = self.hidden_size + self.input_size
        
        # Standard LSTM parameters
        standard_total = (
            4 * self.hidden_size * z_dim +  # Wf, Wi, Wc, Wo
            self.output_size * self.hidden_size +  # Wy
            4 * self.hidden_size +  # bf, bi, bc, bo
            self.output_size  # by
        )
        
        # CIFG-only (no attention)
        cifg_total = (
            3 * self.hidden_size * z_dim +  # Wf, Wc, Wo (no Wi)
            self.output_size * self.hidden_size +  # Wy
            3 * self.hidden_size +  # bf, bc, bo (no bi)
            self.output_size  # by
        )
        
        # CIFG + Attention (current model)
        cifg_att_total = self.count_parameters()
        
        # Attention-only overhead
        attention_params = self.hidden_size * self.hidden_size + self.hidden_size
        
        return {
            'standard_lstm_params': standard_total,
            'cifg_only_params': cifg_total,
            'cifg_attention_params': cifg_att_total,
            'attention_overhead': attention_params,
            'net_change_vs_standard': cifg_att_total - standard_total,
            'net_change_percentage': (cifg_att_total - standard_total) / standard_total * 100
        }
