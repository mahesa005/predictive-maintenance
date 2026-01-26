"""
Nested LSTM + Attention Model with GPU Optimization.

This is the most complex variant, combining:
1. Nested LSTM: Hierarchical cell state with inner gates (f_inner, i_inner)
2. Global Attention: Context vector from weighted sum of outer hidden states

Architecture:
- Outer LSTM gates: f, i, o, c̃ (standard)
- Inner gates: f_inner, i_inner (nested cell update)
- Attention: operates on outer hidden states h_t

The nested structure provides hierarchical memory dynamics, while attention
enables global timestep selection. This combination is powerful for complex
temporal patterns with multi-scale dependencies.

References:
- Moniz & Krueger, "Nested LSTMs" (2017)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_nested import NestedLSTMModelGPUOptimized


class NestedAttentionLSTMModelGPUOptimized(NestedLSTMModelGPUOptimized):
    """
    Nested LSTM with Global Attention mechanism for GPU.
    
    Inherits from NestedLSTMModelGPUOptimized and adds:
    - __init__: Adds W_att (H × H) and v_att (H × 1) parameters
    - forward_batch: Nested cell update + collects all h_t + attention context
    - backward_batch: Most complex gradient path combining:
        - Attention gradients to all hidden states
        - Nested LSTM gradients through inner and outer gates
        - Multiple paths for c_prev (outer forget, inner forget)
    
    This is the most powerful but also most computationally expensive variant.
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        # Call parent Nested constructor (includes inner gate params)
        super().__init__(input_size, hidden_size, output_size, cw)
        
        # Xavier/Glorot initialization for attention parameters
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
        Batched forward pass with Nested LSTM + Attention.
        X_batch: (batch_size, seq_len, input_size)
        
        Process:
        1. Run Nested LSTM to get all outer hidden states h_1, ..., h_T
           - Outer gates: f, i, o, c̃
           - Inner gates: f_inner, i_inner
           - Hierarchical update: c = f_inner * c_prev + i_inner * tanh(c_temp)
        2. Compute attention on outer hidden states
        3. Output from context vector
        
        Returns: predictions (batch_size,), caches, h_final, c_final
        """
        batch_size, seq_len, _ = X_batch.shape
        
        # Initialize hidden states
        h = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        lstm_caches = []
        all_h = []  # Store all outer hidden states for attention
        
        # Nested LSTM forward pass
        for t in range(seq_len):
            x_t = X_batch[:, t, :].T  # (input_size, batch_size)
            z = cp.vstack((h, x_t))   # (hidden_size + input_size, batch_size)
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
            
            # Save cache for backprop
            lstm_caches.append((z, f, i, c_bar, o, f_inner, i_inner, c_temp, c_new, h_new, c_prev, h))
            
            # Store hidden state for attention
            all_h.append(h_new.copy())
            
            h, c = h_new, c_new
        
        # Stack all hidden states: (seq_len, hidden_size, batch_size)
        H = cp.stack(all_h, axis=0)
        
        # ========== ATTENTION MECHANISM ==========
        # e_t = v_att^T * tanh(W_att * h_t)
        
        # Pre-attention transformation
        S = cp.tanh(cp.tensordot(self.params['W_att'], H, axes=([1], [1])))
        S = cp.transpose(S, (1, 0, 2))  # (seq_len, hidden_size, batch_size)
        
        # Compute scalar scores
        e = cp.tensordot(self.params['v_att'].flatten(), S, axes=([0], [1]))
        # e: (seq_len, batch_size)
        
        # Softmax over time dimension
        e_max = cp.max(e, axis=0, keepdims=True)
        e_exp = cp.exp(e - e_max)
        alpha = e_exp / (cp.sum(e_exp, axis=0, keepdims=True) + 1e-9)
        
        # Context vector
        alpha_expanded = alpha[:, cp.newaxis, :]
        context = cp.sum(H * alpha_expanded, axis=0)  # (hidden_size, batch_size)
        
        # ========== OUTPUT LAYER ==========
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], context) + self.params['by'])
        
        # Store attention cache
        attention_cache = {
            'H': H,
            'S': S,
            'e': e,
            'alpha': alpha,
            'context': context
        }
        
        caches = (lstm_caches, attention_cache)
        
        return y_pred.flatten(), caches, h, c

    def backward_batch(self, y_pred, y_true, caches, cw=1):
        """
        Batched backward pass for Nested LSTM + Attention.
        
        This is the most complex gradient computation with multiple paths:
        
        Gradient Flow Summary:
        =====================
        1. OUTPUT → context
        2. ATTENTION:
           - context → alpha, H
           - alpha → e → S → W_att, v_att
           - H → d_H_total (to be distributed to each h_t)
        
        3. NESTED LSTM (per timestep, with attention gradient injected):
           - h_t receives: dh_next + d_H_total[t]
           - h = o * tanh(c)
             └── do, dc
           
           - c = f_inner * c_prev + i_inner * tanh(c_temp)
             └── df_inner, di_inner, dc_temp
           
           - c_temp = f * c_prev + i * c_bar
             └── df, di, dc_bar
           
           - dc_prev has TWO PATHS:
             a. From outer: dc_temp * f
             b. From inner: dc * f_inner
        """
        batch_size = len(y_true)
        lstm_caches, attention_cache = caches
        seq_len = len(lstm_caches)
        
        # Unpack attention cache
        H = attention_cache['H']
        S = attention_cache['S']
        e = attention_cache['e']
        alpha = attention_cache['alpha']
        context = attention_cache['context']
        
        # Initialize gradients (includes Nested + Attention params)
        grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        
        # ========== OUTPUT LAYER GRADIENTS ==========
        weight_1 = cw
        dy = (y_pred - y_true).reshape(1, -1)
        y_true_reshaped = y_true.reshape(1, -1)
        dy = cp.where(y_true_reshaped == 1, dy * weight_1, dy)
        
        grads['Wy'] = cp.dot(dy, context.T) / batch_size
        grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size
        
        # ========== ATTENTION MECHANISM GRADIENTS ==========
        d_context = cp.dot(self.params['Wy'].T, dy)  # (hidden_size, batch_size)
        
        # context = Σ α_t * h_t
        d_alpha = cp.sum(d_context[cp.newaxis, :, :] * H, axis=1)  # (seq_len, batch_size)
        
        # d_H from attention: (seq_len, hidden_size, batch_size)
        alpha_expanded = alpha[:, cp.newaxis, :]
        d_H_att = alpha_expanded * d_context[cp.newaxis, :, :]
        
        # ========== SOFTMAX BACKWARD ==========
        sum_alpha_dalpha = cp.sum(alpha * d_alpha, axis=0, keepdims=True)
        d_e = alpha * (d_alpha - sum_alpha_dalpha)  # (seq_len, batch_size)
        
        # ========== ATTENTION SCORE BACKWARD ==========
        d_v_att = cp.sum(S * d_e[:, cp.newaxis, :], axis=(0, 2), keepdims=False)
        grads['v_att'] = d_v_att.reshape(-1, 1) / batch_size
        
        d_S = self.params['v_att'] * d_e[:, cp.newaxis, :]
        
        # ========== TANH BACKWARD ==========
        d_pre_S = d_S * (1 - S ** 2)
        
        d_W_att = cp.zeros_like(self.params['W_att'])
        for t in range(seq_len):
            d_W_att += cp.dot(d_pre_S[t], H[t].T)
        grads['W_att'] = d_W_att / batch_size
        
        # d_H from W_att
        d_H_Watt = cp.zeros_like(H)
        for t in range(seq_len):
            d_H_Watt[t] = cp.dot(self.params['W_att'].T, d_pre_S[t])
        
        # Total gradient to hidden states from attention
        d_H_total = d_H_att + d_H_Watt  # (seq_len, hidden_size, batch_size)
        
        # ========== NESTED LSTM BACKWARD (BPTT) ==========
        # This is the most complex part with multiple gradient paths
        
        dh_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        dc_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, o, f_inner, i_inner, c_temp, c, h, c_prev, h_prev = lstm_caches[t]
            
            # Gradient from attention for this timestep
            dh_from_att = d_H_total[t]
            
            # Total gradient to h (from next timestep + attention)
            dh = dh_next + dh_from_att
            
            # ===== OUTPUT GATE (Outer) =====
            # h = o * tanh(c)
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo'] += cp.dot(da_o, z.T) / batch_size
            grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # ===== CELL STATE GRADIENT (starts the complex chain) =====
            # Gradient through h = o * tanh(c)
            dc = dh * o * (1 - cp.tanh(c) ** 2) + dc_next
            
            # ===== INNER GATES BACKPROP =====
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
            
            # Gradient through tanh(c_temp) to c_temp
            # dc_temp = dc * i_inner * (1 - tanh(c_temp)^2)
            dc_temp = dc * i_inner * (1 - tanh_c_temp ** 2)
            
            # ===== OUTER GATES BACKPROP (through c_temp) =====
            # c_temp = f * c_prev + i * c_bar
            
            # Candidate gradient
            dc_bar = dc_temp * i
            da_c = dc_bar * (1 - c_bar ** 2)
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
            
            # ===== GRADIENTS FOR NEXT TIMESTEP =====
            # Gradient through z (ALL gates contribute)
            dz = (cp.dot(self.params['Wf'].T, da_f) +       # Outer forget
                  cp.dot(self.params['Wi'].T, da_i) +       # Outer input
                  cp.dot(self.params['Wc'].T, da_c) +       # Candidate
                  cp.dot(self.params['Wo'].T, da_o) +       # Outer output
                  cp.dot(self.params['Wf_i'].T, da_f_inner) +  # Inner forget
                  cp.dot(self.params['Wi_i'].T, da_i_inner))   # Inner input
            
            dh_next = dz[:self.hidden_size, :]
            
            # ===== CRITICAL: dc_prev has TWO PATHS =====
            # Path 1: Through outer forget gate in c_temp
            #         c_temp = f * c_prev + ...
            #         dc_prev_outer = dc_temp * f
            # Path 2: Through inner forget gate
            #         c = f_inner * c_prev + ...
            #         dc_prev_inner = dc * f_inner
            dc_next = (dc_temp * f +      # From outer forget in c_temp
                       dc * f_inner)      # From inner forget
        
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
        alpha = attention_cache['alpha']
        
        if GPU_AVAILABLE:
            return cp.asnumpy(alpha.T)
        return alpha.T

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
        
        # Nested-only (no attention)
        nested_only_total = (
            4 * self.hidden_size * z_dim +      # Standard gates
            2 * self.hidden_size * z_dim +       # Inner gates (Wf_i, Wi_i)
            self.output_size * self.hidden_size +
            4 * self.hidden_size + 2 * self.hidden_size +  # All biases
            self.output_size
        )
        
        # Nested + Attention (current model)
        current_total = self.count_parameters()
        
        # Overheads
        nested_overhead = 2 * self.hidden_size * z_dim + 2 * self.hidden_size
        attention_overhead = self.hidden_size * self.hidden_size + self.hidden_size
        
        return {
            'standard_lstm_params': standard_total,
            'nested_only_params': nested_only_total,
            'nested_attention_params': current_total,
            'nested_overhead': nested_overhead,
            'attention_overhead': attention_overhead,
            'total_overhead_vs_standard': current_total - standard_total
        }
