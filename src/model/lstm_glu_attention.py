"""
GLU + Attention LSTM Model with GPU Optimization.

This variant combines two powerful mechanisms:
1. GLU (Gated Linear Unit): c̃_t = A ⊙ σ(B) for stable gradient flow
2. Global Attention: Context vector from weighted sum of all hidden states

Benefits:
- GLU stabilizes information flow, especially on noisy sensor data
- Attention selects the most relevant moments in the sequence
- Combined: Robust feature extraction + intelligent timestep selection

References:
- Dauphin et al., "Language Modeling with Gated Convolutional Networks" (2017)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

import numpy as np
from .lstm_glu import GLULSTMModelGPUOptimized


class GLUAttentionLSTMModelGPUOptimized(GLULSTMModelGPUOptimized):
    """
    GLU LSTM with Global Attention mechanism for GPU.
    
    Inherits from GLULSTMModelGPUOptimized and adds:
    - __init__: Adds W_att (H × H) and v_att (H × 1) parameters
    - forward_batch: GLU candidate + collects all h_t + attention context
    - backward_batch: Combined GLU + Attention gradient computation
    
    Ideal for:
    - Noisy sensor data (GLU provides stability)
    - Long sequences where specific moments matter (Attention provides focus)
    - Predictive maintenance with high-noise industrial sensor data
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, cw=1):
        # Call parent GLU constructor (includes W_glu, b_glu)
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
        Batched forward pass with GLU + Attention.
        X_batch: (batch_size, seq_len, input_size)
        
        Process:
        1. Run GLU-LSTM (c̃_t = A ⊙ σ(B)) to get all hidden states
        2. Compute attention scores: e_t = v_att^T * tanh(W_att * h_t)
        3. Normalize with softmax: α_t = softmax(e)
        4. Compute context vector: c = Σ α_t * h_t
        5. Output: ŷ = σ(Wy · c + by)
        
        Returns: predictions (batch_size,), caches, h_final, c_final
        """
        batch_size, seq_len, _ = X_batch.shape
        
        # Initialize hidden states for batch
        h = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        c = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        lstm_caches = []
        all_h = []  # Store all hidden states for attention
        
        # GLU-LSTM forward pass
        for t in range(seq_len):
            x_t = X_batch[:, t, :].T  # (input_size, batch_size)
            z = cp.vstack((h, x_t))   # (hidden_size + input_size, batch_size)
            
            # Standard gates
            f = self.sigmoid(cp.dot(self.params['Wf'], z) + self.params['bf'])
            i = self.sigmoid(cp.dot(self.params['Wi'], z) + self.params['bi'])
            o = self.sigmoid(cp.dot(self.params['Wo'], z) + self.params['bo'])
            
            # GLU candidate state: c̃_t = A ⊙ σ(B)
            A = cp.dot(self.params['Wc'], z) + self.params['bc']       # Linear projection
            B = cp.dot(self.params['W_glu'], z) + self.params['b_glu'] # Gate projection
            gate_B = self.sigmoid(B)                                    # GLU gate
            c_bar = A * gate_B                                          # GLU output
            
            # Cell and hidden state update
            c_new = f * c + i * c_bar
            h_new = o * cp.tanh(c_new)
            
            # Save cache for backprop (include GLU components)
            lstm_caches.append((z, f, i, c_bar, c_new, o, h_new, c, h, A, B, gate_B))
            
            # Store hidden state for attention
            all_h.append(h_new.copy())
            
            h, c = h_new, c_new
        
        # Stack all hidden states: (seq_len, hidden_size, batch_size)
        H = cp.stack(all_h, axis=0)
        
        # ========== ATTENTION MECHANISM ==========
        # e_t = v_att^T * tanh(W_att * h_t)
        
        # Pre-attention transformation: S_t = tanh(W_att @ h_t)
        S = cp.tanh(cp.tensordot(self.params['W_att'], H, axes=([1], [1])))
        S = cp.transpose(S, (1, 0, 2))  # -> (seq_len, hidden_size, batch_size)
        
        # Compute scalar scores: e_t = v_att^T @ S_t
        e = cp.tensordot(self.params['v_att'].flatten(), S, axes=([0], [1]))
        # e: (seq_len, batch_size)
        
        # Softmax over time dimension
        e_max = cp.max(e, axis=0, keepdims=True)
        e_exp = cp.exp(e - e_max)
        alpha = e_exp / (cp.sum(e_exp, axis=0, keepdims=True) + 1e-9)
        
        # Context vector: c = Σ α_t * h_t
        alpha_expanded = alpha[:, cp.newaxis, :]  # (seq_len, 1, batch_size)
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
        Batched backward pass combining GLU and Attention gradients.
        
        Gradient flow:
        1. dy -> context vector
        2. context -> attention weights (alpha) and hidden states (H)
        3. alpha -> scores (e) -> S -> W_att, v_att
        4. H -> all h_t -> GLU-LSTM backprop with:
           - GLU gradient: dA = dc̃ ⊙ σ(B), dB = dc̃ ⊙ A ⊙ σ(B) ⊙ (1-σ(B))
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
        
        # Initialize gradients (including GLU and Attention params)
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
        
        # d_H from attention
        alpha_expanded = alpha[:, cp.newaxis, :]
        d_H_att = alpha_expanded * d_context[cp.newaxis, :, :]  # (seq_len, H, batch)
        
        # ========== SOFTMAX BACKWARD ==========
        sum_alpha_dalpha = cp.sum(alpha * d_alpha, axis=0, keepdims=True)
        d_e = alpha * (d_alpha - sum_alpha_dalpha)  # (seq_len, batch_size)
        
        # ========== ATTENTION SCORE BACKWARD ==========
        d_v_att = cp.sum(S * d_e[:, cp.newaxis, :], axis=(0, 2), keepdims=False)
        grads['v_att'] = d_v_att.reshape(-1, 1) / batch_size
        
        d_S = self.params['v_att'] * d_e[:, cp.newaxis, :]  # (seq_len, H, batch)
        
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
        
        # ========== GLU-LSTM BACKWARD (BPTT) ==========
        dh_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        dc_next = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
        
        for t in reversed(range(seq_len)):
            z, f, i, c_bar, c, o, h, c_prev, h_prev, A, B, gate_B = lstm_caches[t]
            
            # Gradient from attention for this timestep
            dh_from_att = d_H_total[t]
            dh = dh_next + dh_from_att
            
            # Output gate
            do = dh * cp.tanh(c)
            da_o = do * o * (1 - o)
            grads['Wo'] += cp.dot(da_o, z.T) / batch_size
            grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size
            
            # Cell state
            dc = dh * o * (1 - cp.tanh(c) ** 2) + dc_next
            
            # ===== GLU Candidate Gradient =====
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
                  cp.dot(self.params['Wc'].T, dA) +       # A gradient (GLU)
                  cp.dot(self.params['W_glu'].T, dB) +    # B gradient (GLU)
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
        
        # GLU-only (no attention)
        glu_only_total = (
            4 * self.hidden_size * z_dim +      # Standard gates
            self.hidden_size * z_dim +           # W_glu
            self.output_size * self.hidden_size +
            4 * self.hidden_size + self.hidden_size +  # biases including b_glu
            self.output_size
        )
        
        # GLU + Attention (current model)
        current_total = self.count_parameters()
        
        # Attention overhead
        attention_overhead = self.hidden_size * self.hidden_size + self.hidden_size
        
        return {
            'standard_lstm_params': standard_total,
            'glu_only_params': glu_only_total,
            'glu_attention_params': current_total,
            'glu_overhead': glu_only_total - standard_total,
            'attention_overhead': attention_overhead,
            'total_overhead_vs_standard': current_total - standard_total
        }
