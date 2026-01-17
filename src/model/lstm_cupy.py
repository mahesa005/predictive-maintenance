"""
CuPy-accelerated LSTM Model for GPU Training.
Drop-in replacement for the NumPy version.
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ CuPy detected - Using GPU acceleration")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("⚠️ CuPy not found - Falling back to NumPy (CPU)")


class LSTMModelGPU:
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        GPU-accelerated LSTM for Tabular Data
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        z_dim = hidden_size + input_size
        
        # Parameter Initialization
        self.params = {}
        for gate in ['f', 'i', 'c', 'o']:
            self.params[f'W{gate}'] = cp.random.randn(hidden_size, z_dim).astype(cp.float32) * 0.1
            self.params[f'b{gate}'] = cp.zeros((hidden_size, 1), dtype=cp.float32)
        
        self.params['Wy'] = cp.random.randn(output_size, hidden_size).astype(cp.float32) * 0.1
        self.params['by'] = cp.zeros((output_size, 1), dtype=cp.float32)
        
        self.grads = {}
        self.reset_gradients()

    def reset_gradients(self):
        for key in self.params:
            self.grads[f'd{key}'] = cp.zeros_like(self.params[key])

    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-cp.clip(x, -500, 500)))

    def tanh(self, x):
        return cp.tanh(x)

    def forward_step(self, x_t, h_prev, c_prev):
        z = cp.row_stack((h_prev, x_t))
        
        f_t = self.sigmoid(cp.dot(self.params['Wf'], z) + self.params['bf'])
        i_t = self.sigmoid(cp.dot(self.params['Wi'], z) + self.params['bi'])
        c_bar = self.tanh(cp.dot(self.params['Wc'], z) + self.params['bc'])
        
        c_t = f_t * c_prev + i_t * c_bar
        
        o_t = self.sigmoid(cp.dot(self.params['Wo'], z) + self.params['bo'])
        h_t = o_t * self.tanh(c_t)
        
        y_pred = self.sigmoid(cp.dot(self.params['Wy'], h_t) + self.params['by'])
        
        cache = (z, f_t, i_t, c_bar, c_t, o_t, h_t, c_prev, h_prev)
        return h_t, c_t, y_pred, cache

    def backward_step(self, dy, dh_next, dc_next, cache):
        z, f, i, c_bar, c, o, h, c_prev, h_prev = cache
        
        self.grads['dWy'] += cp.dot(dy, h.T)
        self.grads['dby'] += dy
        
        dh = cp.dot(self.params['Wy'].T, dy) + dh_next
        
        do = dh * self.tanh(c)
        da_o = do * o * (1 - o)
        self.grads['dWo'] += cp.dot(da_o, z.T)
        self.grads['dbo'] += da_o
        
        dc = dh * o * (1 - self.tanh(c)**2) + dc_next
        
        dc_bar = dc * i
        da_c = dc_bar * (1 - c_bar**2)
        self.grads['dWc'] += cp.dot(da_c, z.T)
        self.grads['dbc'] += da_c
        
        di = dc * c_bar
        da_i = di * i * (1 - i)
        self.grads['dWi'] += cp.dot(da_i, z.T)
        self.grads['dbi'] += da_i
        
        df = dc * c_prev
        da_f = df * f * (1 - f)
        self.grads['dWf'] += cp.dot(da_f, z.T)
        self.grads['dbf'] += da_f
        
        dz = (cp.dot(self.params['Wf'].T, da_f) + 
              cp.dot(self.params['Wi'].T, da_i) + 
              cp.dot(self.params['Wc'].T, da_c) + 
              cp.dot(self.params['Wo'].T, da_o))
        
        dh_prev = dz[:self.hidden_size, :]
        dc_prev = f * dc
        
        return dh_prev, dc_prev

    def update_parameters(self, lr):
        for key in self.params:
            self.params[key] -= lr * self.grads[f'd{key}']

    def train(self, X, y, epochs=10, lr=0.01):
        """
        Main Training Loop (GPU-accelerated)
        X: NumPy array (will be converted to CuPy)
        y: NumPy array (will be converted to CuPy)
        """
        # Convert to GPU arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        
        for epoch in range(epochs):
            loss_history = []
            for i in range(len(X_gpu)):
                h_prev = cp.zeros((self.hidden_size, 1), dtype=cp.float32)
                c_prev = cp.zeros((self.hidden_size, 1), dtype=cp.float32)
                self.reset_gradients()
                
                caches = []
                x_sequence = X_gpu[i]
                y_true = y_gpu[i]

                # Forward Pass
                for t in range(len(x_sequence)):
                    x_t = x_sequence[t].reshape(-1, 1)
                    h_prev, c_prev, y_pred, cache = self.forward_step(x_t, h_prev, c_prev)
                    caches.append(cache)
                
                # Loss (Binary Cross Entropy)
                loss = - (y_true * cp.log(y_pred + 1e-9) + (1 - y_true) * cp.log(1 - y_pred + 1e-9))
                loss_history.append(float(cp.squeeze(loss)))
                
                # Backward Pass (BPTT)
                dy = y_pred - y_true
                dh_next = cp.zeros_like(h_prev)
                dc_next = cp.zeros_like(c_prev)
                
                for t in reversed(range(len(x_sequence))):
                    step_dy = dy if t == len(x_sequence) - 1 else cp.zeros_like(dy)
                    dh_next, dc_next = self.backward_step(step_dy, dh_next, dc_next, caches[t])
                
                self.update_parameters(lr)
            
            # Sync GPU before printing
            if GPU_AVAILABLE:
                cp.cuda.Stream.null.synchronize()
            
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {sum(loss_history)/len(loss_history):.4f}")
    
    def predict(self, X):
        """Generate predictions (returns NumPy array)"""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        predictions = []
        
        for i in range(len(X_gpu)):
            h = cp.zeros((self.hidden_size, 1), dtype=cp.float32)
            c = cp.zeros((self.hidden_size, 1), dtype=cp.float32)
            
            for t in range(len(X_gpu[i])):
                x_t = X_gpu[i][t].reshape(-1, 1)
                h, c, y_pred, _ = self.forward_step(x_t, h, c)
            
            predictions.append(float(cp.squeeze(y_pred)))
        
        return predictions
