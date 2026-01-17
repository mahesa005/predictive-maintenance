import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        Initialize the LSTM for Tabular Data
        input_size: Number of features per row
        hidden_size: Number of internal memory units (neurons)
        output_size: Dimension of the prediction (1 for binary label)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Combined dimension for gate inputs: [hidden_state_prev, current_input]
        z_dim = hidden_size + input_size
        
        # 1. Parameter Initialization (Weights and Biases)
        self.params = {}
        
        # Gates: Forget (f), Input (i), Candidate (c), Output (o)
        # Using a small scale (0.1) for weight initialization to prevent early saturation
        for gate in ['f', 'i', 'c', 'o']:
            self.params[f'W{gate}'] = np.random.randn(hidden_size, z_dim) * 0.1
            self.params[f'b{gate}'] = np.zeros((hidden_size, 1))
            
        # Output prediction layer (Linear transformation before sigmoid)
        self.params['Wy'] = np.random.randn(output_size, hidden_size) * 0.1
        self.params['by'] = np.zeros((output_size, 1))
        
        # 2. Gradient Accumulators (The "Piggy Bank")
        self.grads = {}
        self.reset_gradients()

    def reset_gradients(self):
        """Zero out all stored gradients."""
        for key in self.params:
            self.grads[f'd{key}'] = np.zeros_like(self.params[key])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(x)

    def forward_step(self, x_t, h_prev, c_prev):
        """One single forward pass for one time step."""
        # Concatenate hidden state and current input
        z = np.row_stack((h_prev, x_t))
        
        # Calculate gate activations
        f_t = self.sigmoid(np.dot(self.params['Wf'], z) + self.params['bf'])
        i_t = self.sigmoid(np.dot(self.params['Wi'], z) + self.params['bi'])
        c_bar = self.tanh(np.dot(self.params['Wc'], z) + self.params['bc'])
        
        # Cell State update (Long-term memory)
        c_t = f_t * c_prev + i_t * c_bar
        
        # Output Gate and Hidden State update (Short-term memory)
        o_t = self.sigmoid(np.dot(self.params['Wo'], z) + self.params['bo'])
        h_t = o_t * self.tanh(c_t)
        
        # Probability prediction
        y_pred = self.sigmoid(np.dot(self.params['Wy'], h_t) + self.params['by'])
        
        # Store values for backprop
        cache = (z, f_t, i_t, c_bar, c_t, o_t, h_t, c_prev, h_prev)
        return h_t, c_t, y_pred, cache

    def backward_step(self, dy, dh_next, dc_next, cache):
        """Backpropagation through one time step."""
        z, f, i, c_bar, c, o, h, c_prev, h_prev = cache
        
        # 1. Output Layer gradients
        self.grads['dWy'] += np.dot(dy, h.T)
        self.grads['dby'] += dy
        
        # 2. Hidden State gradient
        dh = np.dot(self.params['Wy'].T, dy) + dh_next
        
        # 3. Output Gate backprop
        do = dh * self.tanh(c)
        da_o = do * o * (1 - o)
        self.grads['dWo'] += np.dot(da_o, z.T)
        self.grads['dbo'] += da_o
        
        # 4. Cell State backprop (The Gradient Highway)
        dc = dh * o * (1 - self.tanh(c)**2) + dc_next
        
        # 5. Candidate backprop
        dc_bar = dc * i
        da_c = dc_bar * (1 - c_bar**2)
        self.grads['dWc'] += np.dot(da_c, z.T)
        self.grads['dbc'] += da_c
        
        # 6. Input Gate backprop
        di = dc * c_bar
        da_i = di * i * (1 - i)
        self.grads['dWi'] += np.dot(da_i, z.T)
        self.grads['dbi'] += da_i
        
        # 7. Forget Gate backprop
        df = dc * c_prev
        da_f = df * f * (1 - f)
        self.grads['dWf'] += np.dot(da_f, z.T)
        self.grads['dbf'] += da_f
        
        # 8. Calculate gradients for the concatenated input z
        dz = (np.dot(self.params['Wf'].T, da_f) + 
              np.dot(self.params['Wi'].T, da_i) + 
              np.dot(self.params['Wc'].T, da_c) + 
              np.dot(self.params['Wo'].T, da_o))
        
        # Split dz to get the gradient for the previous hidden state
        dh_prev = dz[:self.hidden_size, :]
        dc_prev = f * dc
        
        return dh_prev, dc_prev

    def update_parameters(self, lr):
        """Apply gradients using simple Stochastic Gradient Descent (SGD)."""
        for key in self.params:
            self.params[key] -= lr * self.grads[f'd{key}']

    def train(self, X, y, epochs=10, lr=0.01):
        """
        Main Training Loop
        X: Sequence data (samples, time_steps, features)
        y: Labels (samples,)
        """
        for epoch in range(epochs):
            loss_history = []
            for i in range(len(X)):
                # Initialize states for each sequence
                h_prev = np.zeros((self.hidden_size, 1))
                c_prev = np.zeros((self.hidden_size, 1))
                self.reset_gradients()
                
                caches = []
                x_sequence = X[i]
                y_true = y[i]

                # --- Forward Pass ---
                for t in range(len(x_sequence)):
                    x_t = x_sequence[t].reshape(-1, 1)
                    h_prev, c_prev, y_pred, cache = self.forward_step(x_t, h_prev, c_prev)
                    caches.append(cache)
                
                # --- Loss Calculation (Binary Cross Entropy) ---
                loss = - (y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
                loss_history.append(np.squeeze(loss))
                
                # --- Backward Pass (BPTT) ---
                dy = y_pred - y_true
                dh_next = np.zeros_like(h_prev)
                dc_next = np.zeros_like(c_prev)
                
                for t in reversed(range(len(x_sequence))):
                    # We only calculate the outer error (dy) at the very last step
                    step_dy = dy if t == len(x_sequence) - 1 else 0
                    dh_next, dc_next = self.backward_step(step_dy, dh_next, dc_next, caches[t])
                
                # --- Optimization ---
                self.update_parameters(lr)
            
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {np.mean(loss_history):.4f}")