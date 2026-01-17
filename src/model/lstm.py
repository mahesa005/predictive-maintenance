import numpy as np

class LSTMModel:
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        Initialize the LSTM for Tabular Data
        input_size: Number of features in your dataset (e.g., SLA, Duration, etc.)
        hidden_size: Number of internal memory units
        output_size: Dimension of prediction (1 for binary classification)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Combined dimension for the input to gates: (hidden_state + current_input)
        # In your case, if hidden_size is 64 and input_size is 5, z_dim is 69.
        z_dim = hidden_size + input_size
        
        # 1. Parameter Initialization (Weights and Biases)
        self.params = {}
        
        # Gates: Forget (f), Input (i), Candidate (c), Output (o)
        for gate in ['f', 'i', 'c', 'o']:
            # Weight shape: (hidden_size, hidden_size + input_size)
            self.params[f'W{gate}'] = np.random.randn(hidden_size, z_dim) * 0.1
            self.params[f'b{gate}'] = np.zeros((hidden_size, 1))
            
        # Prediction layer (to map hidden state to the final label 0 or 1)
        self.params['Wy'] = np.random.randn(output_size, hidden_size) * 0.1
        self.params['by'] = np.zeros((output_size, 1))
        
        # 2. Gradient Accumulators (The "Piggy Bank" for backprop)
        self.grads = {}
        self.reset_gradients()

    def reset_gradients(self):
        """Initialize or reset gradients to zero at the start of each iteration."""
        for key in self.params:
            self.grads[f'd{key}'] = np.zeros_like(self.params[key])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward_step(self, x_t, h_prev, c_prev):
        """
        One single forward pass for one time step (one row of data)
        x_t: Feature vector of the current row (shape: input_size x 1)
        """
        # Concatenate hidden state and current input: [h_prev, x_t]
        z = np.row_stack((h_prev, x_t))
        
        # Calculate gate activations
        f_t = self.sigmoid(np.dot(self.params['Wf'], z) + self.params['bf'])
        i_t = self.sigmoid(np.dot(self.params['Wi'], z) + self.params['bi'])
        c_bar = self.tanh(np.dot(self.params['Wc'], z) + self.params['bc'])
        
        # Update Cell State (Long-term Memory)
        c_t = f_t * c_prev + i_t * c_bar
        
        # Calculate Output Gate and Hidden State (Short-term Memory)
        o_t = self.sigmoid(np.dot(self.params['Wo'], z) + self.params['bo'])
        h_t = o_t * self.tanh(c_t)
        
        # Final Prediction (for this step)
        y_pred = self.sigmoid(np.dot(self.params['Wy'], h_t) + self.params['by'])
        
        # Save values for backward pass (Cache)
        cache = (z, f_t, i_t, c_bar, c_t, o_t, h_t, c_prev, h_prev)
        return h_t, c_t, y_pred, cache

    def backward_step(self, dy, dh_next, dc_next, cache):
        """
        One single backward pass for one time step
        dy: Error from the current prediction (y_pred - y_actual)
        dh_next: Gradient coming back from the next time step (h_t+1)
        dc_next: Gradient coming back from the next cell state (c_t+1)
        """
        z, f, i, c_bar, c, o, h, c_prev, h_prev = cache
        
        # 1. Backprop through the output prediction layer
        self.grads['dWy'] += np.dot(dy, h.T)
        self.grads['dby'] += dy
        
        # 2. Gradient of Hidden State (h)
        dh = np.dot(self.params['Wy'].T, dy) + dh_next
        
        # 3. Backprop through Output Gate
        do = dh * self.tanh(c)
        da_o = do * o * (1 - o) # derivative of sigmoid
        self.grads['dWo'] += np.dot(da_o, z.T)
        self.grads['dbo'] += da_o
        
        # 4. Backprop through Cell State (c)
        dc = dh * o * (1 - self.tanh(c)**2) + dc_next
        
        # 5. Backprop through Candidate (c_bar)
        dc_bar = dc * i
        da_c = dc_bar * (1 - c_bar**2) # derivative of tanh
        self.grads['dWc'] += np.dot(da_c, z.T)
        self.grads['dbc'] += da_c
        
        # 6. Backprop through Input Gate
        di = dc * c_bar
        da_i = di * i * (1 - i)
        self.grads['dWi'] += np.dot(da_i, z.T)
        self.grads['dbi'] += da_i
        
        # 7. Backprop through Forget Gate
        df = dc * c_prev
        da_f = df * f * (1 - f)
        self.grads['dWf'] += np.dot(da_f, z.T)
        self.grads['dbf'] += da_f
        
        # 8. Pass gradient to previous time step (h_t-1 and c_t-1)
        dz = (np.dot(self.params['Wf'].T, da_f) + 
              np.dot(self.params['Wi'].T, da_i) + 
              np.dot(self.params['Wc'].T, da_c) + 
              np.dot(self.params['Wo'].T, da_o))
        
        dh_prev = dz[:self.hidden_size, :]
        dc_prev = f * dc
        
        return dh_prev, dc_prev

    def update_parameters(self, learning_rate):
        """Standard Gradient Descent update."""
        for key in self.params:
            self.params[key] -= learning_rate * self.grads[f'd{key}']