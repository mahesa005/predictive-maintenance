import numpy as np
import cupy as cp
import pickle
from ..model.lstm_cupy_optimized import LSTMModelGPUOptimized

def save_model(model, filepath):
    # Konversi params dari CuPy ke NumPy
    cpu_params = {k: cp.asnumpy(v) for k, v in model.params.items()}
    with open(filepath, 'wb') as f:
        pickle.dump(cpu_params, f)
    print(f"✅ Model saved to {filepath}")

def load_and_build_model(filepath):
    # 1. Load from pickle (currently in NumPy/CPU format)
    with open(filepath, 'rb') as f:
        cpu_params = pickle.load(f)
    
    # 2. Extract hidden_size and input_size from the shape of the Wf matrix
    # Wf shape: (hidden_size, hidden_size + input_size)
    h_size = cpu_params['Wf'].shape[0]
    i_size = cpu_params['Wf'].shape[1] - h_size
    
    # 3. Initialize a new model instance with the extracted sizes
    model = LSTMModelGPUOptimized(input_size=i_size, hidden_size=h_size)
    
    # 4. Convert all parameters to CuPy (GPU) and load them into the model
    model.params = {k: cp.asarray(v) for k, v in cpu_params.items()}
    
    print(f"✅ Model built successfully: Input={i_size}, Hidden={h_size}")
    return model


# # Usage:
# model_ready_to_use = load_and_build_model("best_model.pkl")
# # Start prediction immediately!
# results = model_ready_to_use.predict(X_test)