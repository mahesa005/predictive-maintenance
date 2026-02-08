# LSTM Optimization Report: Predictive Maintenance for IT Infrastructure

**Project:** Infomedia Predictive Maintenance
**Date:** February 8, 2026
**Author:** Data Science Team
**Version:** 1.0

---

## 1. Executive Summary

### Champion Model Configuration
| Metric | Value |
|--------|-------|
| **Best F1-Score** | **82.94%** |
| **ROC-AUC** | 90.94% |
| **PR-AUC** | 92.63% |
| **Architecture** | Nested LSTM |
| **Features** | f19-ultimate (19 features) |
| **Window Size** | 48 timesteps (24-hour lookback) |
| **Sampling Period** | 30 minutes |
| **Optimal Threshold** | 0.402 |

### Key Findings

1. **Nested LSTM outperforms all other architectures** - The hierarchical cell state mechanism with dual memory paths provides superior gradient flow for 24-hour lookback windows.

2. **Feature engineering contributes marginal improvement (~1% F1)** - With optimal configuration (win48, 30min, SMOTE, Attention), even baseline f2 achieves 81.97%. Moving to f19 improves to 82.94% (+0.97%).

3. **30-minute sampling with 48-timestep window is optimal** - This configuration captures exactly 24 hours of operational context, aligning with daily IT operations cycles.

4. **SMOTE is essential for class balance** - Synthetic oversampling consistently outperformed class weighting and undersampling approaches.

5. **Optimal configuration enables high baseline performance** - With win48, 30min, SMOTE, even f2 baseline achieves 81.97% F1 (Attention) - additional features provide only ~1% improvement.

---

## 2. Introduction

### 2.1 Problem Background

IT infrastructure incidents at Infomedia require rapid response to minimize service disruption. Traditional reactive monitoring waits for incidents to occur before alerting, resulting in:
- Extended mean-time-to-resolution (MTTR)
- Cascading failures from undetected early warning signs
- Suboptimal resource allocation during incident response

### 2.2 Current Baseline Limitations

Prior approaches using rule-based thresholds on Priority and Impact scores achieved:
- High false positive rates (~40%)
- Inability to capture temporal patterns
- No consideration of service context or historical trends

### 2.3 Objective

Develop an LSTM-based predictive maintenance system that:
- Achieves F1-Score > 80% (balanced precision/recall)
- Provides 30-minute advance warning of incidents
- Operates on streaming ticket data with minimal latency

---

## 3. Methodology & Scope

### 3.1 Problem Formulation

**Task:** Supervised Time-Series Binary Classification
**Input:** Sliding window of ticket features over past 24 hours
**Output:** Probability of incident in next 30-minute interval
**Target Definition:** `incident = (Priority >= 2) AND (Impact >= 2)`

### 3.2 Data Balancing Approach

**Selected Method:** SMOTE (Synthetic Minority Over-sampling Technique)

| Method | F1-Score | Rationale |
|--------|----------|-----------|
| **SMOTE** | **82.94%** | Best performance; preserves majority class information |
| Class Weighting | 78.50% | Gradient instability on imbalanced batches |
| Undersampling | 75.46% | Loses valuable majority class patterns |

SMOTE was selected because:
1. Generates synthetic minority samples in feature space
2. Preserves all majority class information
3. Works well with temporal windowing (applied post-windowing)

### 3.3 Validation Strategy

**Selected Method:** Train-Validation-Test Split (60/20/20)

| Strategy | Pros | Cons | Selected |
|----------|------|------|----------|
| **Single Split** | Fast iteration, deterministic | May overfit to split | Yes |
| K-Fold CV | More robust estimate | 5x training time | Validation only |

**Rationale for Single Split:**
- Dataset has 14,592 samples after windowing (sufficient for reliable split)
- K-Fold validation showed only ~0.5% F1 reduction (acceptable generalization gap)
- Single split enables faster experimentation across 116 model configurations

**K-Fold Validation Results:**
| Model | Single Split F1 | 5-Fold Mean F1 | Gap |
|-------|-----------------|----------------|-----|
| f13-SMOTE-Attention | 82.53% | 81.98% | -0.55% |
| f13-SMOTE-BiDir-Attention | 81.53% | 81.07% | -0.46% |

---

## 4. Data Pipeline & Feature Engineering

### 4.1 Data Flow

```
Raw Tickets (18,969 records)
        ↓
    Filtering (Type = 'restoration')
        ↓
    Feature Engineering (19 features)
        ↓
    Resampling (30-min aggregation → 14,640 intervals)
        ↓
    Windowing (48 timesteps → 14,592 samples)
        ↓
    Train/Val/Test Split
        ↓
    SMOTE (training set only)
        ↓
    LSTM Training
```

### 4.2 Feature Package Evolution

#### **f2-baseline**: Base Features (2 features)

```python
FEATURE_COLS = ['Priority', 'Impact']
```

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Priority | Ordinal | 1-4 | Urgency level from ticketing system |
| Impact | Ordinal | 1-4 | Business impact level |

**Performance:** F1 81.97% with Attention+SMOTE (win48, 30min)

---

#### **f7-quick_win**: Interaction Features (7 features)

```python
# Tier 1: Simple interactions
df['priority_impact_product'] = df['Priority'] * df['Impact']
df['priority_impact_sum'] = df['Priority'] + df['Impact']
df['risk_score'] = (0.6 * df['Priority']) + (0.4 * df['Impact'])

# Tier 2: Polynomial features
df['priority_squared'] = df['Priority'] ** 2
df['impact_squared'] = df['Impact'] ** 2

FEATURE_COLS = [
    'Priority', 'Impact',
    'priority_impact_product', 'priority_impact_sum', 'risk_score',
    'priority_squared', 'impact_squared'
]
```

**Hypothesis:** Polynomial and interaction terms enable non-linear decision boundaries without deeper networks.

| Feature | Formula | Rationale |
|---------|---------|-----------|
| priority_impact_product | P × I | Joint severity (P=3,I=3 → 9 vs P=1,I=4 → 4) |
| risk_score | 0.6P + 0.4I | Domain-weighted priority emphasis |
| priority_squared | P² | Amplifies high-priority signals |

**Performance:** F1 82.05% with Attention+SMOTE (+0.08% from f2)

---

#### **f11-moderate**: Temporal Interactions (11 features)

```python
# Cyclical hour encoding (smooth transitions)
df['hour'] = df['Timestamp'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Time-aware severity
df['priority_hour_interaction'] = df['Priority'] * df['hour']
df['impact_hour_interaction'] = df['Impact'] * df['hour']

FEATURE_COLS = f7 + [
    'priority_hour_interaction', 'impact_hour_interaction',
    'hour_sin', 'hour_cos'
]
```

**Hypothesis:** Incidents follow diurnal patterns; high Priority at night may indicate different severity than during business hours.

**Performance:** F1 82.39% with Attention+SMOTE (+0.34% from f7)

---

#### **f13-service_moderate**: Service Context (13 features)

```python
# Target encoding by service type
service_incident_rates = df.groupby('Service Type')['incident'].mean()
df['service_risk_score'] = df['Service Type'].map(service_incident_rates)

# Frequency encoding
service_freq = df['Service Type'].value_counts(normalize=True)
df['service_frequency'] = df['Service Type'].map(service_freq)

FEATURE_COLS = f11 + ['service_risk_score', 'service_frequency']
```

**Hypothesis:** Some service types (e.g., Network, Database) have inherently higher incident rates; encoding this prior probability improves predictions.

| Feature | Calculation | Example |
|---------|-------------|---------|
| service_risk_score | Historical incident rate per service type | Network: 0.42, Email: 0.18 |
| service_frequency | Ticket volume proportion | Network: 0.35, Email: 0.08 |

**Performance:** F1 82.53% (+2.5% from f11) - **Best efficiency (features vs. performance)**

---

#### **f19-ultimate**: Sequence Features (19 features)

```python
# Volatility (instability indicators)
df['priority_volatility'] = df['Priority'].rolling(window=4, min_periods=1).std()
df['impact_volatility'] = df['Impact'].rolling(window=4, min_periods=1).std()

# Historical context
df['incidents_last_2h'] = df['incident'].rolling(window=4, min_periods=1).sum()
df['consecutive_high_priority'] = (df['Priority'] >= 2).rolling(window=6, min_periods=1).sum()

# Dynamics (rate of change)
df['priority_change'] = df['Priority'].diff()
df['priority_acceleration'] = df['Priority'].diff().diff()

FEATURE_COLS = f13 + [
    'priority_volatility', 'impact_volatility',
    'incidents_last_2h', 'consecutive_high_priority',
    'priority_change', 'priority_acceleration'
]
```

**Hypothesis:** Explicit temporal statistics reduce the burden on LSTM to learn these patterns implicitly, especially:
- **Volatility:** Rapid priority fluctuations signal emerging issues
- **Acceleration:** Increasing rate of change indicates escalating situations
- **Historical context:** Incidents tend to cluster

**Performance:** F1 82.94% (+0.4% from f13) - **Best overall performance**

---

### 4.3 Feature Package Comparison

| Package | Features | F1-Score | Δ from Baseline | Training Time |
|---------|----------|----------|-----------------|---------------|
| f2-baseline | 2 | 81.97%* | - | ~5400s |
| f7-quick_win | 7 | 82.05% | +0.08% | ~2600s |
| f11-moderate | 11 | 82.39% | +0.42% | ~2000s |
| f13-service_moderate | 13 | 82.53% | +0.56% | ~1200s |
| f15-full | 15 | 82.31% | +0.34% | ~1200s |
| **f19-ultimate** | **19** | **82.94%** | **+0.97%** | ~1300s |

*Note: f2-baseline with Attention achieves 81.97% due to attention's ability to learn implicit patterns, but requires 4x longer training.

---

## 5. LSTM Implementation from Scratch

This project implements LSTM (Long Short-Term Memory) networks entirely from scratch using NumPy/CuPy, without relying on deep learning frameworks like TensorFlow or PyTorch. This section provides comprehensive documentation of the implementation, covering both conceptual understanding and technical details.

### 5.1 Baseline LSTM: Conceptual Overview

#### **What is LSTM?**

LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) designed to learn long-term dependencies in sequential data. Unlike vanilla RNNs, which suffer from vanishing gradients when processing long sequences, LSTMs use a **gating mechanism** to selectively remember or forget information.

**The Key Insight:** LSTMs maintain two separate state vectors:
1. **Cell State (c):** The "long-term memory" that flows through time with minimal modification
2. **Hidden State (h):** The "short-term memory" or working memory used for predictions

**Why Gates?** Instead of having the network learn what to remember directly (which fails due to vanishing gradients), LSTMs learn **when to remember** through multiplicative gates. These gates are differentiable switches (values between 0-1) that control information flow.

#### **The Four Gates of LSTM**

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     LSTM Cell                           │
   c_{t-1} ─────────┼──────[×]────────────[+]────────────────────────────────┼──── c_t
                    │        ↑              ↑                                 │
                    │      f_t           i_t × c̃_t                            │
                    │   (forget)         (input)                              │
                    │        ↑              ↑                                 │
                    │     ┌──┴──┐        ┌──┴──┐      ┌──────┐               │
                    │     │  σ  │        │  σ  │ tanh │      │               │
                    │     └──┬──┘        └──┬──┘      └──┬───┘               │
   h_{t-1} ─────────┼───────┼──────────────┼───────────┼─────────[×]─────────┼──── h_t
                    │       └──────────────┴───────────┘          ↑          │
                    │                 z = [h_{t-1}; x_t]          │          │
                    │                                           o_t × tanh(c_t)
   x_t ─────────────┼─────────────────────────────────────────────┘          │
                    └─────────────────────────────────────────────────────────┘
```

| Gate | Symbol | Activation | Purpose |
|------|--------|------------|---------|
| **Forget Gate** | f_t | Sigmoid (0-1) | Decides what to discard from cell state |
| **Input Gate** | i_t | Sigmoid (0-1) | Decides what new information to add |
| **Candidate** | c̃_t | Tanh (-1 to 1) | Proposes new information to potentially add |
| **Output Gate** | o_t | Sigmoid (0-1) | Decides what to output from cell state |

---

### 5.2 Baseline LSTM: Technical Implementation

**File:** `src/model/lstm_cupy_optimized.py`

#### **5.2.1 Parameter Initialization**

The LSTM has the following learnable parameters:

```python
# Dimensions
input_size = 19      # Number of input features (e.g., f19-ultimate)
hidden_size = 128    # Number of hidden units
z_dim = hidden_size + input_size  # Concatenated input dimension = 147

# Gate weight matrices: (hidden_size, z_dim) = (128, 147)
Wf, Wi, Wc, Wo ∈ ℝ^(128 × 147)

# Gate bias vectors: (hidden_size, 1) = (128, 1)
bf, bi, bc, bo ∈ ℝ^(128 × 1)

# Output layer
Wy ∈ ℝ^(1 × 128)    # Output weights
by ∈ ℝ^(1 × 1)      # Output bias
```

**Xavier-like Initialization:** Weights are initialized with small random values (scale=0.1) to prevent exploding activations:

```python
self.params = {
    'Wf': cp.random.randn(hidden_size, z_dim).astype(cp.float32) * 0.1,
    'Wi': cp.random.randn(hidden_size, z_dim).astype(cp.float32) * 0.1,
    'Wc': cp.random.randn(hidden_size, z_dim).astype(cp.float32) * 0.1,
    'Wo': cp.random.randn(hidden_size, z_dim).astype(cp.float32) * 0.1,
    'Wy': cp.random.randn(output_size, hidden_size).astype(cp.float32) * 0.1,
    'bf': cp.zeros((hidden_size, 1)),  # Biases start at zero
    'bi': cp.zeros((hidden_size, 1)),
    'bc': cp.zeros((hidden_size, 1)),
    'bo': cp.zeros((hidden_size, 1)),
    'by': cp.zeros((output_size, 1)),
}
```

**Total Parameters:** For hidden_size=128, input_size=19:
- Gate weights: 4 × 128 × 147 = 75,264
- Gate biases: 4 × 128 = 512
- Output layer: 128 + 1 = 129
- **Total: ~75,905 parameters**

#### **5.2.2 Forward Pass (Inference)**

The forward pass processes a sequence of inputs and produces a prediction.

**Input Shape:** `X_batch = (batch_size, seq_len, input_size)` e.g., `(128, 48, 19)`

```python
def forward_batch(self, X_batch):
    batch_size, seq_len, _ = X_batch.shape

    # Initialize states to zeros: (hidden_size, batch_size) = (128, 128)
    h = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)
    c = cp.zeros((self.hidden_size, batch_size), dtype=cp.float32)

    caches = []  # Store intermediate values for backpropagation

    # Process each timestep t = 0, 1, ..., 47
    for t in range(seq_len):
        # Extract input at timestep t: (input_size, batch_size) = (19, 128)
        x_t = X_batch[:, t, :].T

        # Step 1: Concatenate h and x → z: (147, 128)
        z = cp.vstack((h, x_t))

        # Step 2: Compute gates (all parallelized across batch)
        f = sigmoid(Wf @ z + bf)      # Forget gate: (128, 128)
        i = sigmoid(Wi @ z + bi)      # Input gate: (128, 128)
        c_bar = tanh(Wc @ z + bc)     # Candidate: (128, 128)
        o = sigmoid(Wo @ z + bo)      # Output gate: (128, 128)

        # Step 3: Update cell state
        # c_new = (what to keep from old) + (what to add new)
        c_new = f * c + i * c_bar     # Element-wise: (128, 128)

        # Step 4: Compute new hidden state
        h_new = o * tanh(c_new)       # Element-wise: (128, 128)

        # Save for backpropagation
        caches.append((z, f, i, c_bar, c_new, o, h_new, c, h))

        # Update states for next timestep
        h, c = h_new, c_new

    # Step 5: Output layer (after processing all timesteps)
    # Use final hidden state h_T for prediction
    y_pred = sigmoid(Wy @ h + by)     # (1, 128) → probabilities

    return y_pred.flatten(), caches, h, c
```

**Activation Functions:**

```python
def sigmoid(x):
    """Maps any value to (0, 1) - used for gates"""
    return 1 / (1 + exp(-clip(x, -500, 500)))  # Clipping prevents overflow

def tanh(x):
    """Maps any value to (-1, 1) - used for candidate and cell state output"""
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

#### **5.2.3 Backward Pass (Backpropagation Through Time - BPTT)**

Backpropagation computes gradients of the loss with respect to all parameters, allowing the optimizer to update weights.

**Loss Function:** Binary Cross-Entropy (BCE) with optional class weighting

```
L = -1/N Σ [w_1 · y · log(ŷ) + (1-y) · log(1-ŷ)]
```

where w_1 is the class weight for positive samples (to handle imbalance).

**Chain Rule Application:**

The key insight is that we need to propagate gradients backward through time, from t=T to t=0.

```python
def backward_batch(self, y_pred, y_true, caches, cw=1):
    batch_size = len(y_true)
    seq_len = len(caches)

    # Initialize gradients dictionary
    grads = {k: cp.zeros_like(v) for k, v in self.params.items()}

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Output Layer Gradient
    # ═══════════════════════════════════════════════════════════
    # Loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    # dL/dŷ = (ŷ - y) / (ŷ(1-ŷ))
    # For sigmoid output: dL/d(pre_activation) = ŷ - y

    dy = (y_pred - y_true).reshape(1, -1)  # (1, batch_size)

    # Apply class weight: penalize mistakes on positive class more
    dy = cp.where(y_true == 1, dy * cw, dy)

    # Get final hidden state
    h_final = caches[-1][6]  # h_new from last timestep

    # Gradient for output weights: dL/dWy = dy @ h.T
    grads['Wy'] = cp.dot(dy, h_final.T) / batch_size
    grads['by'] = cp.sum(dy, axis=1, keepdims=True) / batch_size

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Backpropagate Through Hidden State
    # ═══════════════════════════════════════════════════════════
    # y = sigmoid(Wy @ h + by)
    # dL/dh = Wy.T @ dy

    dh_next = cp.dot(self.params['Wy'].T, dy)  # (hidden_size, batch_size)
    dc_next = cp.zeros((self.hidden_size, batch_size))

    # ═══════════════════════════════════════════════════════════
    # STEP 3: BPTT - Backward Through Time (t = T-1 to 0)
    # ═══════════════════════════════════════════════════════════
    for t in reversed(range(seq_len)):
        z, f, i, c_bar, c, o, h, c_prev, h_prev = caches[t]

        dh = dh_next

        # ─────────────────────────────────────────────────────
        # OUTPUT GATE GRADIENT
        # h = o * tanh(c)
        # dL/do = dL/dh * tanh(c)
        # ─────────────────────────────────────────────────────
        do = dh * cp.tanh(c)
        # Sigmoid derivative: σ'(x) = σ(x)(1 - σ(x))
        da_o = do * o * (1 - o)

        grads['Wo'] += cp.dot(da_o, z.T) / batch_size
        grads['bo'] += cp.sum(da_o, axis=1, keepdims=True) / batch_size

        # ─────────────────────────────────────────────────────
        # CELL STATE GRADIENT
        # h = o * tanh(c)
        # dL/dc = dL/dh * o * (1 - tanh²(c)) + dc_next
        # ─────────────────────────────────────────────────────
        dc = dh * o * (1 - cp.tanh(c)**2) + dc_next

        # ─────────────────────────────────────────────────────
        # CANDIDATE GRADIENT
        # c = f * c_prev + i * c_bar
        # dL/dc_bar = dL/dc * i
        # ─────────────────────────────────────────────────────
        dc_bar = dc * i
        # Tanh derivative: tanh'(x) = 1 - tanh²(x)
        da_c = dc_bar * (1 - c_bar**2)

        grads['Wc'] += cp.dot(da_c, z.T) / batch_size
        grads['bc'] += cp.sum(da_c, axis=1, keepdims=True) / batch_size

        # ─────────────────────────────────────────────────────
        # INPUT GATE GRADIENT
        # c = f * c_prev + i * c_bar
        # dL/di = dL/dc * c_bar
        # ─────────────────────────────────────────────────────
        di = dc * c_bar
        da_i = di * i * (1 - i)  # Sigmoid derivative

        grads['Wi'] += cp.dot(da_i, z.T) / batch_size
        grads['bi'] += cp.sum(da_i, axis=1, keepdims=True) / batch_size

        # ─────────────────────────────────────────────────────
        # FORGET GATE GRADIENT
        # c = f * c_prev + i * c_bar
        # dL/df = dL/dc * c_prev
        # ─────────────────────────────────────────────────────
        df = dc * c_prev
        da_f = df * f * (1 - f)  # Sigmoid derivative

        grads['Wf'] += cp.dot(da_f, z.T) / batch_size
        grads['bf'] += cp.sum(da_f, axis=1, keepdims=True) / batch_size

        # ─────────────────────────────────────────────────────
        # GRADIENTS FOR PREVIOUS TIMESTEP
        # z = [h_{t-1}; x_t]
        # dL/dz = sum of gradients from all gates
        # ─────────────────────────────────────────────────────
        dz = (cp.dot(Wf.T, da_f) + cp.dot(Wi.T, da_i) +
              cp.dot(Wc.T, da_c) + cp.dot(Wo.T, da_o))

        # Split gradient: dz = [dh_prev; dx_t]
        dh_next = dz[:self.hidden_size, :]  # Gradient to previous h

        # Cell state gradient through forget gate
        dc_next = f * dc  # Critical for long-term memory!

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Gradient Clipping (Prevent Exploding Gradients)
    # ═══════════════════════════════════════════════════════════
    for k in grads:
        grads[k] = cp.clip(grads[k], -5, 5)

    return grads
```

**Why Gradient Clipping?** Long sequences (48 timesteps) can cause gradients to explode during BPTT. Clipping at ±5 stabilizes training.

---

### 5.3 Adam Optimizer

Adam (Adaptive Moment Estimation) combines the benefits of:
1. **Momentum:** Uses exponential moving average of gradients (helps escape local minima)
2. **RMSprop:** Uses exponential moving average of squared gradients (adapts learning rate per parameter)

**Mathematical Formulation:**

```
For each parameter θ at timestep t:

1. Compute gradient: g_t = ∇L(θ)

2. Update biased first moment (momentum):
   m_t = β₁ · m_{t-1} + (1 - β₁) · g_t

3. Update biased second moment (RMSprop):
   v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

4. Bias correction (important in early training):
   m̂_t = m_t / (1 - β₁^t)
   v̂_t = v_t / (1 - β₂^t)

5. Update parameters:
   θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

**Hyperparameters:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| α (lr) | 0.001 | Base learning rate |
| β₁ | 0.9 | Momentum decay (how much past gradients matter) |
| β₂ | 0.999 | RMSprop decay (how much past squared gradients matter) |
| ε | 1e-8 | Prevents division by zero |
| λ (L2) | 0.01 | L2 regularization strength |

**Implementation:**

```python
def update_adam(self, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, lambda_l2=0.01):
    self.t += 1  # Increment timestep for bias correction

    for k in self.params:
        # Add L2 regularization to weight gradients
        actual_grad = grads[k]
        if k.startswith('W'):
            actual_grad += lambda_l2 * self.params[k]

        # Update biased moments
        self.m[k] = beta1 * self.m[k] + (1 - beta1) * actual_grad
        self.v[k] = beta2 * self.v[k] + (1 - beta2) * (actual_grad ** 2)

        # Bias correction
        m_hat = self.m[k] / (1 - beta1 ** self.t)
        v_hat = self.v[k] / (1 - beta2 ** self.t)

        # Parameter update
        self.params[k] -= lr * m_hat / (cp.sqrt(v_hat) + eps)
```

**Why Adam?**
- **Adaptive learning rates:** Each parameter gets its own effective learning rate based on gradient history
- **Handles sparse gradients:** Works well with varying gradient magnitudes across features
- **Bias correction:** Prevents early training instability

---

### 5.4 Architecture Variations

Each architecture variation modifies specific components of the baseline LSTM. This section details **exactly what changes** in forward and backward passes.

#### **5.4.1 Nested LSTM**
**File:** `src/model/lstm_nested.py`

**Concept:** Adds an **inner LSTM** within the cell state update, creating a hierarchical memory system with two paths for gradient flow.

**Modification to Forward Pass:**

```python
# BASELINE LSTM:
c_new = f * c_prev + i * c_bar

# NESTED LSTM:
# Step 1: Compute temporary cell state (same as baseline)
c_temp = f * c_prev + i * c_bar

# Step 2: Apply INNER gates to create final cell state
f_inner = sigmoid(Wf_i @ z + bf_i)  # Inner forget gate
i_inner = sigmoid(Wi_i @ z + bi_i)  # Inner input gate
c_new = f_inner * c_prev + i_inner * tanh(c_temp)  # Hierarchical update
```

**Why This Works:** The gradient now has TWO paths to flow back through time:
1. **Outer path:** Through `f * c_prev` in c_temp
2. **Inner path:** Through `f_inner * c_prev` directly

```
Gradient paths:
dc_prev = dc_temp * f        (outer: short-term patterns)
        + dc * f_inner       (inner: long-term patterns)
```

**Modification to Backward Pass:**

```python
# Additional gradients for inner gates
tanh_c_temp = cp.tanh(c_temp)

# Inner input gate gradient: dc/di_inner = tanh(c_temp)
di_inner = dc * tanh_c_temp
da_i_inner = di_inner * i_inner * (1 - i_inner)
grads['Wi_i'] += cp.dot(da_i_inner, z.T) / batch_size

# Inner forget gate gradient: dc/df_inner = c_prev
df_inner = dc * c_prev
da_f_inner = df_inner * f_inner * (1 - f_inner)
grads['Wf_i'] += cp.dot(da_f_inner, z.T) / batch_size

# Gradient through tanh(c_temp)
dc_temp = dc * i_inner * (1 - tanh_c_temp**2)

# dc_next now has contributions from BOTH paths
dc_next = dc_temp * f + dc * f_inner
```

**Additional Parameters:** +2 weight matrices (Wf_i, Wi_i) and +2 bias vectors (bf_i, bi_i)

---

#### **5.4.2 CIFG (Coupled Input-Forget Gate)**
**File:** `src/model/lstm_cifg.py`

**Concept:** Eliminates the input gate by coupling it with the forget gate: `i = 1 - f`. This enforces that old + new information always sums to 1.

**Modification to Forward Pass:**

```python
# BASELINE LSTM:
f = sigmoid(Wf @ z + bf)
i = sigmoid(Wi @ z + bi)  # Separate input gate

# CIFG LSTM:
f = sigmoid(Wf @ z + bf)
i = 1 - f                  # Coupled: no separate Wi, bi!

# Cell update unchanged:
c_new = f * c_prev + i * c_bar
# Equivalent to: c_new = f * c_prev + (1-f) * c_bar
```

**Modification to Backward Pass:**

```python
# The gradient that would go to input gate now affects forget gate
# Since i = 1 - f, we have di/df = -1

# Original gradients
df = dc * c_prev          # From forget path
di = dc * c_bar           # From input path (would go to Wi)

# Combined gradient (input gradient negated and added to forget)
df_combined = df - di     # Key change!

da_f = df_combined * f * (1 - f)
grads['Wf'] += cp.dot(da_f, z.T) / batch_size

# No Wi, bi gradients needed!
```

**Parameter Reduction:** Removes Wi and bi → ~25% fewer parameters

---

#### **5.4.3 Peephole LSTM**
**File:** `src/model/lstm_peephole.py`

**Concept:** Gates can directly "peek" at the cell state, allowing more precise timing decisions.

**Modification to Forward Pass:**

```python
# BASELINE LSTM:
f = sigmoid(Wf @ z + bf)
i = sigmoid(Wi @ z + bi)
o = sigmoid(Wo @ z + bo)

# PEEPHOLE LSTM:
# Forget and input gates see PREVIOUS cell state
f = sigmoid(Wf @ z + Vf * c_prev + bf)  # Peephole from c_{t-1}
i = sigmoid(Wi @ z + Vi * c_prev + bi)  # Peephole from c_{t-1}

c_new = f * c_prev + i * c_bar  # Same as baseline

# Output gate sees CURRENT cell state
o = sigmoid(Wo @ z + Vo * c_new + bo)  # Peephole from c_t!

h_new = o * tanh(c_new)
```

**Modification to Backward Pass:**

```python
# Additional peephole gradients
# V parameters are element-wise multiplied with cell state

# Output gate peephole: uses c_t (current cell state)
grads['Vo'] += cp.sum(da_o * c, axis=1, keepdims=True) / batch_size

# Input gate peephole: uses c_prev
grads['Vi'] += cp.sum(da_i * c_prev, axis=1, keepdims=True) / batch_size

# Forget gate peephole: uses c_prev
grads['Vf'] += cp.sum(da_f * c_prev, axis=1, keepdims=True) / batch_size

# dc also receives gradient through output gate peephole
dc += da_o * self.params['Vo']

# dc_next receives gradient through f and i peepholes
dc_next = f * dc + da_f * Vf + da_i * Vi
```

**Additional Parameters:** +3 peephole vectors (Vf, Vi, Vo) of size hidden_size each

---

#### **5.4.4 Bidirectional LSTM**
**File:** `src/model/lstm_bidirectional.py`

**Concept:** Processes the sequence in both forward and backward directions, then concatenates the final states.

**Architecture:**

```
Forward:  x_0 → x_1 → ... → x_T → h_f
Backward: x_0 ← x_1 ← ... ← x_T → h_b

Output: y = sigmoid(Wy @ [h_f; h_b] + by)
```

**Modification to Forward Pass:**

```python
def forward_batch(self, X_batch):
    batch_size, seq_len, _ = X_batch.shape

    # Separate states for each direction
    h_f, c_f = zeros(), zeros()
    h_b, c_b = zeros(), zeros()

    # Forward pass: t = 0, 1, ..., T-1
    for t in range(seq_len):
        x_t = X_batch[:, t, :].T
        h_f, c_f = lstm_cell(x_t, h_f, c_f, params='_f')

    # Backward pass: t = T-1, T-2, ..., 0
    for t in reversed(range(seq_len)):
        x_t = X_batch[:, t, :].T
        h_b, c_b = lstm_cell(x_t, h_b, c_b, params='_b')

    # Concatenate final states: (2*hidden_size, batch_size)
    h_combined = vstack((h_f, h_b))

    # Output uses concatenated representation
    y_pred = sigmoid(Wy @ h_combined + by)  # Wy is now (1, 2*hidden_size)

    return y_pred
```

**Modification to Backward Pass:**

```python
# Split gradient to both directions
dh_combined = Wy.T @ dy  # (2*hidden_size, batch_size)
dh_f = dh_combined[:hidden_size, :]
dh_b = dh_combined[hidden_size:, :]

# BPTT for forward LSTM (t = T-1 → 0)
for t in reversed(range(seq_len)):
    # Standard BPTT with '_f' parameters

# BPTT for backward LSTM (t = 0 → T-1)
for t in range(seq_len):
    # Standard BPTT with '_b' parameters
```

**Parameter Count:** ~2x parameters (separate weights for forward/backward)

---

#### **5.4.5 GLU (Gated Linear Unit) LSTM**
**File:** `src/model/lstm_glu.py`

**Concept:** Replaces `tanh` in candidate with a learnable gating mechanism, providing a linear path for gradients.

**Modification to Forward Pass:**

```python
# BASELINE LSTM:
c_bar = tanh(Wc @ z + bc)

# GLU LSTM:
A = Wc @ z + bc           # Linear projection (same weights)
B = W_glu @ z + b_glu     # Gate projection (new weights)
gate_B = sigmoid(B)
c_bar = A * gate_B        # GLU output: A ⊙ σ(B)
```

**Why GLU?**
- **Linear path through A:** Gradients flow directly without squashing
- **Learnable gate B:** Network learns which parts of A to keep
- **Better stability:** No saturation like tanh(-5) ≈ -1

**Modification to Backward Pass:**

```python
# GLU gradient: c_bar = A * sigmoid(B)
# dc_bar/dA = sigmoid(B) = gate_B
# dc_bar/dB = A * sigmoid(B) * (1 - sigmoid(B)) = A * gate_B * (1 - gate_B)

dc_bar = dc * i

# Gradient for A (linear path)
dA = dc_bar * gate_B
grads['Wc'] += cp.dot(dA, z.T) / batch_size
grads['bc'] += cp.sum(dA, axis=1, keepdims=True) / batch_size

# Gradient for B (gate path)
dB = dc_bar * A * gate_B * (1 - gate_B)
grads['W_glu'] += cp.dot(dB, z.T) / batch_size
grads['b_glu'] += cp.sum(dB, axis=1, keepdims=True) / batch_size

# dz now includes contribution from GLU gate
dz = ... + cp.dot(W_glu.T, dB)
```

**Additional Parameters:** +1 weight matrix (W_glu) and +1 bias vector (b_glu)

---

### 5.5 Attention Mechanism

**File:** `src/model/lstm_attention_optimized.py`

**Concept:** Instead of using only the final hidden state h_T, attention computes a weighted sum of ALL hidden states, allowing the model to focus on the most relevant timesteps.

#### **5.5.1 Attention Forward Pass**

```python
def forward_batch(self, X_batch):
    # Step 1: Run standard LSTM, but save ALL hidden states
    all_h = []
    for t in range(seq_len):
        h, c = lstm_step(x_t, h, c)
        all_h.append(h)

    # Stack: H = (seq_len, hidden_size, batch_size)
    H = cp.stack(all_h, axis=0)

    # Step 2: Compute attention scores
    # For each timestep t, compute: e_t = v_att^T @ tanh(W_att @ h_t)

    # S = tanh(W_att @ H) for all timesteps
    S = cp.tanh(W_att @ H)  # (seq_len, hidden_size, batch_size)

    # e = v_att^T @ S → scalar score per timestep
    e = v_att.T @ S         # (seq_len, batch_size)

    # Step 3: Softmax to get attention weights
    # α_t = exp(e_t) / Σ exp(e_j)
    e_exp = cp.exp(e - cp.max(e, axis=0))  # Numerical stability
    alpha = e_exp / cp.sum(e_exp, axis=0)  # (seq_len, batch_size)

    # Step 4: Compute context vector (weighted sum)
    # context = Σ α_t * h_t
    context = cp.sum(alpha[:, None, :] * H, axis=0)  # (hidden_size, batch_size)

    # Step 5: Output using context instead of final hidden state
    y_pred = sigmoid(Wy @ context + by)

    return y_pred
```

**Attention Visualization:**
```
Timestep:  t-47  t-46  ...  t-5   t-4   t-3   t-2   t-1   t
Hidden:    h_0   h_1   ...  h_43  h_44  h_45  h_46  h_47
Weight:    0.01  0.01  ...  0.05  0.08  0.15  0.25  0.35  0.10
                              ↑     ↑     ↑     ↑     ↑
                           Model attends to recent high-impact events
```

#### **5.5.2 Attention Backward Pass**

The attention mechanism adds several gradient paths:

```python
def backward_batch(self, y_pred, y_true, caches):
    # Output layer gradient
    dy = (y_pred - y_true)
    d_context = Wy.T @ dy  # Gradient to context vector

    # ═══════════════════════════════════════════════════════════
    # ATTENTION GRADIENTS
    # ═══════════════════════════════════════════════════════════

    # Gradient to alpha: context = Σ α_t * h_t
    # d_alpha[t] = d_context · h_t (dot product over hidden dim)
    d_alpha = cp.sum(d_context * H, axis=1)  # (seq_len, batch_size)

    # Gradient to H from attention: d_H[t] = α_t * d_context
    d_H_att = alpha[:, None, :] * d_context  # (seq_len, hidden_size, batch_size)

    # ═══════════════════════════════════════════════════════════
    # SOFTMAX GRADIENT
    # α = softmax(e)
    # Jacobian: dα_i/de_j = α_i(δ_ij - α_j)
    # Simplified: d_e = α * (d_alpha - Σ α * d_alpha)
    # ═══════════════════════════════════════════════════════════
    sum_alpha_dalpha = cp.sum(alpha * d_alpha, axis=0)
    d_e = alpha * (d_alpha - sum_alpha_dalpha)

    # ═══════════════════════════════════════════════════════════
    # ATTENTION PARAMETER GRADIENTS
    # e_t = v_att^T @ tanh(W_att @ h_t)
    # ═══════════════════════════════════════════════════════════

    # Gradient for v_att: dv_att = Σ S_t * d_e_t
    d_v_att = cp.sum(S * d_e[:, None, :], axis=(0, 2))
    grads['v_att'] = d_v_att.reshape(-1, 1) / batch_size

    # Gradient through tanh: d_S = d_e * v_att
    d_S = v_att * d_e[:, None, :]

    # Gradient for W_att: d_pre_S = d_S * (1 - S²)
    d_pre_S = d_S * (1 - S**2)
    for t in range(seq_len):
        d_W_att += d_pre_S[t] @ H[t].T
    grads['W_att'] = d_W_att / batch_size

    # Gradient to H from W_att transformation
    d_H_Watt = W_att.T @ d_pre_S

    # Total gradient to hidden states
    d_H_total = d_H_att + d_H_Watt

    # ═══════════════════════════════════════════════════════════
    # BPTT with attention gradients
    # Each h_t now receives gradient from attention mechanism
    # ═══════════════════════════════════════════════════════════
    for t in reversed(range(seq_len)):
        dh = dh_next + d_H_total[t]  # Add attention gradient!
        # ... standard BPTT continues
```

**Additional Parameters:**
- W_att: (hidden_size, hidden_size) - transforms hidden states for scoring
- v_att: (hidden_size, 1) - projects to scalar attention score

---

### 5.6 Architecture Comparison Summary

| Architecture | Forward Pass Modification | Backward Pass Modification | Parameter Change |
|--------------|---------------------------|----------------------------|------------------|
| **Baseline** | Standard 4-gate LSTM | Standard BPTT | Baseline (~76K) |
| **Nested** | Inner gates for hierarchical cell update | Dual gradient paths through inner/outer gates | +~16K (+21%) |
| **CIFG** | i = 1 - f (coupled gates) | df_combined = df - di | -~19K (-25%) |
| **Peephole** | Gates observe cell state via V vectors | Additional V gradients + dc through output peephole | +384 (+0.5%) |
| **BiDir** | Forward + backward processing | Separate BPTT for each direction | +~76K (2x) |
| **GLU** | c_bar = A ⊙ σ(B) instead of tanh | GLU gradient: dA and dB paths | +~19K (+25%) |
| **Attention** | Context = weighted sum of all h_t | Gradient to all h_t through attention | +~17K (+22%) |

---

## 6. Experimental Setup

### 6.1 Fixed Variables

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sampling Period | 30 minutes | Balances granularity vs. noise |
| Hidden Units | 128 | Sufficient capacity; larger showed no improvement |
| Optimizer | Adam | Adaptive learning rate handles varied gradients |
| L2 Regularization | 0.01 | Prevents overfitting on small dataset |
| Gradient Clipping | ±5 | Stabilizes training on long sequences |
| Batch Size | 128 | Maximizes GPU utilization |
| Learning Rate | 0.001 | Standard Adam default |

### 6.2 Optimization Variables

| Parameter | Values Tested | Best Value |
|-----------|---------------|------------|
| Window Size | 12, 24, 36, 48, 60, 72, 84, 96 | **48** |
| Feature Package | f2 to f19 | **f19-ultimate** |
| Architecture | 8 base + 5 attention variants | **Nested** |
| Data Balancing | SMOTE, Undersampling, Class Weight | **SMOTE** |

### 6.3 Hyperparameter Search Space

```python
WINDOW_SIZES = [48, 60]  # 24-30 hour lookback
SAMPLING_PERIODS = ['30min']
MODEL_TYPES = [
    'Optimized', 'Nested', 'CIFG', 'Peephole', 'BiDirectional',
    'GLU', 'LayerNorm', 'Residual',
    'Attention', 'Nested-Attention', 'CIFG-Attention',
    'BiDirectional-Attention', 'GLU-Attention'
]
FEATURE_STRATEGIES = [
    'baseline', 'quick_win', 'moderate', 'full',
    'service_quick_win', 'service_moderate', 'service_full',
    'sequence_quick_win', 'sequence_full', 'ultimate'
]
```

---

## 7. Results

### 7.1 Top 20 Model Configurations

| Rank | Interval | Win | Features | Architecture | F1 | Precision | Recall | ROC-AUC | PR-AUC | Threshold | Time |
|------|----------|-----|----------|--------------|-----|-----------|--------|---------|--------|-----------|------|
| 1 | 30min | 48 | f19-ultimate | Nested | **0.8294** | 0.8340 | 0.8249 | 0.9094 | 0.9263 | 0.402 | 1235s |
| 2 | 30min | 48 | f19-ultimate | CIFG | 0.8267 | 0.8010 | 0.8541 | 0.9091 | 0.9261 | 0.378 | 516s |
| 3 | 30min | 48 | f13-service_moderate | Attention | 0.8253 | 0.8247 | 0.8259 | 0.9090 | 0.9264 | 0.430 | 1197s |
| 4 | 30min | 48 | f19-ultimate | Peephole | 0.8252 | 0.8299 | 0.8205 | 0.9091 | 0.9264 | 0.430 | 1054s |
| 5 | 30min | 48 | f19-ultimate | Optimized | 0.8251 | 0.8412 | 0.8096 | 0.9094 | 0.9265 | 0.425 | 762s |
| 6 | 30min | 48 | f18-service_full | Attention | 0.8248 | 0.8515 | 0.7997 | 0.9080 | 0.9259 | 0.470 | 1388s |
| 7 | 30min | 48 | f13 | SMOTE-Attention | 0.8247 | 0.8167 | 0.8328 | 0.9092 | 0.9253 | 0.461 | 1556s |
| 8 | 30min | 48 | f13-service_moderate | Optimized | 0.8246 | 0.8170 | 0.8323 | 0.9075 | 0.9257 | 0.409 | 1194s |
| 9 | 30min | 48 | f11 | SMOTE-Attention | 0.8239 | 0.8118 | 0.8363 | 0.9074 | 0.9238 | 0.413 | 1043s |
| 10 | 30min | 48 | f19 | SMOTE-Nested | 0.8237 | 0.8191 | 0.8284 | 0.9090 | 0.9259 | 0.435 | 1292s |
| 11 | 30min | 48 | f15 | SMOTE-Attention | 0.8231 | 0.8468 | 0.8007 | 0.9117 | 0.9281 | 0.437 | 1190s |
| 12 | 30min | 48 | f19-ultimate | Attention | 0.8228 | 0.7887 | 0.8600 | 0.9087 | 0.9257 | 0.332 | 2514s |
| 13 | 30min | 48 | f18-service_full | Peephole | 0.8227 | 0.8470 | 0.7997 | 0.9088 | 0.9271 | 0.434 | 1013s |
| 14 | 30min | 48 | f13-service_moderate | CIFG | 0.8225 | 0.8059 | 0.8398 | 0.9073 | 0.9244 | 0.407 | 449s |
| 15 | 30min | 48 | f18-service_full | Nested | 0.8224 | 0.8263 | 0.8185 | 0.9102 | 0.9273 | 0.432 | 2067s |
| 16 | 30min | 48 | f13-service_moderate | Nested | 0.8223 | 0.8011 | 0.8447 | 0.9078 | 0.9251 | 0.356 | 1281s |
| 17 | 30min | 48 | f18-service_full | CIFG | 0.8223 | 0.8153 | 0.8294 | 0.9089 | 0.9266 | 0.393 | 678s |
| 18 | 30min | 48 | f13 | SMOTE-Attention | 0.8220 | 0.8220 | 0.8220 | 0.9070 | 0.9244 | 0.439 | 2298s |
| 19 | 30min | 48 | f18-service_full | Optimized | 0.8219 | 0.8453 | 0.7997 | 0.9094 | 0.9270 | 0.515 | 455s |
| 20 | 30min | 48 | f19 | SMOTE-Attention | 0.8212 | 0.8103 | 0.8323 | 0.9090 | 0.9257 | 0.384 | 2084s |

### 7.2 Window Size Analysis

| Window | Hours | Best F1 | Best Architecture | Observations |
|--------|-------|---------|-------------------|--------------|
| 12 | 6h | 0.7527 | LayerNorm | Insufficient context |
| 24 | 12h | 0.7835 | Residual | Captures half-day patterns |
| 36 | 18h | 0.8094 | CIFG | Approaching optimal |
| **48** | **24h** | **0.8294** | **Nested** | **Optimal: full daily cycle** |
| 60 | 30h | 0.8210 | Attention | Diminishing returns |
| 72 | 36h | 0.8080 | Attention | Starts degrading |
| 84 | 42h | 0.8134 | Attention | Gradient issues |
| 96 | 48h | 0.7726 | Attention | Too long, vanishing gradients |

![Window Size Performance](data/output/window_analysis.png)
*TODO: Generate visualization*

### 7.3 Architecture Performance by Feature Package

| Architecture | f2 | f7 | f13 | f19 | Δ (f2→f19) |
|--------------|-----|-----|------|------|------------|
| Nested | N/A | N/A | 82.23% | **82.94%** | - |
| CIFG | N/A | N/A | 82.25% | 82.67% | - |
| Attention | 81.97% | 82.05% | 82.53% | 82.28% | +0.31% |
| Optimized | N/A | N/A | 82.46% | 82.51% | - |

**Observation:** Nested LSTM benefits most from rich features (f19), while Attention can compensate for simpler features through learned attention patterns.

---

## 8. Analysis & Discussion

### 8.1 Architecture Analysis

#### Why Nested LSTM Performed Best

The Nested LSTM's **dual memory path** architecture is particularly suited for 24-hour incident prediction:

```
Standard LSTM gradient path:
dc_prev = dc * f  (single path, can vanish over 48 steps)

Nested LSTM gradient paths:
dc_prev = dc_temp * f       (outer path: short-term)
        + dc * f_inner      (inner path: long-term)
```

This dual path:
1. **Outer cell** (f, i gates): Captures immediate Priority/Impact changes
2. **Inner cell** (f_inner, i_inner): Maintains 24-hour historical context

For incident prediction, this matches the operational reality:
- Short-term: Sudden priority spikes (minutes to hours)
- Long-term: Service degradation patterns (hours to days)

#### Attention Impact Analysis

| Scenario | With Attention | Without Attention | Attention Benefit | Explanation |
|----------|----------------|-------------------|-------------------|-------------|
| Sparse features (f2) | 81.97% | ~75-78%* | +4-7% F1 | Attention compensates for missing features |
| Rich features (f19) | 82.28% | 82.94% (Nested) | -0.66% | Nested outperforms Attention with rich features |

*Estimated without SMOTE+Attention optimization

**Conclusion:** Attention is most valuable when feature engineering is limited. With f19-ultimate features, Nested LSTM outperforms Attention because explicit temporal statistics (volatility, acceleration) already capture what attention would learn.

### 8.2 Feature Engineering Impact

#### Cumulative Feature Contribution (with Attention+SMOTE, win48, 30min)

| Feature Group | Package | Features | F1 | Δ F1 |
|---------------|---------|----------|-----|------|
| Base | f2 | Priority, Impact | 81.97% | - |
| Interactions | f7 | +P×I, P², I² | 82.05% | +0.08% |
| Temporal | f11 | +hour_sin, hour_cos | 82.39% | +0.34% |
| Service | f13 | +risk_score, frequency | 82.53% | +0.14% |
| Full | f15 | +temporal features | 82.31% | -0.22% |
| Sequence | f19 | +volatility, acceleration | 82.94%* | +0.41% |

*Note: f19 best result (82.94%) uses Nested LSTM, not Attention.

#### Key Insight: Configuration > Features

The most significant finding is that **optimal configuration** (win48, 30min, SMOTE) enables even the simplest f2 model to achieve 81.97% F1. This suggests:

1. **Window size (48 = 24h lookback)** provides critical temporal context
2. **SMOTE** handles class imbalance effectively
3. **Attention mechanism** learns implicit patterns from raw Priority/Impact signals

Additional features from f7 to f19 provide only marginal gains (+0.97% total), indicating that the base Priority and Impact signals already contain most of the predictive information when properly windowed and balanced.

### 8.3 Window Size Analysis

#### Optimal Window: 48 Timesteps (24 Hours)

**Operational Alignment:**
- IT operations follow 24-hour cycles (day shift, night shift)
- Incidents often relate to daily batch jobs, peak usage hours
- 24-hour context captures complete operational pattern

**Technical Considerations:**
- 48 timesteps × 128 hidden units = manageable gradient flow
- Nested LSTM's dual path handles 48 steps effectively
- Beyond 60 steps, even Nested LSTM shows gradient degradation

---

## 9. Conclusion & Recommendations

### 9.1 Production Deployment Recommendation

**Champion Model:**
- **Architecture:** Nested LSTM
- **Features:** f19-ultimate (19 features)
- **Window:** 48 timesteps (24-hour lookback)
- **Sampling:** 30-minute intervals
- **Threshold:** 0.40 (optimized for balanced F1)

**Expected Performance:**
| Metric | Value |
|--------|-------|
| F1-Score | 82.94% |
| Precision | 83.40% |
| Recall | 82.49% |
| ROC-AUC | 90.94% |

### 9.2 Key Learnings

1. **Optimal Configuration is Critical**
   - With optimal config (win48, 30min, SMOTE), even f2 baseline achieves 81.97%
   - Moving from f2 to f19 provides only ~1% additional F1 improvement
   - The combination of window size, sampling, and SMOTE matters more than feature count

2. **Window Size Matters**
   - Too short (12): Misses daily patterns
   - Too long (96): Vanishing gradients, noise accumulation
   - Sweet spot: 24-30 hours (48-60 timesteps)

3. **SMOTE is Essential**
   - 2.2:1 class imbalance significantly hurts performance without balancing
   - SMOTE preserves all information while generating synthetic minorities

4. **Nested > Attention for Rich Features**
   - When features already encode temporal patterns, attention adds complexity without proportional benefit
   - Nested LSTM's hierarchical memory is more parameter-efficient

### 9.3 Next Steps

1. **Retraining Policy:**
   - Retrain weekly with new incident data
   - Monitor prediction calibration monthly
   - Alert if F1 drops below 80%

2. **Monitoring:**
   - Track prediction confidence distribution
   - Log false positives/negatives for analysis
   - Dashboard with attention weights for interpretability

3. **Future Experiments:**
   - Ensemble (Nested + Attention) for robustness
   - Real-time streaming inference optimization
   - Multi-task learning (incident + severity prediction)

---

## Appendix A: Complete Results Table

See [data/output/](data/output/) for individual experiment results.

**Total Experiments Conducted:** 116
**Best Model Checkpoints:** [src/model/checkpoints/](src/model/checkpoints/)

## Appendix B: Reproducibility

```bash
# Environment setup
pip install -r requirements.txt

# Run champion model training
python -c "
from notebooks.train_lstm_optimized_experiments import run_feature_gridsearch
results = run_feature_gridsearch(
    feature_strategies=['ultimate'],
    window_sizes=[48],
    sampling_periods=['30min'],
    model_types=['Nested']
)
"
```

## Appendix C: References

1. Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
2. Moniz & Krueger (2017). "Nested LSTMs"
3. Greff et al. (2017). "LSTM: A Search Space Odyssey"
4. Bahdanau et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate"
5. Dauphin et al. (2017). "Language Modeling with Gated Convolutional Networks"
6. Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"

---

*Generated on February 8, 2026*
*Total training time across all experiments: ~72 hours*
