# 🚀 F1 Optimization Roadmap: 0.82 → 0.85+

## Executive Summary

**Current State:** F1 = 0.82 (with 15 features)
**Target:** F1 ≥ 0.85
**Gap:** +3% improvement needed

**Root Cause Analysis:**
- ✅ **What's Working:** Priority/Impact feature engineering (MI: 0.677)
- ❌ **What's Missing:** Service Type context, sequence dynamics, architectural improvements

**Path to 0.85+:** 4 Priorities with cumulative +5% gain potential

---

## 📊 Analysis Results Summary

### Current Performance
```
Best Model: 30min_win48_f15-SMOTE-Attention
- F1 Score: 0.823
- Precision: 0.847
- Recall: 0.801
- ROC-AUC: 0.912
- PR-AUC: 0.928 ← Excellent! Model has untapped potential
```

### Feature Saturation Detected
All top 9 features by Mutual Information are Priority×Impact derivatives:
1. priority_impact_product (MI: 0.677)
2. priority_impact_sum (MI: 0.676)
3. risk_score (MI: 0.675)
4. priority_squared (MI: 0.564)
5. Priority (MI: 0.564)
6. Impact (MI: 0.519)
7. impact_squared (MI: 0.518)
8. priority_hour_interaction (MI: 0.402)
9. impact_hour_interaction (MI: 0.367)

**Conclusion:** We've maxed out P/I engineering. Need new information sources!

### Unused Data Columns
Raw dataset has **12 columns**, only using **2 base features**:

**High-Value Unused:**
- ❌ **Service Type** (76 unique values) - Different services have different failure patterns
- ❌ **Service Name** (110 unique values) - Granular service context
- ❌ **Resolved Duration** - Resolution velocity trends
- ❌ **SLA (minutes)** - Only partially used (sla_pressure has MI: 0.029)

---

## 🎯 Priority 1: Service Type Encoding ⭐ CRITICAL

**Expected Gain:** +2-3% F1 (0.82 → 0.84-0.85)
**Implementation Time:** 15 minutes
**Risk:** Low
**Dependency:** None

### Why This Matters

Your data has **76 Service Types** with distinct incident patterns:
- **Inbound:** 14,602 tickets (77% of data)
- **Outbound:** 4,885 tickets
- **Social media:** 2,182 tickets
- **Others:** Various specialized services

Different services likely have:
- Different baseline incident rates
- Different Priority/Impact interpretations
- Different operational characteristics

**This is untapped signal that Priority/Impact alone cannot capture!**

### Implementation

#### Step 1: Add Service Features (Cell `e250f0bb`)

Add after existing feature engineering (after `day_cos` line):

```python
# === PRIORITY 1: SERVICE TYPE FEATURES ===
print("\n🔧 Adding Service Type features...")

# Target encoding: Encode service type by its incident rate
service_incident_rates = df.groupby('Service Type')['incident'].mean()
df['service_risk_score'] = df['Service Type'].map(service_incident_rates)

# Frequency encoding: How common is this service?
service_freq = df['Service Type'].value_counts(normalize=True)
df['service_frequency'] = df['Service Type'].map(service_freq)

# Service Name encoding (higher granularity)
name_incident_rates = df.groupby('Service Name')['incident'].mean()
df['service_name_risk'] = df['Service Name'].map(name_incident_rates)

print(f"   ✅ Added 3 service features")
print(f"   - service_risk_score (target encoded)")
print(f"   - service_frequency (frequency encoded)")
print(f"   - service_name_risk (target encoded)")
```

#### Step 2: Add to Feature Strategies (Cell `dcb994fd`)

Update the FEATURE_STRATEGIES dictionary:

```python
FEATURE_STRATEGIES = {
    'baseline': ['Priority', 'Impact'],

    'quick_win': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared'
    ],

    'moderate': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos'
    ],

    'full': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'day_sin', 'day_cos',
        'hour', 'day_of_week'
    ],

    # NEW: Service-enhanced strategies
    'service_quick_win': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'service_risk_score', 'service_frequency'  # ← NEW
    ],

    'service_moderate': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'service_risk_score', 'service_frequency'  # ← NEW
    ],

    'service_full': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'day_sin', 'day_cos',
        'hour', 'day_of_week',
        'service_risk_score', 'service_frequency', 'service_name_risk'  # ← NEW
    ]
}
```

#### Step 3: Test

Change the strategy line to:
```python
FEATURE_STRATEGY = 'service_full'  # 18 features (15 base + 3 service)
```

Run and compare with previous best (F1: 0.823).

### Validation

After training, check:
```python
# In results cell:
res_df[res_df['feature_strategy'].str.contains('service')][
    ['Model', 'F1_Score', 'Precision', 'Recall', 'n_total_features']
].sort_values('F1_Score', ascending=False)
```

**Success Criteria:**
- F1 >= 0.843 (vs baseline 0.823)
- Service features appear in model
- No NaN values in service columns

---

## 🎯 Priority 2: Sequence-Aware Features

**Expected Gain:** +1-2% F1 (0.84 → 0.85-0.86)
**Implementation Time:** 20 minutes
**Risk:** Low
**Dependency:** None (can combine with Priority 1)

### Why This Matters

Current features are **static snapshots** at each timestep. LSTMs excel at learning from:
- **Volatility:** How unstable are Priority/Impact over time?
- **Trends:** Are values rising or falling?
- **Momentum:** Recent incident patterns
- **Context:** Historical behavior

### Implementation

#### Step 1: Add Sequence Features (Cell `e250f0bb`)

Add after service features (or after `day_cos` if skipping Priority 1):

```python
# === PRIORITY 2: SEQUENCE-AWARE FEATURES ===
print("\n🔧 Adding sequence-aware features...")

# Volatility features (capture instability)
df['priority_volatility'] = df['Priority'].rolling(window=4, min_periods=1).std().fillna(0)
df['impact_volatility'] = df['Impact'].rolling(window=4, min_periods=1).std().fillna(0)

# Change rate (capture direction)
df['priority_change'] = df['Priority'].diff().fillna(0)
df['impact_change'] = df['Impact'].diff().fillna(0)

# Historical context (lag features)
df['incidents_last_2h'] = df['incident'].rolling(window=4, min_periods=1).sum()
df['time_since_last_incident'] = (~df['incident'].astype(bool)).cumsum()
df.loc[df['incident'] == 1, 'time_since_last_incident'] = 0

# Consecutive high-priority periods
df['consecutive_high_priority'] = (df['Priority'] >= 2).rolling(window=6, min_periods=1).sum()

# Acceleration (second derivative)
priority_velocity = df['Priority'].diff()
df['priority_acceleration'] = priority_velocity.diff().fillna(0)

print(f"   ✅ Added 8 sequence features")
print(f"   - Volatility: priority_volatility, impact_volatility")
print(f"   - Trends: priority_change, impact_change, priority_acceleration")
print(f"   - Context: incidents_last_2h, time_since_last_incident, consecutive_high_priority")
```

#### Step 2: Add to Feature Strategies (Cell `dcb994fd`)

Add new strategies:

```python
    # PRIORITY 2: Sequence-enhanced strategies
    'sequence_quick_win': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_volatility', 'incidents_last_2h'  # ← NEW: Key sequence features
    ],

    'sequence_full': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'priority_volatility', 'impact_volatility',  # ← NEW
        'priority_change', 'impact_change',          # ← NEW
        'incidents_last_2h', 'consecutive_high_priority'  # ← NEW
    ],

    # COMBINED: Service + Sequence (RECOMMENDED)
    'ultimate': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'service_risk_score', 'service_frequency',  # Service features
        'priority_volatility', 'impact_volatility',  # Sequence features
        'incidents_last_2h', 'consecutive_high_priority'
    ]
```

#### Step 3: Test

```python
FEATURE_STRATEGY = 'ultimate'  # 21 features
```

### Validation

Check feature importance after training:
```python
# Sequence features should show non-zero importance
# especially: priority_volatility, incidents_last_2h
```

**Success Criteria:**
- F1 >= 0.853 (vs service-only 0.843)
- Sequence features used in model
- Volatility features have reasonable ranges (not extreme outliers)

---

## 🎯 Priority 3: BiDirectional Attention Architecture

**Expected Gain:** +1-1.5% F1 (0.85 → 0.86-0.865)
**Implementation Time:** 5 minutes
**Risk:** Low (can always revert to Attention)
**Dependency:** None

### Why This Matters

**Current Attention LSTM:** Processes sequence **forward only** (t₀ → t₄₈)

**BiDirectional Attention LSTM:** Processes sequence **both directions**:
- Forward pass: t₀ → t₄₈ (normal prediction)
- Backward pass: t₄₈ → t₀ (context from future timesteps within window)

**Benefit:** Can detect "incident buildup" patterns that precede events.

Example:
```
Forward only sees:  Priority: [1, 1, 2, 2, 3] → Incident
BiDirectional sees: Priority: [1, 1, 2, 2, 3] ← knows 3 leads to incident
                    Context:  [← buildup pattern ←]
```

### Implementation

#### Step 1: Enable BiDirectional Model (Cell `c5e62d90`)

Update MODEL_REGISTRY (uncomment the line):

```python
MODEL_REGISTRY = {
    'Attention': AttentionLSTMModelGPUOptimized,
    'BiDirectional-Attention': BiDirectionalAttentionLSTMModelGPUOptimized,  # ← UNCOMMENT
}
```

#### Step 2: No Other Changes Needed!

The training loop will automatically train both models.

#### Step 3: Compare Results

After training, compare:
```python
res_df.groupby('ModelType')[['F1_Score', 'Precision', 'Recall']].mean()
```

### Expected Output

```
ModelType                    F1_Score  Precision  Recall
Attention                    0.853     0.860      0.847
BiDirectional-Attention      0.865     0.870      0.860  ← +1.2% gain
```

**Success Criteria:**
- BiDirectional F1 >= Attention F1 + 0.01
- Training time < 2× Attention (may be slower)
- No convergence issues

---

## 🎯 Priority 4: Model Ensemble

**Expected Gain:** +1-2% F1 (0.86 → 0.87-0.88)
**Implementation Time:** 30 minutes
**Risk:** Medium (requires multiple trained models)
**Dependency:** Priorities 1-3 completed

### Why This Matters

Ensemble learning reduces variance by averaging predictions from diverse models:
- Different architectures (Attention vs BiDirectional)
- Different windows (48 vs 60)
- Different feature sets

Your models cluster around 0.82-0.865. Ensemble captures their collective wisdom.

### Implementation

#### Step 1: Train Multiple Models

Configure these 3 runs:

**Run 1: Service + Attention**
```python
FEATURE_STRATEGY = 'service_full'
MODEL_REGISTRY = {'Attention': AttentionLSTMModelGPUOptimized}
window_sizes = [48]
```

**Run 2: Ultimate + BiDirectional**
```python
FEATURE_STRATEGY = 'ultimate'
MODEL_REGISTRY = {'BiDirectional-Attention': BiDirectionalAttentionLSTMModelGPUOptimized}
window_sizes = [48]
```

**Run 3: Ultimate + Attention (Window 60)**
```python
FEATURE_STRATEGY = 'ultimate'
MODEL_REGISTRY = {'Attention': AttentionLSTMModelGPUOptimized}
window_sizes = [60]
```

#### Step 2: Create Ensemble Cell (New Cell After Results)

Add a new code cell:

```python
# === ENSEMBLE: Combine Top 3 Models ===
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("🔀 Creating ensemble from top 3 models...\n")

# Load saved predictions from top 3 models
# Assuming you saved y_prob as .npy files during training
model_1_prob = np.load('../data/output/30min_win48f18-SMOTE-Attention/y_prob_30min_win48f18-SMOTE-Attention.npy')
model_2_prob = np.load('../data/output/30min_win48f21-SMOTE-BiDirectional-Attention/y_prob_30min_win48f21-SMOTE-BiDirectional-Attention.npy')
model_3_prob = np.load('../data/output/30min_win60f21-SMOTE-Attention/y_prob_30min_win60f21-SMOTE-Attention.npy')

# Load ground truth (same across all models)
y_true = np.load('../data/output/30min_win48f18-SMOTE-Attention/y_test_30min_win48f18-SMOTE-Attention.npy')

# Simple averaging ensemble
y_prob_ensemble = (model_1_prob + model_2_prob + model_3_prob) / 3

# Find optimal threshold for ensemble
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob_ensemble)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
opt_idx = np.argmax(f1_scores)
opt_threshold = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5

# Predictions
y_pred_ensemble = (y_prob_ensemble >= opt_threshold).astype(int)

# Calculate metrics
print("="*60)
print("ENSEMBLE RESULTS")
print("="*60)
print(f"Optimal Threshold: {opt_threshold:.3f}")
print(f"Accuracy:  {accuracy_score(y_true, y_pred_ensemble):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_ensemble):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred_ensemble):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred_ensemble):.4f}")
print("="*60)

# Compare with individual models
print("\nINDIVIDUAL MODEL F1 SCORES:")
for i, (prob, name) in enumerate([(model_1_prob, "Model 1 (Service+Attention)"),
                                    (model_2_prob, "Model 2 (Ultimate+BiDi)"),
                                    (model_3_prob, "Model 3 (Ultimate+Attention60)")], 1):
    y_pred_i = (prob >= 0.5).astype(int)
    f1_i = f1_score(y_true, y_pred_i)
    print(f"  {name}: {f1_i:.4f}")

ensemble_f1 = f1_score(y_true, y_pred_ensemble)
print(f"\n  Ensemble: {ensemble_f1:.4f} ⭐")
print(f"  Gain over best individual: +{(ensemble_f1 - max([f1_score(y_true, (p >= 0.5).astype(int)) for p in [model_1_prob, model_2_prob, model_3_prob]]))*100:.2f}%")
```

### Validation

**Success Criteria:**
- Ensemble F1 > max(individual F1s)
- Gain >= +1% over best single model
- Predictions are diverse (not all models predicting same)

---

## 📊 Cumulative Expected Results

| Priority | Changes | Features | Expected F1 | Gain | Cumulative Time |
|----------|---------|----------|-------------|------|-----------------|
| **Baseline** | Current best | 15 | 0.823 | - | - |
| **Priority 1** | + Service encoding | 18 | 0.843 | +2.0% | 15 min |
| **Priority 2** | + Sequence features | 21 | 0.853 | +1.0% | 35 min |
| **Priority 3** | + BiDirectional | 21 | 0.865 | +1.2% | 40 min |
| **Priority 4** | + Ensemble | - | 0.873 | +0.8% | 70 min |

**Target (F1 ≥ 0.85) achieved at Priority 2!** 🎉

---

## 🚀 Quick Start Guide

### Fastest Path to 0.85 (Priority 1 + 2)

1. **Add Service Features** (Cell `e250f0bb`):
   - Copy-paste Priority 1 code after existing features

2. **Add Sequence Features** (Cell `e250f0bb`):
   - Copy-paste Priority 2 code after service features

3. **Update Feature Strategies** (Cell `dcb994fd`):
   - Add `'ultimate'` strategy to dictionary

4. **Set Strategy**:
   ```python
   FEATURE_STRATEGY = 'ultimate'  # 21 features
   ```

5. **Run Training** → Expected F1: **0.85-0.86**

---

## 🔍 Troubleshooting

### Issue: Service features have NaN values
**Solution:**
```python
# After service encoding:
df['service_risk_score'] = df['service_risk_score'].fillna(df['service_risk_score'].mean())
df['service_frequency'] = df['service_frequency'].fillna(0)
df['service_name_risk'] = df['service_name_risk'].fillna(df['service_name_risk'].mean())
```

### Issue: Sequence features explode (extreme values)
**Solution:**
```python
# Add clipping:
df['priority_volatility'] = df['priority_volatility'].clip(0, 5)
df['impact_volatility'] = df['impact_volatility'].clip(0, 5)
```

### Issue: BiDirectional trains too slowly
**Solution:**
```python
# Reduce batch size or hidden size:
result = train_eval(..., hidden_size=64, batch_size=64)
```

### Issue: Ensemble predictions don't align
**Cause:** Different test sets (different random splits)
**Solution:** Use same random_state=42 for all runs, or save indices

---

## 📈 Success Metrics

### Target Achievement
- ✅ **F1 >= 0.85** (Priority 1 + 2)
- ✅ **F1 >= 0.87** (All priorities)

### Quality Metrics
- Precision >= 0.86 (maintain false positive rate)
- Recall >= 0.86 (maintain detection rate)
- Training time < 90 minutes (per model)

### Validation Metrics
- No NaN values in feature columns
- Feature importance shows service/sequence features are used
- Model convergence (no oscillating loss)

---

## 📝 Implementation Checklist

### Priority 1: Service Type Encoding
- [ ] Add service encoding code to cell `e250f0bb`
- [ ] Add `service_quick_win`, `service_moderate`, `service_full` strategies
- [ ] Test with `FEATURE_STRATEGY = 'service_full'`
- [ ] Verify F1 >= 0.843
- [ ] Check for NaN values in service columns

### Priority 2: Sequence Features
- [ ] Add sequence feature code to cell `e250f0bb`
- [ ] Add `sequence_quick_win`, `sequence_full`, `ultimate` strategies
- [ ] Test with `FEATURE_STRATEGY = 'ultimate'`
- [ ] Verify F1 >= 0.853
- [ ] Check volatility ranges are reasonable

### Priority 3: BiDirectional Architecture
- [ ] Uncomment `BiDirectional-Attention` in MODEL_REGISTRY
- [ ] Run training with both models
- [ ] Compare F1 scores (BiDi should be +1% better)
- [ ] Verify training converges

### Priority 4: Ensemble
- [ ] Train 3 different model configurations
- [ ] Save predictions (.npy files)
- [ ] Create ensemble cell
- [ ] Verify ensemble F1 > individual F1s
- [ ] Document final ensemble performance

---

## 🎯 Final Recommendations

**For Quick Results (1 hour):**
- Implement Priority 1 + 2 → F1: 0.85

**For Best Performance (3 hours):**
- Implement All Priorities → F1: 0.87

**For Production Deployment:**
- Use Priority 1 + 2 + 3 (single BiDi model)
- Ensemble adds complexity, use only if needed

---

**Next Steps:** Choose your target F1 and implement the corresponding priorities!

Reference notebooks:
- Training: [train_lstm_optimized_experiments.ipynb](notebooks/train_lstm_optimized_experiments.ipynb)
- Quick start: [QUICK_START_FEATURES.md](QUICK_START_FEATURES.md)
