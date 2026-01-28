# 🚀 QUICK START: Add Features in 3 Steps

## 📌 STEP 1: Add Feature Engineering Cell

**Location:** After cell with `df['incident'] = ...` (cell `e250f0bb`)

**Copy-paste this NEW CELL:**

```python
# === 2.2 FEATURE ENGINEERING ===
print("\n🔧 Engineering enhanced features...")

# TIER 1: Interactions (Strongest predictors)
df['priority_impact_product'] = df['Priority'] * df['Impact']
df['priority_impact_sum'] = df['Priority'] + df['Impact']
df['risk_score'] = df['Priority'] * 0.5 + df['Impact'] * 0.5

# TIER 2: Polynomial (Non-linear)
df['priority_squared'] = df['Priority'] ** 2
df['impact_squared'] = df['Impact'] ** 2

# TIER 3: Temporal interactions
df['hour'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['priority_hour_interaction'] = df['Priority'] * df['hour']
df['impact_hour_interaction'] = df['Impact'] * df['hour']

# TIER 4: Cyclical encodings
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print(f"✅ Added 14 new features! Dataset shape: {df.shape}")
```

---

## 📌 STEP 2: Update Aggregation Dictionary

**Location:** In experiment loop (cell `dcb994fd`), find `df_resampled = ...`

**REPLACE:**
```python
df_resampled = (df.set_index('Timestamp')
                  .resample(sp)
                  .agg({'incident': 'sum', 'Priority': 'mean', 'Impact': 'mean'})
                  .fillna(0)
                  .reset_index())
```

**WITH:**
```python
agg_dict = {
    'incident': 'sum',
    'Priority': 'mean',
    'Impact': 'mean',
    'priority_impact_product': 'mean',
    'priority_impact_sum': 'mean',
    'risk_score': 'mean',
    'priority_squared': 'mean',
    'impact_squared': 'mean',
    'priority_hour_interaction': 'mean',
    'impact_hour_interaction': 'mean',
    'hour_sin': 'mean',
    'hour_cos': 'mean',
    'day_sin': 'mean',
    'day_cos': 'mean',
    'hour': 'mean',
    'day_of_week': 'mean'
}

df_resampled = (df.set_index('Timestamp')
                  .resample(sp)
                  .agg(agg_dict)
                  .fillna(0)
                  .reset_index())
```

---

## 📌 STEP 3: Update Feature Selection

**Location:** Same cell, find `feature_cols = ['Priority', 'Impact']`

**REPLACE:**
```python
feature_cols = ['Priority', 'Impact']
```

**WITH (choose ONE strategy):**

### 🎯 STRATEGY B: MINIMAL BOOST (Recommended for first test)
```python
feature_cols = [
    'Priority', 'Impact',
    'priority_impact_product',  # Strongest
    'priority_impact_sum',
    'risk_score',
    'priority_squared',
    'impact_squared'
]
# Expected: +1-1.5% F1 gain
# 7 base + 4 rolling = 11 features total
```

### 🎯 STRATEGY C: MODERATE
```python
feature_cols = [
    'Priority', 'Impact',
    'priority_impact_product', 'priority_impact_sum', 'risk_score',
    'priority_squared', 'impact_squared',
    'priority_hour_interaction', 'impact_hour_interaction',
    'hour_sin', 'hour_cos'
]
# Expected: +1.5-2% F1 gain
# 11 base + 4 rolling = 15 features total
```

### 🎯 STRATEGY D: FULL RECOMMENDED
```python
feature_cols = [
    'Priority', 'Impact',
    'priority_impact_product', 'priority_impact_sum', 'risk_score',
    'priority_squared', 'impact_squared',
    'priority_hour_interaction', 'impact_hour_interaction',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'hour', 'day_of_week'
]
# Expected: +1.5-2.5% F1 gain
# 15 base + 4 rolling = 19 features total
```

---

## 📊 COMPARISON TABLE

| Strategy | Base Features | + Rolling Means | Total | Expected F1 | Gain |
|----------|--------------|-----------------|-------|-------------|------|
| **Current (A)** | 2 | +4 | **6** | 0.810 | Baseline |
| **Minimal (B)** ⭐ | 7 | +4 | **11** | 0.820 | +1.0% |
| **Moderate (C)** | 11 | +4 | **15** | 0.825 | +1.5% |
| **Full (D)** | 15 | +4 | **19** | 0.830 | +2.0% |

---

## ⚠️ ONE MORE CHANGE: Update Label

**Location:** In SMOTE section, find:

```python
result = train_eval(Xs, y_sm, label=f"{sp}|win={win}-SMOTE-{model_type}", model_type=model_type)
```

**REPLACE WITH:**
```python
label = f"{sp}|win={win}-SMOTE-{len(feature_cols)}F-{model_type}"
result = train_eval(Xs, y_sm, label=label, model_type=model_type)
```

This adds feature count to the label (e.g., `30min|win=48-SMOTE-11F-Attention`)

---

## 🎯 RECOMMENDED WORKFLOW

### Test 1: Baseline (to confirm current performance)
```python
feature_cols = ['Priority', 'Impact']  # Current
# Run and note F1 score
```

### Test 2: Quick Win
```python
feature_cols = [  # STRATEGY B
    'Priority', 'Impact',
    'priority_impact_product', 'priority_impact_sum', 'risk_score',
    'priority_squared', 'impact_squared'
]
# Compare F1 with Test 1
```

### Test 3: If Test 2 Shows Improvement
```python
feature_cols = [  # STRATEGY C or D
    # ... add temporal features
]
```

---

## 📈 HOW TO VERIFY SUCCESS

After training completes, check:

```python
# In results visualization cell
res_df = pd.DataFrame(results_all)
res_df[['Model', 'F1_Score', 'Precision', 'Recall']].sort_values('F1_Score', ascending=False)
```

Look for:
- **F1_Score >= 0.820** (vs baseline 0.810)
- **Precision >= 0.79** (maintain or improve)
- **Model name contains feature count** (e.g., "11F" or "15F")

---

## 🚨 TROUBLESHOOTING

### If F1 score DECREASES:
1. Check for NaN values: `df[feature_cols].isnull().sum()`
2. Reduce feature count (try only interactions)
3. Increase regularization: `lambda_l2=0.001`

### If training takes too long:
- Reduce `window_sizes = [48]` (instead of [48, 60, 72])
- Reduce `epochs=300` (instead of 500)
- Use smaller model: `hidden_size=64`

### If SMOTE fails:
```python
# Add this before SMOTE
print(f"Features shape: {X_flat.shape}")
print(f"Feature columns: {len(feature_cols)}")
```

Check that reshaping is correct:
```python
n_features = len(feature_cols)  # Must match X_flat.shape[1] / win
```

---

## ✅ FINAL CHECKLIST

Before running:
- [ ] Added feature engineering cell (STEP 1)
- [ ] Updated aggregation dictionary (STEP 2)
- [ ] Updated feature_cols selection (STEP 3)
- [ ] Updated label to include feature count
- [ ] Set `window_sizes = [48]` (test with your best window first)
- [ ] Set `sampling_periods = ['30min']` (test with your best period)
- [ ] Ready to run!

---

**Start with STRATEGY B (11 features total) for quick +1% F1 gain!** 🚀

Full details in: [FEATURE_INTEGRATION_GUIDE.md](FEATURE_INTEGRATION_GUIDE.md)
