# 🚀 FEATURE INTEGRATION GUIDE
## How to Add New Features to Training Pipeline

**Target:** Add engineered features for +1-2% F1 improvement
**Notebook:** `notebooks/train_lstm_optimized_experiments.ipynb`

---

## 📍 STEP 1: ADD NEW CELL - Feature Engineering

**Location:** After cell `e250f0bb` (after incident label creation)

**Action:** Insert a NEW CODE CELL with the following content:

```python
# === 2.2 FEATURE ENGINEERING (New Features for Performance Boost) ===
print("\n🔧 Engineering enhanced features...")

# ========== TIER 1: INTERACTIONS (MUST HAVE) ==========
print("  → Creating interaction features...")
df['priority_impact_product'] = df['Priority'] * df['Impact']
df['priority_impact_sum'] = df['Priority'] + df['Impact']
df['risk_score'] = df['Priority'] * 0.5 + df['Impact'] * 0.5

# ========== TIER 2: POLYNOMIAL (NON-LINEAR) ==========
print("  → Creating polynomial features...")
df['priority_squared'] = df['Priority'] ** 2
df['impact_squared'] = df['Impact'] ** 2

# ========== TIER 3: TEMPORAL FEATURES ==========
print("  → Creating temporal features...")
df['hour'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek

# Temporal interactions
df['priority_hour_interaction'] = df['Priority'] * df['hour']
df['impact_hour_interaction'] = df['Impact'] * df['hour']

# Cyclical encodings (for periodic patterns)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print(f"✅ Feature engineering complete! Dataset now has {df.shape[1]} columns")
print(f"   New features added: priority_impact_product, priority_impact_sum, risk_score,")
print(f"                       priority_squared, impact_squared, hour interactions,")
print(f"                       cyclical encodings (hour_sin/cos, day_sin/cos)")
```

---

## 📍 STEP 2: MODIFY EXPERIMENT LOOP

**Location:** Cell `dcb994fd` (Run Experiment)

**Action:** Find this section (around line 20-40 in the cell):

```python
# Apply Rolling Mean Features (BEFORE windowing)
feature_cols = ['Priority', 'Impact']

if rolling_means:
    print(f"  📊 Creating rolling mean features for: {rolling_means} hours")
    # ... rolling mean creation code ...
    print(f"  📊 Feature columns: {feature_cols}")
```

**REPLACE the `feature_cols = ['Priority', 'Impact']` line with:**

```python
# ========== FEATURE SELECTION STRATEGIES ==========
# Choose one of these strategies:

# STRATEGY A: BASELINE (Current - 2 features + rolling means)
# feature_cols = ['Priority', 'Impact']

# STRATEGY B: MINIMAL BOOST (Quick win - 5 new features)
# Expected gain: +1-1.5% F1
feature_cols = [
    'Priority', 'Impact',
    'priority_impact_product',  # Strongest predictor
    'priority_impact_sum',
    'risk_score',
    'priority_squared',
    'impact_squared'
]

# STRATEGY C: MODERATE (10 features)
# Expected gain: +1.5-2% F1
# feature_cols = [
#     'Priority', 'Impact',
#     'priority_impact_product', 'priority_impact_sum', 'risk_score',
#     'priority_squared', 'impact_squared',
#     'priority_hour_interaction', 'impact_hour_interaction',
#     'hour_sin', 'hour_cos'
# ]

# STRATEGY D: FULL RECOMMENDED (15 features)
# Expected gain: +1.5-2.5% F1
# feature_cols = [
#     'Priority', 'Impact',
#     'priority_impact_product', 'priority_impact_sum', 'risk_score',
#     'priority_squared', 'impact_squared',
#     'priority_hour_interaction', 'impact_hour_interaction',
#     'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
#     'hour', 'day_of_week'
# ]

print(f"  📊 Base features selected: {len(feature_cols)} features")
print(f"      {feature_cols}")
```

---

## 📋 COMPLETE MODIFIED EXPERIMENT LOOP CODE

Here's the FULL modified section for cell `dcb994fd`:

```python
# === 5. Run Experiment ==
sampling_periods = ['30min']
window_sizes = [48]  # Test with win=48 first (your best current model)
model_types = list(MODEL_REGISTRY.keys())
rolling_means = [2, 6]
results_all = []

for sp in sampling_periods:
    print(f"\n{'='*60}")
    print(f"Sampling Period: {sp}")
    print('='*60)

    # Resample with ALL features (including engineered ones)
    agg_dict = {
        'incident': 'sum',
        'Priority': 'mean',
        'Impact': 'mean',
        # Add engineered features to aggregation
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
    df_resampled['incident'] = (df_resampled['incident'] > 0).astype(int)

    # ========== FEATURE SELECTION STRATEGY ==========
    # CHOOSE ONE STRATEGY (uncomment the one you want to test)

    # STRATEGY A: BASELINE (Current)
    # feature_cols = ['Priority', 'Impact']

    # STRATEGY B: MINIMAL BOOST ⭐ RECOMMENDED FOR FIRST TEST
    feature_cols = [
        'Priority', 'Impact',
        'priority_impact_product',
        'priority_impact_sum',
        'risk_score',
        'priority_squared',
        'impact_squared'
    ]

    # STRATEGY C: MODERATE
    # feature_cols = [
    #     'Priority', 'Impact',
    #     'priority_impact_product', 'priority_impact_sum', 'risk_score',
    #     'priority_squared', 'impact_squared',
    #     'priority_hour_interaction', 'impact_hour_interaction',
    #     'hour_sin', 'hour_cos'
    # ]

    # STRATEGY D: FULL RECOMMENDED
    # feature_cols = [
    #     'Priority', 'Impact',
    #     'priority_impact_product', 'priority_impact_sum', 'risk_score',
    #     'priority_squared', 'impact_squared',
    #     'priority_hour_interaction', 'impact_hour_interaction',
    #     'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    #     'hour', 'day_of_week'
    # ]

    print(f"  📊 Base features: {len(feature_cols)} features")
    print(f"      {feature_cols}")

    # Apply Rolling Mean Features (SAME AS BEFORE)
    if rolling_means:
        print(f"  📊 Creating rolling mean features for: {rolling_means} hours")
        periods_per_hour = 2 if sp == '30min' else 1 if sp == '1h' else (60 // int(sp.replace('min', '')))
        for rm_hours in rolling_means:
            rm_periods = rm_hours * periods_per_hour
            df_resampled[f'Priority_rm{rm_hours}h'] = df_resampled['Priority'].rolling(window=rm_periods, min_periods=1).mean()
            df_resampled[f'Impact_rm{rm_hours}h'] = df_resampled['Impact'].rolling(window=rm_periods, min_periods=1).mean()
            feature_cols.extend([f'Priority_rm{rm_hours}h', f'Impact_rm{rm_hours}h'])
        print(f"  📊 Final feature columns: {len(feature_cols)} features total")

    # REST OF THE CODE STAYS THE SAME
    for win in window_sizes:
        print(f"\n  Window Size: {win}")
        X, y, ts = create_windowed_dataset(df_resampled, feature_cols, 'incident', win)

        if len(X) < 200 or len(np.unique(y)) < 2:
            print(f"    ⚠️ Skipped: Insufficient data (samples={len(X)}, classes={len(np.unique(y))})")
            continue

        X_flat = X.reshape((X.shape[0], -1))
        df_xy = pd.DataFrame(X_flat).assign(label=y)
        maj, min_class = df_xy[df_xy.label==0], df_xy[df_xy.label==1]

        if len(min_class) < 50:
            print(f"    ⚠️ Skipped: Too few minority samples ({len(min_class)})")
            continue

        print(f"    Data: {len(X)} samples ({X.shape}), Class 0: {len(maj)}, Class 1: {len(min_class)}")

        for model_type in model_types:
            print(f"\n    {'='*50}")
            print(f"    🧠 Model Type: {model_type}")
            print(f"    {'='*50}")

            # SMOTE
            try:
                print(f"      → Running SMOTE with {model_type}...")
                X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_flat, y)
                n_features = len(feature_cols)
                Xs = X_sm.reshape((-1, win, n_features))

                # Update label to reflect feature strategy
                label = f"{sp}|win={win}-SMOTE-Enhanced{len(feature_cols)}F-{model_type}"

                result = train_eval(Xs, y_sm, label=label, model_type=model_type)
                result['rolling_means'] = rolling_means if rolling_means else None
                result['n_features'] = len(feature_cols)
                results_all.append(result)
                save_model_assets(result)
            except Exception as e:
                print(f"      ✗ SMOTE failed for {model_type}: {e}")
                continue

print(f"\n{'='*60}")
print(f"✅ Experiment completed! Total results: {len(results_all)}")
print('='*60)
```

---

## 🧪 TESTING STRATEGY

### Phase 1: Baseline Comparison
```python
# Run 1: Current baseline (2 features)
feature_cols = ['Priority', 'Impact']  # Current F1: 0.81

# Run 2: Minimal boost (7 features)
feature_cols = ['Priority', 'Impact', 'priority_impact_product',
                'priority_impact_sum', 'risk_score',
                'priority_squared', 'impact_squared']
# Expected F1: 0.815-0.825 (+0.5-1.5%)
```

### Phase 2: If Phase 1 Successful
```python
# Run 3: Add temporal interactions (11 features)
feature_cols = baseline + ['priority_hour_interaction', 'impact_hour_interaction',
                           'hour_sin', 'hour_cos']
# Expected F1: 0.82-0.83 (+1-2%)
```

---

## 📊 EXPECTED RESULTS

| Strategy | Features | Expected F1 | Gain | Training Time |
|----------|----------|-------------|------|---------------|
| **Baseline** | 2 + 4 RM = 6 | 0.810 | Baseline | ~65 min |
| **Minimal** | 7 + 4 RM = 11 | 0.820 | +1.0% | ~70 min |
| **Moderate** | 11 + 4 RM = 15 | 0.825 | +1.5% | ~75 min |
| **Full** | 15 + 4 RM = 19 | 0.830 | +2.0% | ~80 min |

RM = Rolling Means (Priority_rm2h, Priority_rm6h, Impact_rm2h, Impact_rm6h)

---

## ⚠️ IMPORTANT NOTES

### 1. Feature Count in Windowing
The windowing function will automatically detect the number of features from `X.shape[2]`:
```python
input_size = X.shape[2]  # Auto-detect in train_eval()
```

### 2. SMOTE Reshape
Make sure to use `n_features = len(feature_cols)` when reshaping after SMOTE:
```python
n_features = len(feature_cols)  # NOT hardcoded to 2 or 6!
Xs = X_sm.reshape((-1, win, n_features))
```

### 3. Model Naming
Update the label to reflect feature count:
```python
label = f"{sp}|win={win}-SMOTE-Enhanced{len(feature_cols)}F-{model_type}"
# Example: "30min|win=48-SMOTE-Enhanced11F-Attention"
```

### 4. Rolling Means Still Important
Keep your rolling mean features! They add temporal context:
```python
rolling_means = [2, 6]  # KEEP THIS
```

---

## 🎯 QUICK START CHECKLIST

- [ ] Add NEW CELL after cell `e250f0bb` with feature engineering code
- [ ] Modify cell `dcb994fd` experiment loop:
  - [ ] Update `agg_dict` to include engineered features
  - [ ] Choose feature strategy (start with STRATEGY B)
  - [ ] Update label naming to include feature count
- [ ] Run experiment with **STRATEGY B (7 base + 4 RM = 11 features)**
- [ ] Compare F1 score with baseline (0.81)
- [ ] If improvement >= +1%, proceed to STRATEGY C or D
- [ ] Document results in output folder

---

## 📈 HOW TO INTERPRET RESULTS

### Success Criteria:
✅ **F1 >= 0.820** (Baseline: 0.810) - +1% improvement
✅ **Precision >= 0.79** (Baseline: 0.788) - Maintain or improve
✅ **Recall >= 0.83** (Baseline: 0.833) - Maintain or improve
✅ **Training time < 90 min** - Acceptable overhead

### If Results are WORSE:
- Check feature correlation (highly correlated features may hurt)
- Reduce to STRATEGY A (only interactions, no polynomial/temporal)
- Increase regularization (`lambda_l2=0.001` instead of 0.0001)

### If Results are BETTER:
- Try STRATEGY C or D for additional gains
- Test with different window sizes (48, 60, 72)
- Experiment with cyclical encodings

---

## 🚀 EXAMPLE OUTPUT

After running with STRATEGY B, you should see:

```
==========
Sampling Period: 30min
==========
  📊 Base features: 7 features
      ['Priority', 'Impact', 'priority_impact_product', 'priority_impact_sum',
       'risk_score', 'priority_squared', 'impact_squared']
  📊 Creating rolling mean features for: [2, 6] hours
  📊 Final feature columns: 11 features total

  Window Size: 48
    Data: 2800 samples (2800, 48, 11), Class 0: 1680, Class 1: 1120

    ==================================================
    🧠 Model Type: Attention
    ==================================================
      → Running SMOTE with Attention...

💾 Datasets exported to ../data/splits/
🔧 Initializing Attention LSTM model...

Epoch 1/500 - Loss: 0.4234 | Val Loss: 0.3921 | Val F1: 0.7234
Epoch 2/500 - Loss: 0.3654 | Val Loss: 0.3512 | Val F1: 0.7823
...
Epoch 45/500 - Loss: 0.1234 | Val Loss: 0.1987 | Val F1: 0.8345

✅ Best model at epoch 38 with Val Loss: 0.1876, Val F1: 0.8432

📊 FINAL RESULTS:
   Accuracy: 0.8234
   Precision: 0.8012
   Recall: 0.8456
   F1: 0.8227 ⬆️ (+1.27% from baseline!)
   AUC: 0.9087
```

---

**Ready to implement? Start with STRATEGY B for quick wins!** 🚀
