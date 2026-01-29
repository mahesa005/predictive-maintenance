# 📝 Optional Notebook Modifications for F1 Optimization

This guide provides **copy-paste code snippets** for each priority. Each modification is **optional** and can be enabled/disabled with flags.

---

## 🔧 Cell `e250f0bb` - Feature Engineering Extensions

Replace the entire cell with this version that includes **optional feature groups**:

```python
# === 2. Preprocess Data ===
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df[df['Type'].str.contains('restoration', case=False, na=False)]
df['Priority'] = pd.to_numeric(df['Priority'], errors='coerce')
df['Impact'] = pd.to_numeric(df['Impact'], errors='coerce')
df['incident'] = ((df['Priority'] >= 2) & (df['Impact'] >= 2)).astype(int)

# ============================================================
# CONFIGURATION: Enable/Disable Feature Groups
# ============================================================
ENABLE_TEMPORAL_FEATURES = True      # Base temporal (hour, day_of_week)
ENABLE_INTERACTION_FEATURES = True   # P×I combinations (current features)
ENABLE_SERVICE_FEATURES = True       # Priority 1: Service encoding
ENABLE_SEQUENCE_FEATURES = True      # Priority 2: Sequence dynamics
# ============================================================

print("\n🔧 Feature Engineering Configuration:")
print(f"   Temporal Features: {'✓ Enabled' if ENABLE_TEMPORAL_FEATURES else '✗ Disabled'}")
print(f"   Interaction Features: {'✓ Enabled' if ENABLE_INTERACTION_FEATURES else '✗ Disabled'}")
print(f"   Service Features: {'✓ Enabled' if ENABLE_SERVICE_FEATURES else '✗ Disabled'}")
print(f"   Sequence Features: {'✓ Enabled' if ENABLE_SEQUENCE_FEATURES else '✗ Disabled'}")

feature_count = 2  # Base: Priority, Impact

# === BASE: Temporal Features ===
if ENABLE_TEMPORAL_FEATURES:
    print("\n  → Creating temporal features...")
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    feature_count += 2

# === CURRENT: Interaction Features (P×I derivatives) ===
if ENABLE_INTERACTION_FEATURES:
    print("  → Creating interaction features...")

    # TIER 1: Interactions
    df['priority_impact_product'] = df['Priority'] * df['Impact']
    df['priority_impact_sum'] = df['Priority'] + df['Impact']
    df['risk_score'] = (0.6 * df['Priority']) + (0.4 * df['Impact'])

    # TIER 2: Polynomial
    df['priority_squared'] = df['Priority'] ** 2
    df['impact_squared'] = df['Impact'] ** 2

    # TIER 3: Temporal interactions
    if ENABLE_TEMPORAL_FEATURES:
        df['priority_hour_interaction'] = df['Priority'] * df['hour']
        df['impact_hour_interaction'] = df['Impact'] * df['hour']
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        feature_count += 11  # 5 + 6 temporal interactions
    else:
        feature_count += 5  # Just basic interactions

# === PRIORITY 1: Service Type Features (Target: +2-3% F1) ===
if ENABLE_SERVICE_FEATURES:
    print("  → Creating service type features...")

    # Target encoding: Encode by incident rate
    service_incident_rates = df.groupby('Service Type')['incident'].mean()
    df['service_risk_score'] = df['Service Type'].map(service_incident_rates)

    # Frequency encoding: How common is this service?
    service_freq = df['Service Type'].value_counts(normalize=True)
    df['service_frequency'] = df['Service Type'].map(service_freq)

    # Service Name encoding (higher granularity)
    name_incident_rates = df.groupby('Service Name')['incident'].mean()
    df['service_name_risk'] = df['Service Name'].map(name_incident_rates)

    # Handle NaN values (for rare services without historical data)
    df['service_risk_score'] = df['service_risk_score'].fillna(df['service_risk_score'].mean())
    df['service_frequency'] = df['service_frequency'].fillna(0)
    df['service_name_risk'] = df['service_name_risk'].fillna(df['service_name_risk'].mean())

    feature_count += 3
    print(f"     ✓ Added: service_risk_score, service_frequency, service_name_risk")

# === PRIORITY 2: Sequence-Aware Features (Target: +1-2% F1) ===
if ENABLE_SEQUENCE_FEATURES:
    print("  → Creating sequence-aware features...")

    # Volatility (instability over time)
    df['priority_volatility'] = df['Priority'].rolling(window=4, min_periods=1).std().fillna(0)
    df['impact_volatility'] = df['Impact'].rolling(window=4, min_periods=1).std().fillna(0)

    # Clip extreme volatility values
    df['priority_volatility'] = df['priority_volatility'].clip(0, 5)
    df['impact_volatility'] = df['impact_volatility'].clip(0, 5)

    # Change rate (direction/velocity)
    df['priority_change'] = df['Priority'].diff().fillna(0)
    df['impact_change'] = df['Impact'].diff().fillna(0)

    # Historical context
    df['incidents_last_2h'] = df['incident'].rolling(window=4, min_periods=1).sum()
    df['time_since_last_incident'] = (~df['incident'].astype(bool)).cumsum()
    df.loc[df['incident'] == 1, 'time_since_last_incident'] = 0

    # Consecutive patterns
    df['consecutive_high_priority'] = (df['Priority'] >= 2).rolling(window=6, min_periods=1).sum()

    # Acceleration (second derivative)
    priority_velocity = df['Priority'].diff()
    df['priority_acceleration'] = priority_velocity.diff().fillna(0)

    feature_count += 9
    print(f"     ✓ Added: volatility, change, context, consecutive patterns, acceleration")

print(f"\n✅ Feature engineering complete!")
print(f"   Total base features created: {feature_count}")
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {len(df.columns)}")
```

---

## 🔧 Cell `dcb994fd` - Experiment Configuration with Optional Strategies

Replace the entire cell with this version:

```python
# === 5. Run Experiment ===

# ============================================================
# CONFIGURATION: Select Feature Strategy
# ============================================================
# Choose ONE strategy based on your enabled feature groups:

# BASELINE STRATEGIES (No enhancements)
# FEATURE_STRATEGY = 'baseline'    # 2 base + rolling = 6 features

# INTERACTION-ONLY STRATEGIES (Current features)
# FEATURE_STRATEGY = 'quick_win'   # 7 base + rolling = 11 features
# FEATURE_STRATEGY = 'moderate'    # 11 base + rolling = 15 features
# FEATURE_STRATEGY = 'full'        # 15 base + rolling = 19 features

# PRIORITY 1: SERVICE-ENHANCED STRATEGIES
# FEATURE_STRATEGY = 'service_quick_win'   # 9 base + rolling = 13 features
# FEATURE_STRATEGY = 'service_moderate'    # 13 base + rolling = 17 features
# FEATURE_STRATEGY = 'service_full'        # 18 base + rolling = 22 features

# PRIORITY 2: SEQUENCE-ENHANCED STRATEGIES
# FEATURE_STRATEGY = 'sequence_quick_win'  # 9 base + rolling = 13 features
# FEATURE_STRATEGY = 'sequence_full'       # 17 base + rolling = 21 features

# PRIORITY 1+2: ULTIMATE STRATEGY (RECOMMENDED FOR F1 ≥ 0.85)
FEATURE_STRATEGY = 'ultimate'            # 21 base + rolling = 25 features ⭐

FEATURE_STRATEGIES = {
    # ========== BASELINE ==========
    'baseline': ['Priority', 'Impact'],

    # ========== INTERACTION-ONLY (Current) ==========
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

    # ========== PRIORITY 1: SERVICE-ENHANCED ==========
    'service_quick_win': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'service_risk_score', 'service_frequency'
    ],

    'service_moderate': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'service_risk_score', 'service_frequency'
    ],

    'service_full': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'day_sin', 'day_cos',
        'hour', 'day_of_week',
        'service_risk_score', 'service_frequency', 'service_name_risk'
    ],

    # ========== PRIORITY 2: SEQUENCE-ENHANCED ==========
    'sequence_quick_win': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_volatility', 'incidents_last_2h'
    ],

    'sequence_full': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'priority_volatility', 'impact_volatility',
        'priority_change', 'impact_change',
        'incidents_last_2h', 'consecutive_high_priority',
        'priority_acceleration'
    ],

    # ========== ULTIMATE: SERVICE + SEQUENCE (RECOMMENDED) ==========
    'ultimate': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'service_risk_score', 'service_frequency',          # Service features
        'priority_volatility', 'impact_volatility',         # Volatility
        'incidents_last_2h', 'consecutive_high_priority',   # Context
        'priority_change', 'priority_acceleration'          # Dynamics
    ]
}

# Validate strategy exists
if FEATURE_STRATEGY not in FEATURE_STRATEGIES:
    raise ValueError(f"Unknown strategy: {FEATURE_STRATEGY}. Available: {list(FEATURE_STRATEGIES.keys())}")

base_feature_cols = FEATURE_STRATEGIES[FEATURE_STRATEGY]

print(f"\n{'='*60}")
print(f"🎯 Feature Strategy: {FEATURE_STRATEGY}")
print(f"{'='*60}")
print(f"📊 Base features ({len(base_feature_cols)}): {base_feature_cols}\n")

# ============================================================
# PRIORITY 3: Model Selection (BiDirectional vs Attention)
# ============================================================
# Uncomment to enable BiDirectional Attention model
ENABLE_BIDIRECTIONAL = False  # Set to True for Priority 3

if ENABLE_BIDIRECTIONAL:
    print("✓ BiDirectional Attention model ENABLED (Priority 3)")
    from src.model.lstm_bidirectional_attention import BiDirectionalAttentionLSTMModelGPUOptimized
    MODEL_REGISTRY['BiDirectional-Attention'] = BiDirectionalAttentionLSTMModelGPUOptimized
else:
    print("  BiDirectional model disabled (using Attention only)")

# ============================================================
# Experiment Configuration
# ============================================================
sampling_periods = ['30min']
window_sizes = [48]  # Start with best window
model_types = list(MODEL_REGISTRY.keys())
rolling_means = [2, 6]
results_all = []

print(f"📋 Configuration:")
print(f"   Sampling: {sampling_periods}")
print(f"   Windows: {window_sizes}")
print(f"   Models: {model_types}")
print(f"   Rolling means: {rolling_means}")

# ============================================================
# Main Experiment Loop
# ============================================================
for sp in sampling_periods:
    print(f"\n{'='*60}")
    print(f"Sampling Period: {sp}")
    print('='*60)

    # Build aggregation dictionary for all features
    agg_dict = {'incident': 'sum'}
    for feature in base_feature_cols:
        if feature in ['hour', 'day_of_week']:
            # Use 'first' for temporal features (safer than mode with empty buckets)
            agg_dict[feature] = 'first'
        else:
            agg_dict[feature] = 'mean'

    # Resample with all features
    df_resampled = (df.set_index('Timestamp')
                      .resample(sp)
                      .agg(agg_dict)
                      .fillna(0)
                      .reset_index())
    df_resampled['incident'] = (df_resampled['incident'] > 0).astype(int)

    # Start with base feature columns
    feature_cols = base_feature_cols.copy()

    # Apply Rolling Mean Features (AFTER resampling, BEFORE windowing)
    if rolling_means:
        print(f"  📊 Creating rolling mean features for: {rolling_means} hours")

        # Calculate periods based on sampling period
        periods_per_hour = 2 if sp == '30min' else 1 if sp == '1h' else (60 // int(sp.replace('min', '')))
        for rm_hours in rolling_means:
            rm_periods = rm_hours * periods_per_hour
            df_resampled[f'Priority_rm{rm_hours}h'] = df_resampled['Priority'].rolling(window=rm_periods, min_periods=1).mean()
            df_resampled[f'Impact_rm{rm_hours}h'] = df_resampled['Impact'].rolling(window=rm_periods, min_periods=1).mean()
            feature_cols.extend([f'Priority_rm{rm_hours}h', f'Impact_rm{rm_hours}h'])

        n_total_features = len(feature_cols)
        n_rolling_features = len(rolling_means) * 2
        print(f"  📊 Total features: {n_total_features} ({len(base_feature_cols)} base + {n_rolling_features} rolling)")
        print(f"  📊 Feature columns: {feature_cols}")

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

        print(f"    Data: {len(X)} samples, Class 0: {len(maj)}, Class 1: {len(min_class)}")
        print(f"    Shape: X={X.shape} (samples, window, features)")

        # === Loop through all model types ===
        for model_type in model_types:
            print(f"\n    {'='*50}")
            print(f"    🧠 Model Type: {model_type}")
            print(f"    {'='*50}")

            # SMOTE
            try:
                print(f"      → Running SMOTE with {model_type}...")
                X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_flat, y)

                # Reshape back to 3D with correct number of features
                n_features = len(feature_cols)
                Xs = X_sm.reshape((-1, win, n_features))

                # Create label with feature count
                label = f"{sp}|win={win}|f={n_features}-SMOTE-{model_type}"

                print(f"      → Training with {n_features} features (strategy: {FEATURE_STRATEGY})...")
                result = train_eval(Xs, y_sm, label=label, model_type=model_type)

                # Add metadata
                result['feature_strategy'] = FEATURE_STRATEGY
                result['n_base_features'] = len(base_feature_cols)
                result['n_total_features'] = n_features
                result['rolling_means'] = rolling_means if rolling_means else None

                results_all.append(result)
                save_model_assets(result)
            except Exception as e:
                print(f"      ✗ SMOTE failed for {model_type}: {e}")
                import traceback
                traceback.print_exc()
                continue


print(f"\n{'='*60}")
print(f"✅ Experiment completed! Total results: {len(results_all)}")
print(f"   Feature Strategy: {FEATURE_STRATEGY}")
print(f"   Base Features: {len(base_feature_cols)}")
print(f"   Total Features (with rolling): {len(feature_cols)}")
print(f"   Models tested: {model_types}")
print('='*60)
```

---

## 🔧 NEW Cell - Priority 4: Ensemble (Add After Results Cell)

Add this as a **new cell** after the results visualization cell:

```python
# ============================================================
# PRIORITY 4: MODEL ENSEMBLE (Optional)
# ============================================================
# Set to True to create ensemble from trained models
CREATE_ENSEMBLE = False  # Change to True to enable

if CREATE_ENSEMBLE:
    print("\n" + "="*60)
    print("🔀 CREATING MODEL ENSEMBLE")
    print("="*60)

    # ========== CONFIGURATION: Select Models to Ensemble ==========
    # List your trained models (update paths based on your runs)
    ENSEMBLE_MODELS = [
        '../data/output/30min_win48f22-SMOTE-Attention',              # Service+Sequence+Attention
        '../data/output/30min_win48f22-SMOTE-BiDirectional-Attention',  # Service+Sequence+BiDi
        '../data/output/30min_win60f22-SMOTE-Attention',              # Service+Sequence+Attention (win60)
    ]

    # Verify models exist
    import os
    valid_models = []
    for model_path in ENSEMBLE_MODELS:
        prob_file = os.path.join(model_path, f"y_prob_{os.path.basename(model_path)}.npy")
        if os.path.exists(prob_file):
            valid_models.append(model_path)
        else:
            print(f"⚠️ Warning: Model not found: {prob_file}")

    if len(valid_models) < 2:
        print("❌ Need at least 2 trained models for ensemble. Skipping...")
    else:
        print(f"\n✓ Found {len(valid_models)} models for ensemble:")
        for i, model in enumerate(valid_models, 1):
            print(f"  {i}. {os.path.basename(model)}")

        # Load predictions
        y_probs = []
        for model_path in valid_models:
            prob_file = os.path.join(model_path, f"y_prob_{os.path.basename(model_path)}.npy")
            y_prob = np.load(prob_file)
            y_probs.append(y_prob)

        # Load ground truth (same across all models)
        y_true_file = os.path.join(valid_models[0], f"y_test_{os.path.basename(valid_models[0])}.npy")
        y_true = np.load(y_true_file)

        # ========== ENSEMBLE STRATEGIES ==========
        print(f"\n{'='*60}")
        print("Testing ensemble strategies...")
        print('='*60)

        # Strategy 1: Simple Average
        y_prob_avg = np.mean(y_probs, axis=0)

        # Strategy 2: Weighted Average (weight by individual F1 scores)
        individual_f1s = []
        for i, y_prob in enumerate(y_probs):
            y_pred = (y_prob >= 0.5).astype(int)
            f1 = f1_score(y_true, y_pred)
            individual_f1s.append(f1)
            print(f"Model {i+1} F1: {f1:.4f}")

        weights = np.array(individual_f1s) / sum(individual_f1s)
        y_prob_weighted = np.average(y_probs, axis=0, weights=weights)

        # Find optimal thresholds
        print(f"\n{'='*60}")
        print("Finding optimal thresholds...")
        print('='*60)

        results_ensemble = []

        for strategy, y_prob_ens, name in [
            ('average', y_prob_avg, 'Simple Average'),
            ('weighted', y_prob_weighted, 'Weighted Average')
        ]:
            # Find optimal threshold
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob_ens)
            f1_scores_pr = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
            opt_idx = np.argmax(f1_scores_pr)
            opt_threshold = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5

            # Predictions
            y_pred_ens = (y_prob_ens >= opt_threshold).astype(int)

            # Metrics
            result_ens = {
                'Strategy': name,
                'Threshold': opt_threshold,
                'Accuracy': accuracy_score(y_true, y_pred_ens),
                'Precision': precision_score(y_true, y_pred_ens),
                'Recall': recall_score(y_true, y_pred_ens),
                'F1': f1_score(y_true, y_pred_ens)
            }
            results_ensemble.append(result_ens)

            print(f"\n{name}:")
            print(f"  Threshold: {opt_threshold:.3f}")
            print(f"  Accuracy:  {result_ens['Accuracy']:.4f}")
            print(f"  Precision: {result_ens['Precision']:.4f}")
            print(f"  Recall:    {result_ens['Recall']:.4f}")
            print(f"  F1 Score:  {result_ens['F1']:.4f}")

        # Compare with best individual
        best_individual_f1 = max(individual_f1s)
        best_ensemble_f1 = max([r['F1'] for r in results_ensemble])

        print(f"\n{'='*60}")
        print("ENSEMBLE SUMMARY")
        print('='*60)
        print(f"Best Individual F1: {best_individual_f1:.4f}")
        print(f"Best Ensemble F1:   {best_ensemble_f1:.4f}")
        print(f"Improvement:        +{(best_ensemble_f1 - best_individual_f1)*100:.2f}%")

        if best_ensemble_f1 > best_individual_f1:
            print(f"\n✅ Ensemble improves performance!")
        else:
            print(f"\n⚠️ Ensemble does not improve (models may be too similar)")

        print('='*60)

else:
    print("\n[INFO] Ensemble creation disabled (CREATE_ENSEMBLE = False)")
    print("       Set CREATE_ENSEMBLE = True after training multiple models to enable")
```

---

## 📋 Quick Configuration Guide

### For F1 = 0.82 (Current)
```python
# Cell e250f0bb:
ENABLE_TEMPORAL_FEATURES = True
ENABLE_INTERACTION_FEATURES = True
ENABLE_SERVICE_FEATURES = False
ENABLE_SEQUENCE_FEATURES = False

# Cell dcb994fd:
FEATURE_STRATEGY = 'full'
ENABLE_BIDIRECTIONAL = False
```

### For F1 = 0.84-0.85 (Priority 1)
```python
# Cell e250f0bb:
ENABLE_TEMPORAL_FEATURES = True
ENABLE_INTERACTION_FEATURES = True
ENABLE_SERVICE_FEATURES = True      # ← Enable
ENABLE_SEQUENCE_FEATURES = False

# Cell dcb994fd:
FEATURE_STRATEGY = 'service_full'
ENABLE_BIDIRECTIONAL = False
```

### For F1 = 0.85-0.86 (Priority 1 + 2)
```python
# Cell e250f0bb:
ENABLE_TEMPORAL_FEATURES = True
ENABLE_INTERACTION_FEATURES = True
ENABLE_SERVICE_FEATURES = True      # ← Enable
ENABLE_SEQUENCE_FEATURES = True     # ← Enable

# Cell dcb994fd:
FEATURE_STRATEGY = 'ultimate'
ENABLE_BIDIRECTIONAL = False
```

### For F1 = 0.86-0.87 (Priority 1 + 2 + 3)
```python
# Cell e250f0bb:
ENABLE_TEMPORAL_FEATURES = True
ENABLE_INTERACTION_FEATURES = True
ENABLE_SERVICE_FEATURES = True
ENABLE_SEQUENCE_FEATURES = True

# Cell dcb994fd:
FEATURE_STRATEGY = 'ultimate'
ENABLE_BIDIRECTIONAL = True         # ← Enable
```

### For F1 = 0.87+ (All Priorities)
Same as above, plus run ensemble cell with `CREATE_ENSEMBLE = True`

---

## ✅ Validation Checklist

After modifying the notebook:

### Cell `e250f0bb` Validation
- [ ] Configuration flags at top
- [ ] Service features code added (if enabled)
- [ ] Sequence features code added (if enabled)
- [ ] No syntax errors
- [ ] Print statements show enabled features

### Cell `dcb994fd` Validation
- [ ] New strategies added to dictionary
- [ ] `ENABLE_BIDIRECTIONAL` flag added
- [ ] Strategy selector updated
- [ ] No duplicate keys in dictionary

### Ensemble Cell Validation (if added)
- [ ] New cell created after results
- [ ] `CREATE_ENSEMBLE` flag at top
- [ ] Model paths updated to your runs
- [ ] Ensemble logic present

---

## 🔄 Rollback Instructions

If you want to revert to baseline:

```python
# Cell e250f0bb:
ENABLE_SERVICE_FEATURES = False
ENABLE_SEQUENCE_FEATURES = False

# Cell dcb994fd:
FEATURE_STRATEGY = 'full'  # or 'quick_win'
ENABLE_BIDIRECTIONAL = False

# Ensemble cell:
CREATE_ENSEMBLE = False
```

This returns you to the current F1 = 0.82 baseline.

---

**Ready to implement!** Choose your target F1 and set the flags accordingly.
