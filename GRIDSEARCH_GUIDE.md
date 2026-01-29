# 🔍 Feature Grid Search Guide

## Overview

This guide shows you how to use the grid search function to automatically test all feature combinations and find the optimal configuration for your LSTM model.

## Quick Start

### Option 1: Search All Feature Strategies (Default)

```python
# Run grid search on all interaction-based strategies
results = run_feature_gridsearch(
    feature_strategies=['baseline', 'quick_win', 'moderate', 'full'],
    enable_service=False,
    enable_sequence=False,
    window_sizes=[48],
    sampling_periods=['30min'],
    rolling_means_configs=[[2, 6]],
    model_types=['Attention']
)
```

**Expected time:** ~4.5 hours (4 strategies × 70 min each)

### Option 2: Quick Test (2 strategies)

```python
# Compare baseline vs quick_win only
results = run_feature_gridsearch(
    feature_strategies=['baseline', 'quick_win'],
    window_sizes=[48],
    sampling_periods=['30min']
)
```

**Expected time:** ~2.5 hours

### Option 3: Service Feature Exploration

```python
# Test all service-enhanced strategies
results = run_feature_gridsearch(
    feature_strategies=['service_quick_win', 'service_moderate', 'service_full'],
    enable_service=True,  # Must be True for service strategies!
    enable_sequence=False,
    window_sizes=[48],
    sampling_periods=['30min']
)
```

**Expected time:** ~3.5 hours (3 strategies)

### Option 4: Ultimate Grid Search (All Combinations)

```python
# Test EVERYTHING (use with caution - very long!)
results = run_feature_gridsearch(
    feature_strategies='all',  # Tests all 10 strategies
    enable_service=True,
    enable_sequence=True,
    window_sizes=[48, 60, 72],
    sampling_periods=['30min'],
    rolling_means_configs=[[2, 6], [1, 3, 6]]
)
```

**Expected time:** ~50+ hours (10 strategies × 3 windows × 2 rolling configs)

## Grid Search Parameters

### 1. `feature_strategies` (list or 'all')
Strategies to test:
- **Interaction-only**: `'baseline'`, `'quick_win'`, `'moderate'`, `'full'`
- **Service-enhanced**: `'service_quick_win'`, `'service_moderate'`, `'service_full'`
- **Sequence-enhanced**: `'sequence_quick_win'`, `'sequence_full'`
- **Combined**: `'ultimate'`

Use `'all'` to test all 10 strategies.

### 2. `enable_service` (bool)
- `True`: Enables service feature engineering (Priority 1)
- `False`: Disables service features
- **Important**: Must be `True` to use `service_*` or `ultimate` strategies

### 3. `enable_sequence` (bool)
- `True`: Enables sequence-aware features (Priority 2)
- `False`: Disables sequence features
- **Important**: Must be `True` to use `sequence_*` or `ultimate` strategies

### 4. `window_sizes` (list)
Time steps to look back:
- `[48]`: 24 hours at 30min sampling (recommended)
- `[48, 60, 72]`: Test multiple window sizes

### 5. `sampling_periods` (list)
Temporal granularity:
- `['30min']`: 30-minute buckets (recommended)
- `['30min', '1h']`: Test both

### 6. `rolling_means_configs` (list of lists)
Rolling mean window sizes (in hours):
- `[[2, 6]]`: Default (2h and 6h rolling means)
- `[[1, 3, 6]]`: Alternative configuration
- `[[2, 6], [1, 3, 6]]`: Test both configurations

### 7. `model_types` (list)
LSTM architectures to test:
- `['Attention']`: Default (recommended)
- `['Attention', 'BiDirectional-Attention']`: Test both

### 8. `enable_bidirectional` (bool)
- `True`: Adds BiDirectional-Attention to available models (Priority 3)
- `False`: Only use existing models

### 9. `verbose` (int)
Logging level:
- `0`: Minimal output (just progress)
- `1`: Standard output (default)
- `2`: Detailed output (debugging)

## Analyzing Results

### 1. View Top Configurations

```python
# Show best 10 configurations by F1 score
print(results[['Strategy', 'Window', 'n_features', 'F1_Score', 'Precision', 'Recall', 'AUC']].head(10))
```

### 2. Compare by Strategy

```python
# Group by strategy and show average metrics
strategy_comparison = results.groupby('Strategy')[['F1_Score', 'Precision', 'Recall', 'AUC', 'TrainingTime']].agg(['mean', 'std'])
print(strategy_comparison)
```

### 3. Find Pareto Optimal Configurations

```python
# Find configurations with F1 >= 0.82 and TrainingTime < 90 min
efficient = results[(results['F1_Score'] >= 0.82) & (results['TrainingTime'] < 90)]
print(efficient[['Strategy', 'n_features', 'F1_Score', 'TrainingTime']].sort_values('F1_Score', ascending=False))
```

### 4. Visualize Feature Impact

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot F1 score vs number of features
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results, x='n_features', y='F1_Score', hue='Strategy', s=100)
plt.title('Feature Count vs F1 Score')
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.grid(alpha=0.3)
plt.show()
```

### 5. Feature Strategy Comparison

```python
# Bar chart comparing strategies
strategy_avg = results.groupby('Strategy')['F1_Score'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
strategy_avg.plot(kind='bar', color='steelblue')
plt.title('Average F1 Score by Feature Strategy')
plt.ylabel('F1 Score')
plt.xlabel('Strategy')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6. Export Results

```python
# Save to CSV for further analysis
results.to_csv('../data/gridsearch_results.csv', index=False)
print("Results saved to: data/gridsearch_results.csv")
```

## Use Cases

### Use Case 1: Find Best Baseline Configuration
**Goal:** Optimize current features before adding new ones

```python
results = run_feature_gridsearch(
    feature_strategies=['baseline', 'quick_win', 'moderate', 'full'],
    window_sizes=[48, 60, 72],
    rolling_means_configs=[[2, 6], [1, 3, 6]],
    sampling_periods=['30min']
)

best_baseline = results.loc[results['F1_Score'].idxmax()]
print(f"\nBest configuration:")
print(f"  Strategy: {best_baseline['Strategy']}")
print(f"  Window: {best_baseline['Window']}")
print(f"  Rolling Means: {best_baseline['rolling_means']}")
print(f"  F1 Score: {best_baseline['F1_Score']:.4f}")
```

### Use Case 2: Validate Service Features Impact
**Goal:** Measure F1 improvement from service features

```python
# Test without service features
results_baseline = run_feature_gridsearch(
    feature_strategies=['quick_win', 'moderate', 'full'],
    enable_service=False
)

# Test with service features
results_service = run_feature_gridsearch(
    feature_strategies=['service_quick_win', 'service_moderate', 'service_full'],
    enable_service=True
)

# Compare
baseline_f1 = results_baseline['F1_Score'].max()
service_f1 = results_service['F1_Score'].max()
improvement = (service_f1 - baseline_f1) / baseline_f1 * 100

print(f"\nService Features Impact:")
print(f"  Best F1 without service: {baseline_f1:.4f}")
print(f"  Best F1 with service: {service_f1:.4f}")
print(f"  Improvement: +{improvement:.2f}%")
```

### Use Case 3: Test Priority Roadmap Sequentially
**Goal:** Validate expected gains from each priority

```python
# Priority 0: Baseline
p0 = run_feature_gridsearch(feature_strategies=['quick_win'])
f1_p0 = p0['F1_Score'].max()

# Priority 1: Add service features
p1 = run_feature_gridsearch(
    feature_strategies=['service_full'],
    enable_service=True
)
f1_p1 = p1['F1_Score'].max()

# Priority 2: Add sequence features
p2 = run_feature_gridsearch(
    feature_strategies=['sequence_full'],
    enable_service=True,
    enable_sequence=True
)
f1_p2 = p2['F1_Score'].max()

# Priority 3: Add BiDirectional
p3 = run_feature_gridsearch(
    feature_strategies=['ultimate'],
    enable_service=True,
    enable_sequence=True,
    enable_bidirectional=True,
    model_types=['BiDirectional-Attention']
)
f1_p3 = p3['F1_Score'].max()

print("\n📊 Priority Roadmap Results:")
print(f"  P0 (Baseline - quick_win):     F1 = {f1_p0:.4f}")
print(f"  P1 (+ Service features):       F1 = {f1_p1:.4f} (+{(f1_p1-f1_p0)*100:.1f}%)")
print(f"  P2 (+ Sequence features):      F1 = {f1_p2:.4f} (+{(f1_p2-f1_p1)*100:.1f}%)")
print(f"  P3 (+ BiDirectional):          F1 = {f1_p3:.4f} (+{(f1_p3-f1_p2)*100:.1f}%)")
print(f"\n  Total improvement: +{(f1_p3-f1_p0)*100:.1f}%")
```

### Use Case 4: Hyperparameter Tuning
**Goal:** Find optimal window size and rolling mean configuration

```python
results = run_feature_gridsearch(
    feature_strategies=['quick_win'],  # Fix strategy
    window_sizes=[24, 36, 48, 60, 72],  # Test many windows
    rolling_means_configs=[[1, 3], [2, 6], [1, 3, 6], [2, 4, 8]],  # Test many configs
    sampling_periods=['30min']
)

# Find best hyperparameters
best = results.loc[results['F1_Score'].idxmax()]
print(f"\nOptimal Hyperparameters:")
print(f"  Window Size: {best['Window']}")
print(f"  Rolling Means: {best['rolling_means']}")
print(f"  F1 Score: {best['F1_Score']:.4f}")

# Visualize window size impact
import matplotlib.pyplot as plt
window_avg = results.groupby('Window')['F1_Score'].mean()
plt.figure(figsize=(10, 6))
window_avg.plot(kind='line', marker='o', linewidth=2, markersize=8)
plt.title('Window Size Impact on F1 Score')
plt.xlabel('Window Size (timesteps)')
plt.ylabel('Average F1 Score')
plt.grid(alpha=0.3)
plt.show()
```

## Tips for Efficient Grid Search

1. **Start Small**: Test 2-3 strategies first before running full grid search
2. **Fix Hyperparameters**: When testing features, keep window_size and rolling_means constant
3. **Use Verbose Mode**: Set `verbose=2` for first run to catch errors early
4. **Save Intermediate Results**: Results are automatically saved after each iteration
5. **Monitor Progress**: Check output folder to see completed experiments
6. **Run Overnight**: Full grid searches can take 12+ hours

## Troubleshooting

### Grid Search Takes Too Long
- Reduce strategies to test
- Use fewer window sizes: `window_sizes=[48]`
- Test one rolling mean config: `rolling_means_configs=[[2, 6]]`
- Reduce epochs: Modify `train_eval()` to use `epochs=300` instead of 500

### Out of Memory Errors
- Reduce batch size in `train_eval()`: `batch_size=64`
- Test fewer strategies at once
- Close other GPU applications

### Some Strategies Fail
- Check that required feature flags are enabled
- Verify service/sequence strategies require their respective flags
- Check error messages in verbose output

## Advanced: Custom Grid Search

```python
# Define your own parameter grid
from itertools import product

custom_grid = {
    'strategy': ['quick_win', 'service_full'],
    'window': [48, 60],
    'rolling': [[2, 6], [1, 3, 6]],
    'hidden_size': [128, 256],  # New parameter!
    'lr': [0.001, 0.0005]  # New parameter!
}

results_custom = []

for strategy, win, rm, hs, lr in product(
    custom_grid['strategy'],
    custom_grid['window'],
    custom_grid['rolling'],
    custom_grid['hidden_size'],
    custom_grid['lr']
):
    print(f"\nTesting: {strategy} | win={win} | rm={rm} | hs={hs} | lr={lr}")

    # Prepare data with this configuration
    # ... (data preparation code)

    # Train with custom hyperparameters
    result = train_eval(
        Xs, y_sm,
        label=f"30min|win={win}|hs={hs}|lr={lr}-{strategy}",
        hidden_size=hs,
        lr=lr
    )

    result['strategy'] = strategy
    result['window'] = win
    result['hidden_size'] = hs
    result['learning_rate'] = lr

    results_custom.append(result)

df_custom = pd.DataFrame(results_custom)
print(df_custom.sort_values('F1_Score', ascending=False))
```

## Summary

The grid search function helps you:
- ✅ Test all feature strategies systematically
- ✅ Find optimal hyperparameter configurations
- ✅ Validate expected gains from roadmap priorities
- ✅ Compare feature impact objectively
- ✅ Save time with automated experimentation
- ✅ Make data-driven decisions on feature selection

**Recommended first run:**
```python
results = run_feature_gridsearch(
    feature_strategies=['baseline', 'quick_win', 'moderate'],
    window_sizes=[48],
    sampling_periods=['30min'],
    verbose=1
)
```

This will give you a good comparison between your current baseline and enhanced features in ~3-4 hours.
