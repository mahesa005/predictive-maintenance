# 📊 COMPREHENSIVE EDA ANALYSIS REPORT
## Advanced Non-Linear Correlation Analysis for Predictive Maintenance

**Generated:** 2026-01-27
**Dataset:** Restoration Tickets (January 2024)
**Analysis Method:** Mutual Information, Spearman, Pearson Correlations + K-Means Clustering

---

## 📈 EXECUTIVE SUMMARY

### Dataset Overview
- **Total Restoration Tickets:** 18,969 (filtered from 39,259 total records)
- **Incident Rate:** 41.29% (7,833 incidents / 11,136 non-incidents)
- **Features Created:** 47 total (17 temporal + 18 interaction + 12 original)
- **Recommended Features:** 20 high-impact features for LSTM training

### Key Finding
**The relationships between features and incidents are predominantly LINEAR/MONOTONIC**, not highly non-linear. This means:
- ✅ Simpler models may perform well
- ✅ Feature interactions (Priority × Impact) are more important than complex transformations
- ✅ LSTM will benefit from temporal sequences, but non-linearity is moderate

---

## 🎯 TOP 10 FEATURES BY MUTUAL INFORMATION

| Rank | Feature | MI Score | Spearman | Pearson | Type | Insight |
|------|---------|----------|----------|---------|------|---------|
| 1 | **priority_impact_product** | **0.6770** | 0.9275 | 0.8065 | Interaction | 🔥 STRONGEST predictor - multiplicative effect |
| 2 | **priority_impact_sum** | **0.6755** | 0.9269 | 0.8866 | Interaction | 🔥 Combined severity indicator |
| 3 | **risk_score** | **0.6752** | 0.8768 | 0.8823 | Composite | 🔥 Weighted composite score |
| 4 | **priority_squared** | **0.5643** | 0.9166 | 0.7555 | Polynomial | Non-linear Priority transformation |
| 5 | **Priority** | **0.5640** | 0.9166 | 0.8520 | Core | Base Priority level |
| 6 | **Impact** | **0.5193** | 0.8898 | 0.8369 | Core | Base Impact level |
| 7 | **impact_squared** | **0.5182** | 0.8898 | 0.7419 | Polynomial | Non-linear Impact transformation |
| 8 | **priority_hour_interaction** | **0.4016** | 0.7026 | 0.6472 | Temporal | Priority × Time interaction |
| 9 | **impact_hour_interaction** | **0.3673** | 0.6744 | 0.6239 | Temporal | Impact × Time interaction |
| 10 | **log_duration** | **0.0238** | 0.1809 | 0.1491 | Transform | Log-transformed duration |

### 💡 Key Insights:
1. **Interaction features dominate** - Priority × Impact multiplicative effect is THE strongest signal
2. **Polynomial features help** - Squared terms capture some non-linearity
3. **Temporal interactions matter** - Hour interactions with Priority/Impact are important
4. **Duration features are weak** - Log/sqrt transformations don't add much predictive power
5. **Cyclical encodings have LOW MI** - Hour sin/cos features contribute minimally

---

## ⏰ TEMPORAL PATTERN ANALYSIS

### 🔴 CRITICAL FINDINGS

#### **1. Peak Incident Hour: 1:00 AM (56.67% incident rate)**
- **Night shift (00:00-06:00) has HIGHEST risk: 50.70% incident rate**
- Morning shift: Lower risk
- Afternoon shift: Moderate risk
- Evening shift: Moderate risk

#### **2. Business Hours vs Non-Business Hours**
- **Non-business hours: 44.64%** incident rate ⬆️ HIGHER
- **Business hours: 39.82%** incident rate ⬇️ LOWER
- **Implication:** Off-hours tickets are MORE critical

#### **3. Weekend vs Weekday**
- **Weekend: 45.01%** incident rate ⬆️ HIGHER
- **Weekday: 40.73%** incident rate ⬇️ LOWER
- **Implication:** Weekend coverage needs improvement

#### **4. Lowest Risk Period**
- **8:00 AM has LOWEST incident rate: 34.91%**
- Morning shift (06:00-12:00) generally safer

### 📊 Temporal Distribution
```
Hour Range          Incident Rate    Risk Level
00:00 - 06:00      50.70%           🔴 CRITICAL (Night)
06:00 - 12:00      ~40-45%          🟡 MODERATE (Morning)
12:00 - 18:00      ~42-46%          🟠 ELEVATED (Afternoon)
18:00 - 24:00      ~45-48%          🟠 ELEVATED (Evening)
```

---

## 🎨 CLUSTERING ANALYSIS

**Optimal Clusters:** 2 (Silhouette Score: 0.5480 - GOOD separation)

### Cluster 0: [LOW RISK] - 11,048 tickets (58.2%)
- **Incident Rate:** 0.00% ✅ (NO incidents)
- **Avg Priority:** 1.06 (Very Low)
- **Avg Impact:** 1.09 (Very Low)
- **Avg Hour:** 11.4 (Late morning)
- **Weekend %:** 12.1% (Mostly weekday)
- **Business Hours:** 71.7% (Mostly during work hours)
- **Profile:** Routine, low-severity tickets handled during business hours

### Cluster 1: [CRITICAL] - 7,921 tickets (41.8%)
- **Incident Rate:** 98.89% 🔴 (ALMOST ALL incidents)
- **Avg Priority:** 2.22 (High)
- **Avg Impact:** 2.14 (High)
- **Avg Hour:** 12.0 (Midday)
- **Weekend %:** 14.4% (Slightly more weekend)
- **Business Hours:** 66.2% (Still mostly business hours)
- **Profile:** High-severity issues requiring immediate attention

### 🎯 Clustering Insight
**PERFECT BINARY SEPARATION!** The clustering algorithm almost perfectly separates incidents from non-incidents based on Priority/Impact patterns. This confirms:
- ✅ Priority ≥ 2 AND Impact ≥ 2 is an EXCELLENT incident definition
- ✅ The feature space is well-structured for classification
- ✅ LSTM should achieve high accuracy with these features

---

## 🧪 NON-LINEARITY ANALYSIS

### Finding: **NO HIGHLY NON-LINEAR FEATURES DETECTED**

**What this means:**
- Most relationships are **monotonic** (as feature increases, incident probability increases/decreases consistently)
- Spearman correlations are HIGH (0.7-0.9 for top features)
- Pearson correlations are also HIGH (0.6-0.9 for top features)
- NonLinear_Ratio < 1.5 for all features

### Implications for Modeling:
1. ✅ **Linear/Logistic models will perform reasonably well** as a baseline
2. ✅ **LSTMs will excel at temporal patterns**, not complex non-linearity
3. ✅ **Feature interactions are MORE important than non-linear transformations**
4. ⚠️ **Don't over-engineer features** - simple interactions work best

---

## 📋 RECOMMENDED FEATURE LIST (20 Features)

### Tier 1: MUST INCLUDE (MI > 0.35)
1. **priority_impact_product** (MI: 0.677) - Priority × Impact
2. **priority_impact_sum** (MI: 0.676) - Priority + Impact
3. **risk_score** (MI: 0.675) - Composite risk metric
4. **priority_squared** (MI: 0.564) - Priority²
5. **Priority** (MI: 0.564) - Base Priority
6. **Impact** (MI: 0.519) - Base Impact
7. **impact_squared** (MI: 0.518) - Impact²
8. **priority_hour_interaction** (MI: 0.402) - Priority × Hour
9. **impact_hour_interaction** (MI: 0.367) - Impact × Hour

### Tier 2: STRONG SUPPORT (MI: 0.01-0.03)
10. **sla_pressure** (MI: 0.029) - SLA breach risk indicator
11. **log_duration** (MI: 0.024) - Log(resolved duration)
12. **sqrt_duration** (MI: 0.020) - √(resolved duration)
13. **month** (MI: 0.016) - Seasonal pattern
14. **week_of_year** (MI: 0.015) - Weekly pattern

### Tier 3: CYCLICAL & CATEGORICAL (MI < 0.01)
15. **day_sin** (MI: 0.006) - Day of week (sin)
16. **sla_breach** (MI: 0.004) - SLA breach flag
17. **hour_sin** (MI: 0.003) - Hour (sin encoding)
18. **hour_cos** (MI: 0.003) - Hour (cos encoding)
19. **is_business_hours** (MI: 0.000) - Business hours flag
20. **day_cos** (MI: 0.000) - Day of week (cos)

### ⚠️ NOT RECOMMENDED (Very Low MI):
- `priority_impact_diff` (MI: 0.001)
- `is_peak_hours` (MI: 0.002)
- Raw `hour`, `shift`, `day_of_week` (use cyclical encoding instead)

---

## 🚀 ACTIONABLE RECOMMENDATIONS

### For Model Training:

#### 1. Feature Selection Strategy
```python
# Core features (ALWAYS include)
core_features = [
    'Priority', 'Impact',
    'priority_impact_product', 'priority_impact_sum',
    'priority_squared', 'impact_squared',
    'risk_score'
]

# Temporal features (Important for LSTM)
temporal_features = [
    'hour', 'day_of_week', 'shift',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_business_hours', 'is_weekend', 'is_peak_hours'
]

# Interaction features (Strong predictors)
interaction_features = [
    'priority_hour_interaction',
    'impact_hour_interaction',
    'priority_shift_interaction'
]

# Duration features (Weak but might help)
duration_features = [
    'log_duration', 'sqrt_duration'
]

# SLA features (Operational context)
sla_features = [
    'sla_pressure', 'sla_breach', 'has_sla'
]

# FINAL FEATURE SET = core + temporal + interaction + sla (17-20 features)
```

#### 2. Preprocessing Pipeline
```python
# 1. Load feature-engineered dataset
df = pd.read_csv('data/feature_engineered_dataset.csv')

# 2. Select recommended features
features_to_use = [list from recommended_features.txt top 15]

# 3. Create windowed sequences (as you currently do)
window_sizes = [24, 48, 72, 96]  # Test different temporal windows

# 4. Apply SMOTE (class imbalance is significant: 41% vs 59%)

# 5. Train LSTM with enhanced features
```

#### 3. Model Comparison Experiments
Run these experiments to validate feature importance:

**Experiment A: Baseline**
- Features: Priority, Impact only (2 features)
- Expected F1: ~0.75-0.80

**Experiment B: Baseline + Interactions**
- Features: Priority, Impact, priority×impact, risk_score (4-5 features)
- Expected F1: ~0.80-0.85 ⬆️ IMPROVEMENT

**Experiment C: Full Recommended Set**
- Features: All 20 recommended features
- Expected F1: ~0.82-0.87 ⬆️ BEST PERFORMANCE

**Experiment D: With Cyclical Encodings**
- Add hour_sin, hour_cos, day_sin, day_cos
- Expected F1: ~0.83-0.88 (slight improvement for temporal patterns)

#### 4. Hyperparameter Tuning Focus
Based on clustering results, prioritize:
- **Class weighting:** `cw=2.0-2.5` (cluster 1 is 98.89% incidents!)
- **SMOTE ratio:** Test 0.5-0.8 (don't oversample too much)
- **Window size:** Larger windows (48-96) may capture better temporal context
- **Hidden size:** 128-256 (moderate complexity sufficient)

---

## 📊 COMPARISON WITH CURRENT SETUP

### Your Current Best Model (30min_win48-SMOTE-Attention):
- **Features:** Priority, Impact (+ rolling means)
- **ROC-AUC:** 0.8955
- **PR-AUC:** 0.9140
- **F1 (optimized):** 0.8100

### Expected Improvements with New Features:
- **Features:** 15-20 engineered features (interactions + temporal)
- **ROC-AUC:** 0.91-0.93 ⬆️ (+1-3% improvement)
- **PR-AUC:** 0.92-0.94 ⬆️ (+1-2% improvement)
- **F1 (optimized):** 0.82-0.85 ⬆️ (+2-5% improvement)

### Why the improvement will be moderate:
- ✅ Priority × Impact already captures most signal
- ✅ Relationships are mostly linear (not highly non-linear)
- ✅ Temporal patterns are moderate (not extreme)
- ⚠️ Main gains will come from **temporal interactions** and **composite features**

---

## ⚠️ IMPORTANT CAVEATS

### 1. Rolling Mean Features
Your current pipeline uses:
- `Priority_rm2h`, `Priority_rm6h`
- `Impact_rm2h`, `Impact_rm6h`

**These are NOT in the recommended list** because:
- They're created AFTER windowing (dataset-specific)
- EDA was done on raw restoration tickets
- Rolling means are **temporal aggregations** (valuable for LSTM!)

**KEEP ROLLING MEANS** in your pipeline! They add temporal context.

### 2. Feature Count vs Performance
- **Don't use all 47 features** - overfitting risk
- **Sweet spot: 15-20 features** (based on MI scores)
- **Always include:** Priority, Impact, priority×impact, risk_score
- **Test with/without:** Temporal interactions, cyclical encodings

### 3. Cyclical Encodings
- MI scores are LOW (0.003-0.006)
- BUT they capture **periodic patterns** that LSTMs love
- **Include them** for hour/day features
- They prevent artificial discontinuities (23:00 → 00:00)

---

## 🎯 NEXT STEPS

### Immediate Actions:
1. ✅ **Integrate top 15 features** into your preprocessing pipeline
2. ✅ **Run baseline comparison** (current features vs new features)
3. ✅ **Test interaction features** (priority×impact, priority×hour)
4. ✅ **Evaluate cyclical encodings** (hour_sin/cos, day_sin/cos)

### Experimental Pipeline:
```python
# Step 1: Load feature-engineered dataset
df_fe = pd.read_csv('data/feature_engineered_dataset.csv')

# Step 2: Select top features
top_features = [
    'Priority', 'Impact',
    'priority_impact_product', 'priority_impact_sum',
    'priority_squared', 'impact_squared',
    'risk_score',
    'priority_hour_interaction', 'impact_hour_interaction',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_business_hours', 'is_weekend',
    'sla_pressure', 'log_duration'
]

# Step 3: Resample + Create windows
# (Same as your current pipeline, but with more features)

# Step 4: Apply SMOTE (Critical! 41% imbalance)

# Step 5: Train LSTM models (all variants)

# Step 6: Compare performance
```

### Evaluation Metrics:
- **Primary:** F1 Score (optimized threshold)
- **Secondary:** PR-AUC (handles imbalance well)
- **Monitoring:** Precision (minimize false alarms)
- **Baseline:** Your current best (F1=0.81, PR-AUC=0.914)

---

## 📈 EXPECTED OUTCOMES

### Conservative Estimate:
- **F1 improvement:** +2-3% (0.81 → 0.83-0.84)
- **Precision improvement:** +1-2% (0.79 → 0.80-0.81)
- **Recall improvement:** +1-2% (0.83 → 0.84-0.85)

### Optimistic Estimate (if temporal interactions strong):
- **F1 improvement:** +4-5% (0.81 → 0.85-0.86)
- **Precision improvement:** +3-4% (0.79 → 0.82-0.83)
- **Recall improvement:** +2-3% (0.83 → 0.85-0.86)

### Unlikely (relationships are linear):
- **F1 improvement:** >5% - Would require discovering new hidden patterns

---

## 🏁 CONCLUSION

### Main Findings:
1. ✅ **Interaction features are KING** - Priority × Impact dominates
2. ✅ **Temporal patterns exist** but are MODERATE (not extreme)
3. ✅ **Relationships are LINEAR/MONOTONIC** (not highly non-linear)
4. ✅ **Clustering perfectly separates incidents** (binary classification is well-defined)
5. ✅ **Night shift & weekends are highest risk** (operational insight)

### Best Features to Add:
1. **priority_impact_product** (MI: 0.677) 🔥
2. **priority_impact_sum** (MI: 0.676) 🔥
3. **risk_score** (MI: 0.675) 🔥
4. **priority_squared**, **impact_squared** (MI: 0.51-0.56)
5. **priority_hour_interaction**, **impact_hour_interaction** (MI: 0.37-0.40)
6. **Cyclical encodings** for hour & day (MI: 0.003-0.006, but important for LSTMs)

### What NOT to Expect:
- ❌ **Massive accuracy jumps** (relationships are already well-captured)
- ❌ **Complex non-linear patterns** (data is mostly linear/monotonic)
- ❌ **Magic features** that solve everything (no silver bullet)

### Realistic Expectations:
- ✅ **Moderate improvements** (2-5% F1 gain)
- ✅ **Better temporal understanding** (hour interactions help)
- ✅ **More robust features** (interactions generalize better)
- ✅ **Interpretable insights** (risk_score, shift patterns)

---

**Report Generated by:** Advanced EDA Analysis Pipeline
**Files Generated:**
- `data/feature_engineered_dataset.csv` (18,969 rows × 36 columns)
- `data/correlation_analysis_results.csv` (27 features analyzed)
- `data/recommended_features.txt` (20 recommended features)
- `EDA_ANALYSIS_REPORT.md` (this report)

**Contact for Questions:** Check notebook/EDA.ipynb for full analysis and visualizations
