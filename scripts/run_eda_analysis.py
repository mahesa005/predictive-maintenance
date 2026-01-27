"""
Run EDA Analysis and Generate Results
This script executes the advanced EDA with non-linear correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print(" " * 30 + "ADVANCED EDA ANALYSIS")
print("="*100)

# Load dataset
print("\n[1/7] Loading dataset...")
df = pd.read_csv('data/raw_dataset.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Resolved Duration (minutes)'] = (
    df['Resolved Duration (minutes)']
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Filter restoration tickets
df_fe = df[df['Type'].str.contains('restoration', case=False, na=False)].copy()
df_fe['incident'] = ((df_fe['Priority'] >= 2) & (df_fe['Impact'] >= 2)).astype(int)

print(f"After filtering restoration: {df_fe.shape}")
print(f"Incident distribution: {df_fe['incident'].value_counts().to_dict()}")

# ========== TEMPORAL FEATURES ==========
print("\n[2/7] Creating temporal features...")
df_fe['hour'] = df_fe['Timestamp'].dt.hour
df_fe['minute'] = df_fe['Timestamp'].dt.minute
df_fe['day_of_week'] = df_fe['Timestamp'].dt.dayofweek
df_fe['day_name'] = df_fe['Timestamp'].dt.day_name()
df_fe['is_weekend'] = (df_fe['day_of_week'] >= 5).astype(int)

def get_shift(hour):
    if hour < 6: return 0
    elif hour < 12: return 1
    elif hour < 18: return 2
    else: return 3

df_fe['shift'] = df_fe['hour'].apply(get_shift)
shift_names = {0: 'Night', 1: 'Morning', 2: 'Afternoon', 3: 'Evening'}
df_fe['shift_name'] = df_fe['shift'].map(shift_names)

df_fe['is_business_hours'] = ((df_fe['hour'] >= 8) & (df_fe['hour'] < 17) & (df_fe['day_of_week'] < 5)).astype(int)
df_fe['is_peak_hours'] = (((df_fe['hour'] >= 9) & (df_fe['hour'] <= 11)) | ((df_fe['hour'] >= 14) & (df_fe['hour'] <= 16))).astype(int)

df_fe['hour_sin'] = np.sin(2 * np.pi * df_fe['hour'] / 24)
df_fe['hour_cos'] = np.cos(2 * np.pi * df_fe['hour'] / 24)
df_fe['day_sin'] = np.sin(2 * np.pi * df_fe['day_of_week'] / 7)
df_fe['day_cos'] = np.cos(2 * np.pi * df_fe['day_of_week'] / 7)

df_fe['month'] = df_fe['Timestamp'].dt.month
df_fe['day_of_month'] = df_fe['Timestamp'].dt.day
df_fe['week_of_year'] = df_fe['Timestamp'].dt.isocalendar().week

print(f"Created {len([c for c in df_fe.columns if c not in df.columns])} new temporal features")

# ========== INTERACTION FEATURES ==========
print("\n[3/7] Creating interaction features...")
df_fe['priority_impact_product'] = df_fe['Priority'] * df_fe['Impact']
df_fe['priority_impact_sum'] = df_fe['Priority'] + df_fe['Impact']
df_fe['priority_impact_diff'] = abs(df_fe['Priority'] - df_fe['Impact'])
df_fe['priority_impact_max'] = df_fe[['Priority', 'Impact']].max(axis=1)
df_fe['priority_impact_min'] = df_fe[['Priority', 'Impact']].min(axis=1)
df_fe['priority_squared'] = df_fe['Priority'] ** 2
df_fe['impact_squared'] = df_fe['Impact'] ** 2

df_fe['has_sla'] = (~df_fe['SLA (minutes)'].isna()).astype(int)
df_fe['sla_duration_ratio'] = df_fe['Resolved Duration (minutes)'] / (df_fe['SLA (minutes)'] + 1e-6)
df_fe['sla_breach'] = (df_fe['Resolved Duration (minutes)'] > df_fe['SLA (minutes)']).astype(int)
df_fe['sla_pressure'] = np.where(df_fe['SLA (minutes)'].notna(), np.clip(df_fe['sla_duration_ratio'], 0, 2), 0)

df_fe['log_duration'] = np.log1p(df_fe['Resolved Duration (minutes)'])
df_fe['sqrt_duration'] = np.sqrt(df_fe['Resolved Duration (minutes)'])

df_fe['priority_hour_interaction'] = df_fe['Priority'] * df_fe['hour']
df_fe['impact_hour_interaction'] = df_fe['Impact'] * df_fe['hour']
df_fe['priority_shift_interaction'] = df_fe['Priority'] * df_fe['shift']

df_fe['risk_score'] = (df_fe['Priority'] * 0.4 + df_fe['Impact'] * 0.4 +
                       df_fe['is_peak_hours'] * 0.1 + (1 - df_fe['is_business_hours']) * 0.1)

print(f"Total features now: {df_fe.shape[1]}")

# ========== TEMPORAL PATTERN ANALYSIS ==========
print("\n[4/7] Analyzing temporal patterns...")
hour_incident_rate = df_fe.groupby('hour')['incident'].agg(['sum', 'count', 'mean'])
day_incident_rate = df_fe.groupby('day_name')['incident'].agg(['sum', 'count', 'mean'])
shift_incident_rate = df_fe.groupby('shift_name')['incident'].agg(['sum', 'count', 'mean'])
bh_comparison = df_fe.groupby('is_business_hours')['incident'].agg(['sum', 'count', 'mean'])

print("\nTEMPORAL INSIGHTS:")
print(f"  Peak hour: {hour_incident_rate['mean'].idxmax()}:00 ({hour_incident_rate['mean'].max():.2%} incident rate)")
print(f"  Lowest hour: {hour_incident_rate['mean'].idxmin()}:00 ({hour_incident_rate['mean'].min():.2%} incident rate)")
print(f"  Peak shift: {shift_incident_rate['mean'].idxmax()} ({shift_incident_rate['mean'].max():.2%})")
print(f"  Business hours incident rate: {bh_comparison.loc[1, 'mean']:.2%}")
print(f"  Non-business hours incident rate: {bh_comparison.loc[0, 'mean']:.2%}")
print(f"  Weekend incident rate: {df_fe[df_fe['is_weekend']==1]['incident'].mean():.2%}")
print(f"  Weekday incident rate: {df_fe[df_fe['is_weekend']==0]['incident'].mean():.2%}")

# ========== NON-LINEAR CORRELATION ANALYSIS ==========
print("\n[5/7] Computing non-linear correlations...")

numerical_features = [
    'Priority', 'Impact', 'hour', 'day_of_week', 'shift',
    'is_weekend', 'is_business_hours', 'is_peak_hours',
    'priority_impact_product', 'priority_impact_sum', 'priority_impact_diff',
    'priority_squared', 'impact_squared',
    'sla_pressure', 'sla_breach', 'has_sla',
    'log_duration', 'sqrt_duration',
    'priority_hour_interaction', 'impact_hour_interaction',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'risk_score', 'month', 'week_of_year'
]

available_features = [f for f in numerical_features if f in df_fe.columns]
X_corr = df_fe[available_features].fillna(0)
y_corr = df_fe['incident'].values

print(f"Analyzing {len(available_features)} features...")

# Mutual Information
mi_scores = mutual_info_classif(X_corr, y_corr, discrete_features='auto', random_state=42)
mi_df = pd.DataFrame({'Feature': available_features, 'Mutual_Information': mi_scores}).sort_values('Mutual_Information', ascending=False)

# Spearman Correlation
spearman_scores = []
for feature in available_features:
    corr, _ = spearmanr(X_corr[feature], y_corr)
    spearman_scores.append(abs(corr))
spearman_df = pd.DataFrame({'Feature': available_features, 'Spearman_Abs': spearman_scores}).sort_values('Spearman_Abs', ascending=False)

# Pearson Correlation
pearson_scores = []
for feature in available_features:
    corr, _ = pearsonr(X_corr[feature], y_corr)
    pearson_scores.append(abs(corr))
pearson_df = pd.DataFrame({'Feature': available_features, 'Pearson_Abs': pearson_scores}).sort_values('Pearson_Abs', ascending=False)

# Combine
correlation_summary = mi_df.merge(spearman_df, on='Feature').merge(pearson_df, on='Feature')
correlation_summary['Composite_Score'] = (correlation_summary['Mutual_Information'] * 0.5 +
                                          correlation_summary['Spearman_Abs'] * 0.3 +
                                          correlation_summary['Pearson_Abs'] * 0.2)
correlation_summary = correlation_summary.sort_values('Composite_Score', ascending=False)
correlation_summary['NonLinear_Ratio'] = correlation_summary['Mutual_Information'] / (correlation_summary['Pearson_Abs'] + 0.01)
correlation_summary['Is_NonLinear'] = ((correlation_summary['NonLinear_Ratio'] > 1.5) |
                                       ((correlation_summary['Mutual_Information'] > 0.05) & (correlation_summary['Pearson_Abs'] < 0.1)))

print("\nTOP 15 FEATURES BY COMPOSITE SCORE:")
print("="*100)
print(correlation_summary.head(15)[['Feature', 'Mutual_Information', 'Spearman_Abs', 'Pearson_Abs', 'Composite_Score']].to_string(index=False))

print("\n\nHIGHLY NON-LINEAR FEATURES:")
print("="*100)
nonlinear_features = correlation_summary[correlation_summary['Is_NonLinear'] == True].head(10)
if len(nonlinear_features) > 0:
    print(nonlinear_features[['Feature', 'Mutual_Information', 'Pearson_Abs', 'NonLinear_Ratio']].to_string(index=False))
else:
    print("No highly non-linear features detected")

# ========== CLUSTERING ANALYSIS ==========
print("\n[6/7] Performing clustering analysis...")
top_features_composite = correlation_summary.head(15)['Feature'].tolist()
clustering_features = top_features_composite[:10]
X_cluster = df_fe[clustering_features].fillna(0)
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Find optimal k
best_k = 4
best_silhouette = -1
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    score = silhouette_score(X_cluster_scaled, labels)
    if score > best_silhouette:
        best_silhouette = score
        best_k = k

print(f"Optimal k: {best_k} (Silhouette: {best_silhouette:.4f})")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df_fe['cluster'] = kmeans_final.fit_predict(X_cluster_scaled)

print("\nCLUSTER ANALYSIS:")
print("="*100)
for cluster_id in range(best_k):
    cluster_data = df_fe[df_fe['cluster'] == cluster_id]
    incident_rate = cluster_data['incident'].mean()
    size = len(cluster_data)

    print(f"\nCluster {cluster_id}: {size} tickets ({size/len(df_fe)*100:.1f}%)")
    print(f"  Incident Rate: {incident_rate:.2%}")
    print(f"  Avg Priority: {cluster_data['Priority'].mean():.2f}")
    print(f"  Avg Impact: {cluster_data['Impact'].mean():.2f}")
    print(f"  Avg Hour: {cluster_data['hour'].mean():.1f}")
    print(f"  Weekend %: {cluster_data['is_weekend'].mean():.1%}")
    print(f"  Business Hours %: {cluster_data['is_business_hours'].mean():.1%}")

    if incident_rate > 0.7:
        profile = "[CRITICAL]"
    elif incident_rate > 0.4:
        profile = "[HIGH RISK]"
    elif incident_rate > 0.2:
        profile = "[MODERATE]"
    else:
        profile = "[LOW RISK]"
    print(f"  Profile: {profile}")

# ========== EXPORT RESULTS ==========
print("\n[7/7] Exporting results...")

# Export feature-engineered dataset
export_columns = ['Timestamp', 'Priority', 'Impact', 'incident'] + [
    col for col in df_fe.columns
    if col not in ['Timestamp', 'Priority', 'Impact', 'incident', 'NO', 'Service Type',
                   'Service Name', 'Type', 'Status', 'SLA (minutes)', 'Resolved Time',
                   'Resolved Duration (minutes)', 'Month', 'day_name', 'shift_name']
]

df_export = df_fe[export_columns].copy()
df_export.to_csv('data/feature_engineered_dataset.csv', index=False)
print(f"[OK] Saved: data/feature_engineered_dataset.csv ({df_export.shape})")

# Export correlation results
correlation_summary.to_csv('data/correlation_analysis_results.csv', index=False)
print(f"[OK] Saved: data/correlation_analysis_results.csv")

# Export recommended features
recommended_features_final = list(set(
    top_features_composite[:15] +
    ['Priority', 'Impact', 'priority_impact_product', 'priority_impact_sum'] +
    [f for f in available_features if 'sin' in f or 'cos' in f]
))
recommended_sorted = correlation_summary[correlation_summary['Feature'].isin(recommended_features_final)].sort_values('Mutual_Information', ascending=False)

with open('data/recommended_features.txt', 'w', encoding='utf-8') as f:
    f.write(f"# Recommended Features for LSTM Training\n")
    f.write(f"# Total Features: {len(recommended_sorted)}\n\n")
    for i, (idx, row) in enumerate(recommended_sorted.iterrows(), 1):
        f.write(f"{i}. {row['Feature']} (MI: {row['Mutual_Information']:.4f})\n")

print(f"[OK] Saved: data/recommended_features.txt ({len(recommended_sorted)} features)")

print("\n" + "="*100)
print(" " * 35 + "ANALYSIS COMPLETE!")
print("="*100)
print(f"\nKey Statistics:")
print(f"  Total features created: {df_fe.shape[1]}")
print(f"  Recommended features: {len(recommended_sorted)}")
print(f"  Optimal clusters: {best_k}")
print(f"  Best Silhouette Score: {best_silhouette:.4f}")
print(f"  Dataset size: {len(df_fe)} restoration tickets")
print(f"  Incident rate: {df_fe['incident'].mean():.2%}")
print("\nTop 5 Features by Mutual Information:")
for i, (idx, row) in enumerate(mi_df.head(5).iterrows(), 1):
    print(f"  {i}. {row['Feature']:30s} - MI: {row['Mutual_Information']:.4f}")
print("\n" + "="*100)
print("FILES EXPORTED:")
print("  - data/feature_engineered_dataset.csv")
print("  - data/correlation_analysis_results.csv")
print("  - data/recommended_features.txt")
print("="*100)
