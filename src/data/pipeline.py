"""
Data Pipeline for Predictive Maintenance LSTM Model

This module provides functions for:
- Loading raw incident data
- Preprocessing and feature engineering
- Creating windowed datasets for LSTM training
- Feature strategy configurations

Usage:
    from src.data.pipeline import load_and_preprocess, create_windowed_dataset, FEATURE_STRATEGIES

    # Load and preprocess data
    df = load_and_preprocess('data/raw_dataset.csv')

    # Select feature strategy
    feature_cols = FEATURE_STRATEGIES['service_moderate']

    # Create windowed dataset
    X, y, timestamps = create_windowed_dataset(df, feature_cols, 'incident', window_size=48)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any


# ============================================================
# FEATURE STRATEGY DEFINITIONS
# ============================================================

FEATURE_STRATEGIES: Dict[str, List[str]] = {
    # ========== BASELINE ==========
    'baseline': ['Priority', 'Impact'],

    # ========== INTERACTION-ONLY ==========
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

    # ========== SERVICE-ENHANCED ==========
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

    # ========== SEQUENCE-ENHANCED ==========
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

    # ========== ULTIMATE: SERVICE + SEQUENCE ==========
    'ultimate': [
        'Priority', 'Impact',
        'priority_impact_product', 'priority_impact_sum', 'risk_score',
        'priority_squared', 'impact_squared',
        'priority_hour_interaction', 'impact_hour_interaction',
        'hour_sin', 'hour_cos',
        'service_risk_score', 'service_frequency',
        'priority_volatility', 'impact_volatility',
        'incidents_last_2h', 'consecutive_high_priority',
        'priority_change', 'priority_acceleration'
    ]
}


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw incident data from CSV file.

    Args:
        filepath: Path to the raw CSV file

    Returns:
        DataFrame with raw incident data
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw data: filter restoration types and create incident label.

    Args:
        df: Raw DataFrame with incident data

    Returns:
        Preprocessed DataFrame with incident label
    """
    df = df.copy()

    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Filter for restoration type only
    df = df[df['Type'].str.contains('restoration', case=False, na=False)]

    # Convert Priority and Impact to numeric
    df['Priority'] = pd.to_numeric(df['Priority'], errors='coerce')
    df['Impact'] = pd.to_numeric(df['Impact'], errors='coerce')

    # Create incident label: Priority >= 2 AND Impact >= 2
    df['incident'] = ((df['Priority'] >= 2) & (df['Impact'] >= 2)).astype(int)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features for the predictive model.

    Feature groups:
    - Temporal: hour, day_of_week, cyclical encodings
    - Interaction: priority-impact products, sums, risk scores
    - Service: risk scores based on service type/name
    - Sequence: volatility, change rates, historical context

    Args:
        df: Preprocessed DataFrame with incident label

    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()

    # === TEMPORAL FEATURES ===
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek

    # === INTERACTION FEATURES ===
    # Tier 1: Basic interactions
    df['priority_impact_product'] = df['Priority'] * df['Impact']
    df['priority_impact_sum'] = df['Priority'] + df['Impact']
    df['risk_score'] = (0.6 * df['Priority']) + (0.4 * df['Impact'])

    # Tier 2: Polynomial
    df['priority_squared'] = df['Priority'] ** 2
    df['impact_squared'] = df['Impact'] ** 2

    # Tier 3: Temporal interactions
    df['priority_hour_interaction'] = df['Priority'] * df['hour']
    df['impact_hour_interaction'] = df['Impact'] * df['hour']

    # Cyclical encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # === SERVICE TYPE FEATURES ===
    # Target encoding: Encode by incident rate
    service_incident_rates = df.groupby('Service Type')['incident'].mean()
    df['service_risk_score'] = df['Service Type'].map(service_incident_rates)

    # Frequency encoding: How common is this service?
    service_freq = df['Service Type'].value_counts(normalize=True)
    df['service_frequency'] = df['Service Type'].map(service_freq)

    # Service Name encoding (higher granularity)
    name_incident_rates = df.groupby('Service Name')['incident'].mean()
    df['service_name_risk'] = df['Service Name'].map(name_incident_rates)

    # Handle NaN values for service features
    df['service_risk_score'] = df['service_risk_score'].fillna(df['service_risk_score'].mean())
    df['service_frequency'] = df['service_frequency'].fillna(0)
    df['service_name_risk'] = df['service_name_risk'].fillna(df['service_name_risk'].mean())

    # === SEQUENCE-AWARE FEATURES ===
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

    return df


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load raw data and apply full preprocessing pipeline.

    Combines: load_data -> preprocess_data -> create_features

    Args:
        filepath: Path to the raw CSV file

    Returns:
        Fully preprocessed DataFrame with all features
    """
    df = load_data(filepath)
    df = preprocess_data(df)
    df = create_features(df)
    return df


def resample_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    sampling_period: str = '30min'
) -> pd.DataFrame:
    """
    Resample time-series data to specified period.

    Args:
        df: DataFrame with Timestamp index or column
        feature_cols: List of feature columns to include
        sampling_period: Resampling period (e.g., '30min', '1h')

    Returns:
        Resampled DataFrame
    """
    df = df.copy()

    # Build aggregation dictionary
    agg_dict = {'incident': 'sum'}
    for feature in feature_cols:
        if feature in ['hour', 'day_of_week']:
            agg_dict[feature] = 'first'
        else:
            agg_dict[feature] = 'mean'

    # Resample
    df_resampled = (df.set_index('Timestamp')
                     .resample(sampling_period)
                     .agg(agg_dict)
                     .fillna(0)
                     .reset_index())

    # Convert incident back to binary
    df_resampled['incident'] = (df_resampled['incident'] > 0).astype(int)

    return df_resampled


def add_rolling_means(
    df: pd.DataFrame,
    rolling_hours: List[int],
    sampling_period: str = '30min'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add rolling mean features for Priority and Impact.

    Args:
        df: DataFrame with Priority and Impact columns
        rolling_hours: List of hour windows (e.g., [2, 6] for 2h and 6h rolling means)
        sampling_period: Sampling period to calculate correct window size

    Returns:
        Tuple of (DataFrame with rolling features, list of new feature column names)
    """
    df = df.copy()
    new_cols = []

    # Calculate periods per hour based on sampling
    if sampling_period == '30min':
        periods_per_hour = 2
    elif sampling_period == '1h':
        periods_per_hour = 1
    else:
        # Extract minutes from period string
        minutes = int(sampling_period.replace('min', ''))
        periods_per_hour = 60 // minutes

    for rm_hours in rolling_hours:
        rm_periods = rm_hours * periods_per_hour

        priority_col = f'Priority_rm{rm_hours}h'
        impact_col = f'Impact_rm{rm_hours}h'

        df[priority_col] = df['Priority'].rolling(window=rm_periods, min_periods=1).mean()
        df[impact_col] = df['Impact'].rolling(window=rm_periods, min_periods=1).mean()

        new_cols.extend([priority_col, impact_col])

    return df, new_cols


def create_windowed_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding window dataset for LSTM training.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names to use
        target_col: Name of target column
        window_size: Number of timesteps in each window

    Returns:
        Tuple of (X, y, timestamps):
        - X: shape (n_samples, window_size, n_features)
        - y: shape (n_samples,)
        - timestamps: shape (n_samples,) - timestamp of prediction point
    """
    X, y, timestamps = [], [], []

    for i in range(len(df) - window_size):
        window = df[feature_cols].iloc[i:i+window_size].astype(float).values
        label = df[target_col].iloc[i+window_size]
        timestamp = df['Timestamp'].iloc[i+window_size]

        X.append(window)
        y.append(label)
        timestamps.append(timestamp)

    return np.array(X), np.array(y), np.array(timestamps)


def prepare_training_data(
    filepath: str,
    feature_strategy: str = 'service_moderate',
    sampling_period: str = '30min',
    window_size: int = 48,
    rolling_hours: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Complete pipeline to prepare training data from raw CSV.

    Args:
        filepath: Path to raw CSV file
        feature_strategy: Name of feature strategy (see FEATURE_STRATEGIES)
        sampling_period: Resampling period (e.g., '30min', '1h')
        window_size: Number of timesteps in each window
        rolling_hours: Optional list of rolling mean hour windows

    Returns:
        Tuple of (X, y, timestamps, feature_cols):
        - X: shape (n_samples, window_size, n_features)
        - y: shape (n_samples,)
        - timestamps: shape (n_samples,)
        - feature_cols: list of feature column names used
    """
    if feature_strategy not in FEATURE_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {feature_strategy}. "
            f"Available: {list(FEATURE_STRATEGIES.keys())}"
        )

    # Load and preprocess
    df = load_and_preprocess(filepath)

    # Get feature columns for strategy
    feature_cols = FEATURE_STRATEGIES[feature_strategy].copy()

    # Resample
    df_resampled = resample_data(df, feature_cols, sampling_period)

    # Add rolling means if specified
    if rolling_hours:
        df_resampled, new_cols = add_rolling_means(
            df_resampled, rolling_hours, sampling_period
        )
        feature_cols.extend(new_cols)

    # Create windowed dataset
    X, y, timestamps = create_windowed_dataset(
        df_resampled, feature_cols, 'incident', window_size
    )

    return X, y, timestamps, feature_cols


def get_class_distribution(y: np.ndarray) -> Dict[str, Any]:
    """
    Get class distribution statistics.

    Args:
        y: Target array

    Returns:
        Dictionary with class counts and ratios
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    return {
        'total_samples': total,
        'class_counts': dict(zip(unique.astype(int), counts.astype(int))),
        'class_ratios': {int(k): v/total for k, v in zip(unique, counts)},
        'imbalance_ratio': counts.max() / counts.min() if counts.min() > 0 else float('inf')
    }


# ============================================================
# CLI USAGE
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data Pipeline for Predictive Maintenance')
    parser.add_argument('--input', '-i', required=True, help='Path to raw CSV file')
    parser.add_argument('--strategy', '-s', default='service_moderate',
                        choices=list(FEATURE_STRATEGIES.keys()),
                        help='Feature strategy to use')
    parser.add_argument('--sampling', default='30min', help='Sampling period')
    parser.add_argument('--window', '-w', type=int, default=48, help='Window size')
    parser.add_argument('--output', '-o', help='Output directory for processed data')

    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    print(f"Feature strategy: {args.strategy}")
    print(f"Sampling period: {args.sampling}")
    print(f"Window size: {args.window}")

    X, y, timestamps, feature_cols = prepare_training_data(
        args.input,
        feature_strategy=args.strategy,
        sampling_period=args.sampling,
        window_size=args.window
    )

    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    dist = get_class_distribution(y)
    print(f"\nClass distribution:")
    print(f"  Normal (0): {dist['class_counts'].get(0, 0)} ({dist['class_ratios'].get(0, 0)*100:.1f}%)")
    print(f"  Incident (1): {dist['class_counts'].get(1, 0)} ({dist['class_ratios'].get(1, 0)*100:.1f}%)")
    print(f"  Imbalance ratio: 1:{dist['imbalance_ratio']:.1f}")

    if args.output:
        import os
        os.makedirs(args.output, exist_ok=True)

        np.save(os.path.join(args.output, 'X.npy'), X)
        np.save(os.path.join(args.output, 'y.npy'), y)
        np.save(os.path.join(args.output, 'timestamps.npy'), timestamps)

        print(f"\nData saved to: {args.output}")
