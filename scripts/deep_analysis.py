"""
Deep Analysis Script for Predictive Maintenance Model Weaknesses

Implements 5 analysis strategies:
1. Failure Type Analysis (FP vs FN breakdown)
2. Confidence Gap Analysis (high confidence mistakes)
3. Temporal Lead-Time Analysis
4. Attention Heatmap Analysis
5. Feature Set Comparison Analysis

Usage:
    python scripts/deep_analysis.py --experiment f13-service_moderate-Attention
    python scripts/deep_analysis.py --compare f11,f13,f15
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import json
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
CHECKPOINTS_DIR = PROJECT_ROOT / "src" / "model" / "checkpoints"
ANALYSIS_DIR.mkdir(exist_ok=True)


def load_experiment_data(experiment_name: str, window: int = 48, interval: str = "30min") -> Dict:
    """Load y_test and y_prob from an experiment."""
    exp_dir = OUTPUT_DIR / f"{interval}_win{window}_{experiment_name}"

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")

    y_test_file = list(exp_dir.glob("y_test_*.npy"))[0]
    y_prob_file = list(exp_dir.glob("y_prob_*.npy"))[0]

    return {
        'y_test': np.load(y_test_file),
        'y_prob': np.load(y_prob_file),
        'name': experiment_name,
        'dir': exp_dir
    }


def analyze_failure_types(y_test: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Strategy 1: Bedah Tipe Kegagalan (FP vs FN)

    Analyzes False Positives (False Alarms) and False Negatives (Missed Failures)
    to understand the types of errors the model makes.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Get indices for each error type
    tn_mask = (y_test == 0) & (y_pred == 0)
    fp_mask = (y_test == 0) & (y_pred == 1)  # False Alarm
    fn_mask = (y_test == 1) & (y_pred == 0)  # Missed Failure
    tp_mask = (y_test == 1) & (y_pred == 1)

    # Get counts
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Get probability distributions for each error type
    fp_probs = y_prob[fp_mask]
    fn_probs = y_prob[fn_mask]

    # Analyze error severity
    results = {
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        'fp_analysis': {
            'count': int(fp),
            'percentage': fp / (tn + fp) * 100 if (tn + fp) > 0 else 0,
            'avg_confidence': float(np.mean(fp_probs)) if len(fp_probs) > 0 else 0,
            'high_confidence_fp': int(np.sum(fp_probs >= 0.8)),  # Very confident false alarms
            'interpretation': 'Model over-predicts: sees noise as potential failure patterns'
        },
        'fn_analysis': {
            'count': int(fn),
            'percentage': fn / (fn + tp) * 100 if (fn + tp) > 0 else 0,
            'avg_confidence': float(1 - np.mean(fn_probs)) if len(fn_probs) > 0 else 0,
            'low_confidence_fn': int(np.sum(fn_probs < 0.2)),  # Very confident misses
            'interpretation': 'Model misses failures: likely sudden/silent failures without degradation pattern'
        },
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'metrics': {
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
        }
    }

    return results


def analyze_confidence_gap(y_test: np.ndarray, y_prob: np.ndarray,
                           high_conf_threshold: float = 0.8) -> Dict:
    """
    Strategy 2: Analisis Confidence Gap

    Finds predictions that are wrong but highly confident - these reveal
    ambiguous patterns where the model is confidently wrong.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    # High confidence wrong predictions
    wrong_mask = y_pred != y_test

    # Wrong and highly confident for class 1
    high_conf_wrong_pos = wrong_mask & (y_prob >= high_conf_threshold) & (y_test == 0)
    # Wrong and highly confident for class 0
    high_conf_wrong_neg = wrong_mask & (y_prob < (1 - high_conf_threshold)) & (y_test == 1)

    results = {
        'total_wrong': int(np.sum(wrong_mask)),
        'high_confidence_false_positives': {
            'count': int(np.sum(high_conf_wrong_pos)),
            'indices': np.where(high_conf_wrong_pos)[0].tolist()[:20],  # First 20
            'probs': y_prob[high_conf_wrong_pos].tolist()[:20],
            'interpretation': 'Patterns identical to failures but system recovered - model cannot distinguish "healing" vs "fatal" symptoms'
        },
        'high_confidence_false_negatives': {
            'count': int(np.sum(high_conf_wrong_neg)),
            'indices': np.where(high_conf_wrong_neg)[0].tolist()[:20],
            'probs': y_prob[high_conf_wrong_neg].tolist()[:20],
            'interpretation': 'Silent/sudden failures with no degradation pattern - data appears normal before failure'
        },
        'confidence_distribution': {
            'wrong_preds_mean_conf': float(np.mean(np.abs(y_prob[wrong_mask] - 0.5) * 2)) if np.sum(wrong_mask) > 0 else 0,
            'correct_preds_mean_conf': float(np.mean(np.abs(y_prob[~wrong_mask] - 0.5) * 2)) if np.sum(~wrong_mask) > 0 else 0,
        }
    }

    # Add ambiguity analysis (predictions near 0.5)
    ambiguous_mask = (y_prob >= 0.4) & (y_prob <= 0.6)
    results['ambiguous_predictions'] = {
        'count': int(np.sum(ambiguous_mask)),
        'percentage': float(np.sum(ambiguous_mask) / len(y_prob) * 100),
        'correct_rate': float(np.mean((y_pred == y_test)[ambiguous_mask])) if np.sum(ambiguous_mask) > 0 else 0
    }

    return results


def analyze_temporal_leadtime(y_test: np.ndarray, y_prob: np.ndarray,
                               window_minutes: int = 30) -> Dict:
    """
    Strategy 3: Analisis Temporal (Lead-Time)

    Analyzes how early the model starts giving high probability predictions
    before actual incidents. Longer lead time = better early detection.
    """
    threshold = 0.5
    high_alert_threshold = 0.7

    # Find incident windows (consecutive 1s in y_test)
    incident_starts = []
    incident_ends = []

    in_incident = False
    for i, label in enumerate(y_test):
        if label == 1 and not in_incident:
            incident_starts.append(i)
            in_incident = True
        elif label == 0 and in_incident:
            incident_ends.append(i - 1)
            in_incident = False

    if in_incident:
        incident_ends.append(len(y_test) - 1)

    # Analyze lead time for each incident
    lead_times = []
    early_detections = 0
    late_detections = 0
    missed_detections = 0

    for start, end in zip(incident_starts, incident_ends):
        # Look back from incident start to find when model first alerted
        lookback_start = max(0, start - 24)  # Look back up to 12 hours (24 * 30min)

        first_alert = None
        for i in range(lookback_start, start):
            if y_prob[i] >= threshold:
                first_alert = i
                break

        if first_alert is not None:
            lead_time_periods = start - first_alert
            lead_time_hours = lead_time_periods * window_minutes / 60
            lead_times.append(lead_time_hours)

            if lead_time_hours >= 2:  # At least 2 hours warning
                early_detections += 1
            else:
                late_detections += 1
        else:
            # Check if model detected during incident
            if np.any(y_prob[start:end+1] >= threshold):
                late_detections += 1
                lead_times.append(0)
            else:
                missed_detections += 1

    total_incidents = len(incident_starts)

    results = {
        'total_incidents': total_incidents,
        'incident_windows': list(zip(incident_starts, incident_ends))[:10],  # First 10
        'lead_time_stats': {
            'avg_lead_time_hours': float(np.mean(lead_times)) if lead_times else 0,
            'max_lead_time_hours': float(np.max(lead_times)) if lead_times else 0,
            'min_lead_time_hours': float(np.min(lead_times)) if lead_times else 0,
            'std_lead_time_hours': float(np.std(lead_times)) if lead_times else 0,
        },
        'detection_quality': {
            'early_detections_2h_plus': early_detections,
            'late_detections': late_detections,
            'missed_detections': missed_detections,
            'early_detection_rate': early_detections / total_incidents * 100 if total_incidents > 0 else 0
        },
        'interpretation': 'Higher early detection rate = model sees degradation patterns earlier',
        'weakness': 'Low early detection suggests model only reacts to immediate pre-failure patterns, not gradual degradation'
    }

    return results


def find_model_checkpoint(experiment_name: str, window: int = 48, interval: str = "30min") -> Optional[Path]:
    """Find model checkpoint file for an experiment."""
    # Try different naming patterns
    patterns = [
        f"lstm_attention_{interval}_win{window}_{experiment_name}.pkl",
        f"lstm_nested_{interval}_win{window}_{experiment_name}.pkl",
        f"lstm_optimized_{interval}_win{window}_{experiment_name}.pkl",
        f"lstm_bidirectional_{interval}_win{window}_{experiment_name}.pkl",
        f"lstm_peephole_{interval}_win{window}_{experiment_name}.pkl",
        f"lstm_cifg_{interval}_win{window}_{experiment_name}.pkl",
        f"lstm_*_{interval}_win{window}_{experiment_name}.pkl",
    ]

    for pattern in patterns:
        matches = list(CHECKPOINTS_DIR.glob(pattern))
        if matches:
            return matches[0]

    # Try partial match
    for f in CHECKPOINTS_DIR.glob("*.pkl"):
        if experiment_name in f.name and f"{interval}_win{window}" in f.name:
            return f

    return None


def analyze_attention_patterns(experiment_name: str, window: int = 48,
                               interval: str = "30min", sample_size: int = 100,
                               X_test: Optional[np.ndarray] = None,
                               y_test: Optional[np.ndarray] = None) -> Dict:
    """
    Strategy 4: Attention Heatmap Analysis

    Extracts and analyzes attention weights to understand which timesteps
    the model focuses on. If model only attends to recent steps, it ignores
    long-term trends.
    """
    import pickle

    # Find model checkpoint
    model_file = find_model_checkpoint(experiment_name, window, interval)

    results = {
        'model_found': model_file is not None,
        'attention_analysis': None,
        'model_path': str(model_file) if model_file else None
    }

    if model_file is None:
        results['note'] = f'No model checkpoint found for {experiment_name}'
        results['searched_in'] = str(CHECKPOINTS_DIR)
        return results

    # Check if model has attention mechanism
    if 'attention' not in str(model_file).lower() and 'Attention' not in experiment_name:
        results['note'] = 'Model does not appear to have attention mechanism'
        results['model_type'] = model_file.stem.split('_')[1] if '_' in model_file.stem else 'unknown'
        return results

    # Load model
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        results['error'] = f'Failed to load model: {str(e)}'
        return results

    # Check if model has get_attention_weights method
    if not hasattr(model, 'get_attention_weights'):
        results['note'] = 'Model does not have get_attention_weights method'
        results['available_methods'] = [m for m in dir(model) if not m.startswith('_')]
        return results

    # Need test data to analyze attention
    if X_test is None:
        results['note'] = 'X_test data required for attention analysis. Pass X_test parameter.'
        results['model_loaded'] = True
        return results

    # Get attention weights
    try:
        # Sample subset for analysis
        n_samples = min(sample_size, len(X_test))
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[indices]
        y_sample = y_test[indices] if y_test is not None else None

        attention_weights = model.get_attention_weights(X_sample)
        # attention_weights shape: (batch_size, seq_len)

        mean_attention = attention_weights.mean(axis=0)

        # Analyze temporal focus patterns
        seq_len = len(mean_attention)
        quarter = seq_len // 4

        # Split into quarters
        q1_focus = mean_attention[:quarter].sum()  # Oldest history
        q2_focus = mean_attention[quarter:2*quarter].sum()
        q3_focus = mean_attention[2*quarter:3*quarter].sum()
        q4_focus = mean_attention[3*quarter:].sum()  # Most recent

        # Recent vs historical
        recent_focus = mean_attention[-12:].sum()  # Last 6 hours (12 * 30min)
        historical_focus = mean_attention[:-12].sum()

        # Analyze by class if y_test provided
        class_analysis = None
        if y_sample is not None:
            normal_indices = np.where(y_sample == 0)[0]
            incident_indices = np.where(y_sample == 1)[0]

            if len(normal_indices) > 0 and len(incident_indices) > 0:
                normal_attention = attention_weights[normal_indices].mean(axis=0)
                incident_attention = attention_weights[incident_indices].mean(axis=0)

                class_analysis = {
                    'normal_recent_focus': float(normal_attention[-12:].sum()),
                    'incident_recent_focus': float(incident_attention[-12:].sum()),
                    'normal_historical_focus': float(normal_attention[:-12].sum()),
                    'incident_historical_focus': float(incident_attention[:-12].sum()),
                    'incident_focuses_more_recent': bool(incident_attention[-12:].sum() > normal_attention[-12:].sum())
                }

        # Peak detection - find timesteps with highest attention
        peak_threshold = mean_attention.mean() + mean_attention.std()
        peak_indices = np.where(mean_attention > peak_threshold)[0]
        peak_hours_before = [(seq_len - idx) * 0.5 for idx in peak_indices]  # Convert to hours

        results['attention_analysis'] = {
            'temporal_distribution': {
                'q1_oldest_25pct': float(q1_focus),
                'q2_25_50pct': float(q2_focus),
                'q3_50_75pct': float(q3_focus),
                'q4_recent_25pct': float(q4_focus),
            },
            'focus_ratio': {
                'recent_6h': float(recent_focus),
                'historical': float(historical_focus),
                'recent_to_historical_ratio': float(recent_focus / historical_focus) if historical_focus > 0 else float('inf')
            },
            'peak_attention': {
                'peak_timesteps': peak_indices.tolist(),
                'peak_hours_before_prediction': peak_hours_before,
                'num_peaks': len(peak_indices)
            },
            'statistics': {
                'mean': float(mean_attention.mean()),
                'std': float(mean_attention.std()),
                'max': float(mean_attention.max()),
                'max_timestep': int(np.argmax(mean_attention)),
                'entropy': float(-np.sum(mean_attention * np.log(mean_attention + 1e-9)))
            },
            'class_analysis': class_analysis,
            'raw_mean_attention': mean_attention.tolist()
        }

        # Interpretation
        if recent_focus / (historical_focus + 1e-9) > 2:
            results['interpretation'] = 'Model heavily focuses on recent history (last 6h). May miss long-term degradation patterns.'
            results['weakness'] = 'Short-term bias - model acts like instant classifier, not sequence processor'
        elif q4_focus > q1_focus + q2_focus:
            results['interpretation'] = 'Model biased toward recent timesteps. Some historical context used but limited.'
            results['weakness'] = 'Moderate recency bias'
        else:
            results['interpretation'] = 'Model uses balanced attention across time window. Good sequence processing.'
            results['weakness'] = None

        # Save attention heatmap
        save_attention_heatmap(attention_weights, y_sample, experiment_name, window, interval)
        results['heatmap_saved'] = str(ANALYSIS_DIR / f"attention_heatmap_{experiment_name}.png")

    except Exception as e:
        results['error'] = f'Failed to compute attention: {str(e)}'
        import traceback
        results['traceback'] = traceback.format_exc()

    return results


def save_attention_heatmap(attention_weights: np.ndarray, y_test: Optional[np.ndarray],
                           experiment_name: str, window: int, interval: str):
    """Save attention heatmap visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    seq_len = attention_weights.shape[1]
    time_labels = [f'-{(seq_len - i) * 0.5:.1f}h' for i in range(0, seq_len, max(1, seq_len // 8))]

    # 1. Mean attention over time
    mean_attention = attention_weights.mean(axis=0)
    axes[0, 0].bar(range(seq_len), mean_attention, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Timestep (30min intervals)')
    axes[0, 0].set_ylabel('Mean Attention Weight')
    axes[0, 0].set_title('Average Attention Distribution Over Time')
    axes[0, 0].axvline(x=seq_len - 12, color='red', linestyle='--', alpha=0.5, label='Last 6h')
    axes[0, 0].legend()

    # 2. Attention heatmap (subset of samples)
    n_show = min(50, len(attention_weights))
    sns.heatmap(attention_weights[:n_show], ax=axes[0, 1], cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'})
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Sample')
    axes[0, 1].set_title(f'Attention Heatmap (first {n_show} samples)')

    # 3. Attention by class (if y_test provided)
    if y_test is not None:
        normal_mask = y_test == 0
        incident_mask = y_test == 1

        if np.sum(normal_mask) > 0 and np.sum(incident_mask) > 0:
            normal_mean = attention_weights[normal_mask].mean(axis=0)
            incident_mean = attention_weights[incident_mask].mean(axis=0)

            x = np.arange(seq_len)
            width = 0.4
            axes[1, 0].bar(x - width/2, normal_mean, width, label='Normal', color='green', alpha=0.7)
            axes[1, 0].bar(x + width/2, incident_mean, width, label='Incident', color='red', alpha=0.7)
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Mean Attention Weight')
            axes[1, 0].set_title('Attention by Class')
            axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No class labels provided', ha='center', va='center')
        axes[1, 0].set_title('Attention by Class (unavailable)')

    # 4. Cumulative attention
    cumsum = np.cumsum(mean_attention)
    axes[1, 1].plot(range(seq_len), cumsum, 'b-', linewidth=2)
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% attention')
    axes[1, 1].axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% attention')
    # Find where 50% and 90% attention is reached
    idx_50 = np.searchsorted(cumsum, 0.5)
    idx_90 = np.searchsorted(cumsum, 0.9)
    axes[1, 1].axvline(x=idx_50, color='red', linestyle=':', alpha=0.5)
    axes[1, 1].axvline(x=idx_90, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Cumulative Attention')
    axes[1, 1].set_title(f'Cumulative Attention (50% at t={idx_50}, 90% at t={idx_90})')
    axes[1, 1].legend()

    plt.suptitle(f'Attention Analysis: {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = ANALYSIS_DIR / f"attention_heatmap_{experiment_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention heatmap to: {save_path}")


def compare_feature_sets(experiments: List[str], window: int = 48,
                         interval: str = "30min") -> Dict:
    """
    Strategy 5: Feature Set Comparison

    Compares performance characteristics across different feature strategies
    to understand trade-offs (precision vs recall, noise sensitivity, etc.)
    """
    results = {
        'experiments_compared': experiments,
        'detailed_comparison': {}
    }

    for exp_name in experiments:
        try:
            data = load_experiment_data(exp_name, window, interval)
            failure_analysis = analyze_failure_types(data['y_test'], data['y_prob'])
            confidence_analysis = analyze_confidence_gap(data['y_test'], data['y_prob'])

            results['detailed_comparison'][exp_name] = {
                'precision': failure_analysis['metrics']['precision'],
                'recall': failure_analysis['metrics']['recall'],
                'f1': failure_analysis['metrics']['f1'],
                'fp_rate': failure_analysis['fp_analysis']['percentage'],
                'fn_rate': failure_analysis['fn_analysis']['percentage'],
                'high_conf_fp': confidence_analysis['high_confidence_false_positives']['count'],
                'high_conf_fn': confidence_analysis['high_confidence_false_negatives']['count'],
                'ambiguous_pct': confidence_analysis['ambiguous_predictions']['percentage']
            }
        except Exception as e:
            results['detailed_comparison'][exp_name] = {'error': str(e)}

    # Generate insights
    valid_exps = {k: v for k, v in results['detailed_comparison'].items() if 'error' not in v}

    if valid_exps:
        # Find best for each metric
        results['insights'] = {
            'best_precision': max(valid_exps.keys(), key=lambda x: valid_exps[x]['precision']),
            'best_recall': max(valid_exps.keys(), key=lambda x: valid_exps[x]['recall']),
            'best_f1': max(valid_exps.keys(), key=lambda x: valid_exps[x]['f1']),
            'most_selective': min(valid_exps.keys(), key=lambda x: valid_exps[x]['fp_rate']),
            'most_aggressive': max(valid_exps.keys(), key=lambda x: valid_exps[x]['recall']),
        }

        results['trade_off_analysis'] = '''
Model Characteristics:
- High Precision / Low Recall: Conservative, misses subtle failures but avoids false alarms
- Low Precision / High Recall: Aggressive, catches more failures but more false alarms
- High Ambiguous %: Model uncertain, may need more discriminative features
- High Conf FP: Model sees normal patterns as failures (over-fitting to failure patterns)
- High Conf FN: Silent/sudden failures that don't match learned patterns
'''

    return results


def plot_failure_distribution(y_test: np.ndarray, y_prob: np.ndarray,
                              save_path: Optional[Path] = None):
    """Generate visualization for failure analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    y_pred = (y_prob >= 0.5).astype(int)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Normal', 'Incident'],
                yticklabels=['Normal', 'Incident'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    # 2. Probability Distribution by Actual Class
    axes[0, 1].hist(y_prob[y_test == 0], bins=50, alpha=0.5, label='Actual Normal', color='green')
    axes[0, 1].hist(y_prob[y_test == 1], bins=50, alpha=0.5, label='Actual Incident', color='red')
    axes[0, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Probability Distribution by Actual Class')
    axes[0, 1].legend()

    # 3. Confidence of Errors
    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    error_data = []
    error_labels = []
    if np.sum(fp_mask) > 0:
        error_data.append(y_prob[fp_mask])
        error_labels.append('False Positives\n(False Alarms)')
    if np.sum(fn_mask) > 0:
        error_data.append(y_prob[fn_mask])
        error_labels.append('False Negatives\n(Missed Failures)')

    if error_data:
        axes[1, 0].boxplot(error_data, labels=error_labels)
        axes[1, 0].set_ylabel('Predicted Probability')
        axes[1, 0].set_title('Confidence Distribution of Errors')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    # 4. Calibration-like plot
    prob_bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_prob, prob_bins) - 1
    bin_indices = np.clip(bin_indices, 0, 9)

    bin_accuracies = []
    bin_counts = []
    for i in range(10):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(y_test[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)

    bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
    axes[1, 1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, label='Actual Rate')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Actual Incident Rate')
    axes[1, 1].set_title('Model Calibration')
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.close()


def load_X_test(experiment_name: str, window: int = 48, interval: str = "30min") -> Optional[np.ndarray]:
    """Try to load X_test data for attention analysis."""
    exp_dir = OUTPUT_DIR / f"{interval}_win{window}_{experiment_name}"

    # Try to find X_test file
    x_test_files = list(exp_dir.glob("X_test_*.npy"))
    if x_test_files:
        return np.load(x_test_files[0])

    # Try alternative naming
    x_test_files = list(exp_dir.glob("*X_test*.npy"))
    if x_test_files:
        return np.load(x_test_files[0])

    return None


def run_full_analysis(experiment_name: str, window: int = 48,
                      interval: str = "30min") -> Dict:
    """Run all 5 analysis strategies for a single experiment."""

    print(f"\n{'='*60}")
    print(f"DEEP ANALYSIS: {experiment_name}")
    print(f"{'='*60}")

    # Load data
    data = load_experiment_data(experiment_name, window, interval)
    y_test, y_prob = data['y_test'], data['y_prob']

    print(f"Loaded {len(y_test)} samples")
    print(f"Class distribution: Normal={np.sum(y_test==0)}, Incident={np.sum(y_test==1)}")

    # Try to load X_test for attention analysis
    X_test = load_X_test(experiment_name, window, interval)
    if X_test is not None:
        print(f"Loaded X_test: shape={X_test.shape}")
    else:
        print("X_test not found - attention analysis will be limited")

    results = {
        'experiment': experiment_name,
        'config': {'window': window, 'interval': interval},
        'sample_size': len(y_test)
    }

    # Strategy 1: Failure Type Analysis
    print("\n[1/5] Analyzing Failure Types...")
    results['failure_types'] = analyze_failure_types(y_test, y_prob)

    # Strategy 2: Confidence Gap Analysis
    print("[2/5] Analyzing Confidence Gaps...")
    results['confidence_gap'] = analyze_confidence_gap(y_test, y_prob)

    # Strategy 3: Temporal Lead-Time Analysis
    print("[3/5] Analyzing Temporal Lead-Time...")
    results['temporal'] = analyze_temporal_leadtime(y_test, y_prob)

    # Strategy 4: Attention Pattern Analysis
    print("[4/5] Analyzing Attention Patterns...")
    results['attention'] = analyze_attention_patterns(
        experiment_name, window, interval,
        X_test=X_test, y_test=y_test
    )

    # Generate plots
    print("[5/5] Generating Visualizations...")
    plot_path = ANALYSIS_DIR / f"analysis_{experiment_name}.png"
    plot_failure_distribution(y_test, y_prob, plot_path)
    results['plot_path'] = str(plot_path)

    # Save results
    results_path = ANALYSIS_DIR / f"analysis_{experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return results


def print_summary(results: Dict):
    """Print a human-readable summary of analysis results."""

    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    # Failure Types
    ft = results['failure_types']
    print("\n[1] FAILURE TYPE ANALYSIS:")
    print(f"   Precision: {ft['metrics']['precision']*100:.2f}%")
    print(f"   Recall: {ft['metrics']['recall']*100:.2f}%")
    print(f"   F1 Score: {ft['metrics']['f1']*100:.2f}%")
    print(f"\n   False Positives (False Alarms): {ft['fp_analysis']['count']}")
    print(f"   -> Avg Confidence: {ft['fp_analysis']['avg_confidence']:.3f}")
    print(f"   -> High Confidence FPs (>0.8): {ft['fp_analysis']['high_confidence_fp']}")
    print(f"\n   False Negatives (Missed Failures): {ft['fn_analysis']['count']}")
    print(f"   -> Low Confidence FNs (<0.2): {ft['fn_analysis']['low_confidence_fn']}")

    # Confidence Gap
    cg = results['confidence_gap']
    print("\n[2] CONFIDENCE GAP ANALYSIS:")
    print(f"   Total Wrong Predictions: {cg['total_wrong']}")
    print(f"   High-Confidence False Positives: {cg['high_confidence_false_positives']['count']}")
    print(f"   High-Confidence False Negatives: {cg['high_confidence_false_negatives']['count']}")
    print(f"   Ambiguous Predictions (0.4-0.6): {cg['ambiguous_predictions']['percentage']:.1f}%")

    # Temporal
    temp = results['temporal']
    print("\n[3] TEMPORAL LEAD-TIME ANALYSIS:")
    print(f"   Total Incidents: {temp['total_incidents']}")
    print(f"   Avg Lead Time: {temp['lead_time_stats']['avg_lead_time_hours']:.1f} hours")
    print(f"   Early Detections (2h+): {temp['detection_quality']['early_detections_2h_plus']}")
    print(f"   Late Detections: {temp['detection_quality']['late_detections']}")
    print(f"   Missed Detections: {temp['detection_quality']['missed_detections']}")
    print(f"   Early Detection Rate: {temp['detection_quality']['early_detection_rate']:.1f}%")

    # Key Weaknesses
    print("\n[!] KEY WEAKNESSES IDENTIFIED:")

    if ft['fn_analysis']['low_confidence_fn'] > ft['fn_analysis']['count'] * 0.5:
        print("   * Model confidently misses failures -> Sudden/silent failures without degradation pattern")

    if cg['high_confidence_false_positives']['count'] > 10:
        print("   * High-confidence false alarms -> Model confuses recovery patterns with failure patterns")

    if temp['detection_quality']['early_detection_rate'] < 50:
        print("   * Low early detection rate -> Model only reacts to immediate pre-failure patterns")

    if cg['ambiguous_predictions']['percentage'] > 20:
        print("   * High ambiguity rate -> Features may not be discriminative enough")


def main():
    parser = argparse.ArgumentParser(description='Deep Analysis of Predictive Maintenance Model')
    parser.add_argument('--experiment', '-e', type=str,
                        help='Single experiment name to analyze')
    parser.add_argument('--compare', '-c', type=str,
                        help='Comma-separated experiment names to compare')
    parser.add_argument('--window', '-w', type=int, default=48,
                        help='Window size (default: 48)')
    parser.add_argument('--interval', '-i', type=str, default='30min',
                        help='Sampling interval (default: 30min)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available experiments')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for exp_dir in sorted(OUTPUT_DIR.glob("*")):
            if exp_dir.is_dir():
                print(f"  - {exp_dir.name}")
        return

    if args.experiment:
        results = run_full_analysis(args.experiment, args.window, args.interval)
        print_summary(results)

    elif args.compare:
        experiments = [e.strip() for e in args.compare.split(',')]
        print(f"\nComparing {len(experiments)} experiments...")

        comparison = compare_feature_sets(experiments, args.window, args.interval)

        # Save comparison
        comp_path = ANALYSIS_DIR / f"comparison_{'_vs_'.join(experiments[:3])}.json"
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        # Print comparison table
        print("\n" + "="*80)
        print("FEATURE SET COMPARISON")
        print("="*80)
        print(f"\n{'Experiment':<40} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Ambig%':>10}")
        print("-"*80)

        for exp_name, metrics in comparison['detailed_comparison'].items():
            if 'error' not in metrics:
                print(f"{exp_name:<40} {metrics['precision']*100:>9.2f}% {metrics['recall']*100:>9.2f}% {metrics['f1']*100:>9.2f}% {metrics['ambiguous_pct']:>9.1f}%")
            else:
                print(f"{exp_name:<40} ERROR: {metrics['error']}")

        print("\n" + comparison.get('trade_off_analysis', ''))
        print(f"\nResults saved to: {comp_path}")

    else:
        parser.print_help()
        print("\n\nExample usage:")
        print("  python scripts/deep_analysis.py --list")
        print("  python scripts/deep_analysis.py -e f13-service_moderate-Attention")
        print("  python scripts/deep_analysis.py --compare f11-SMOTE-Attention,f13-SMOTE-Attention,f15-SMOTE-Attention")


if __name__ == "__main__":
    main()
