"""
Attention Analysis Script for LSTM Models

Loads model checkpoint and generates attention heatmaps to understand
which timesteps the model focuses on for predictions.

Usage:
    python scripts/analyze_attention.py --model f13-SMOTE-Attention
    python scripts/analyze_attention.py --model f13-SMOTE-Attention --strategy service_moderate
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import pickle
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import prepare_training_data, FEATURE_STRATEGIES
from sklearn.model_selection import train_test_split

CHECKPOINTS_DIR = PROJECT_ROOT / "src" / "model" / "checkpoints"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


def find_model(experiment_name: str) -> Path:
    """Find model checkpoint file."""
    for f in CHECKPOINTS_DIR.glob("*.pkl"):
        if experiment_name in f.name:
            return f
    raise FileNotFoundError(f"No model found for: {experiment_name}")


def infer_feature_strategy(experiment_name: str) -> str:
    """Infer feature strategy from experiment name."""
    strategy_map = {
        'f2': 'baseline',
        'f7-quick_win': 'quick_win',
        'f7-SMOTE': 'quick_win',
        'f11': 'moderate',
        'f15': 'full',
        'f9-service_quick_win': 'service_quick_win',
        'f13': 'service_moderate',
        'f18-service_full': 'service_full',
        'f9-sequence_quick_win': 'sequence_quick_win',
        'f18-sequence_full': 'sequence_full',
        'f19': 'ultimate',
    }

    for key, strategy in strategy_map.items():
        if key in experiment_name:
            return strategy

    # Default
    return 'service_moderate'


def load_model_from_checkpoint(model_path: Path, input_size: int, hidden_size: int = 64):
    """Load model from checkpoint (handles both full model and params dict)."""

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    # If it's already a model object
    if hasattr(data, 'get_attention_weights'):
        return data

    # If it's a params dictionary, reconstruct the model
    if isinstance(data, dict) and 'W_att' in data:
        # Infer sizes from weights
        # Wf shape is (hidden_size, hidden_size + input_size)
        Wf = data['Wf']
        hidden_size = Wf.shape[0]
        z_dim = Wf.shape[1]
        input_size = z_dim - hidden_size

        print(f"  Reconstructing model: input_size={input_size}, hidden_size={hidden_size}")

        # Try to import model class
        try:
            from src.model.lstm_attention_optimized import AttentionLSTMModelGPUOptimized
            model = AttentionLSTMModelGPUOptimized(input_size, hidden_size, output_size=1)

            # Load parameters - try cupy first, fallback to numpy
            try:
                import cupy as cp
                for key, value in data.items():
                    if key in model.params:
                        model.params[key] = cp.asarray(value)
            except ImportError:
                import numpy as np
                for key, value in data.items():
                    if key in model.params:
                        model.params[key] = np.asarray(value)

            return model

        except Exception as e:
            print(f"  Warning: Could not load full model class: {e}")
            # Return a minimal wrapper for attention analysis
            return MinimalAttentionModel(data)

    raise ValueError(f"Unknown checkpoint format: {type(data)}")


class MinimalAttentionModel:
    """Minimal model wrapper for attention weight extraction."""

    def __init__(self, params: dict):
        self.params = params

        # Infer sizes
        Wf = params['Wf']
        self.hidden_size = Wf.shape[0]
        z_dim = Wf.shape[1]
        self.input_size = z_dim - self.hidden_size

        # Convert to numpy if needed
        for key in self.params:
            if hasattr(self.params[key], 'get'):  # cupy array
                self.params[key] = np.asnumpy(self.params[key])
            else:
                self.params[key] = np.asarray(self.params[key])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def get_attention_weights(self, X_batch):
        """
        Get attention weights for a batch of sequences.
        X_batch: (batch_size, seq_len, input_size)
        Returns: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = X_batch.shape

        # Initialize hidden states
        h = np.zeros((self.hidden_size, batch_size), dtype=np.float32)
        c = np.zeros((self.hidden_size, batch_size), dtype=np.float32)

        all_h = []

        # Forward pass through LSTM
        for t in range(seq_len):
            x_t = X_batch[:, t, :].T  # (input_size, batch_size)

            # Concatenate h and x
            z = np.vstack((h, x_t))  # (hidden_size + input_size, batch_size)

            # Gates
            f = self.sigmoid(np.dot(self.params['Wf'], z) + self.params['bf'])
            i = self.sigmoid(np.dot(self.params['Wi'], z) + self.params['bi'])
            c_bar = np.tanh(np.dot(self.params['Wc'], z) + self.params['bc'])
            o = self.sigmoid(np.dot(self.params['Wo'], z) + self.params['bo'])

            # Update cell and hidden state
            c = f * c + i * c_bar
            h = o * np.tanh(c)

            all_h.append(h.copy())

        # Stack all hidden states: (seq_len, hidden_size, batch_size)
        H = np.stack(all_h, axis=0)

        # Attention mechanism
        # S = tanh(W_att @ H)
        S = np.tanh(np.tensordot(self.params['W_att'], H, axes=([1], [1])))
        S = np.transpose(S, (1, 0, 2))  # (seq_len, hidden_size, batch_size)

        # e = v_att^T @ S
        e = np.tensordot(self.params['v_att'].flatten(), S, axes=([0], [1]))
        # e: (seq_len, batch_size)

        # Softmax over time
        e_max = np.max(e, axis=0, keepdims=True)
        e_exp = np.exp(e - e_max)
        alpha = e_exp / (np.sum(e_exp, axis=0, keepdims=True) + 1e-9)
        # alpha: (seq_len, batch_size)

        return alpha.T  # (batch_size, seq_len)


def analyze_attention(model_path: Path, X_test: np.ndarray, y_test: np.ndarray,
                      experiment_name: str, sample_size: int = 200):
    """Analyze attention weights from model."""

    print(f"Loading model from: {model_path}")

    # Get input size from X_test
    input_size = X_test.shape[2]

    try:
        model = load_model_from_checkpoint(model_path, input_size)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

    if not hasattr(model, 'get_attention_weights'):
        print("ERROR: Model does not have attention mechanism!")
        print(f"Model type: {type(model).__name__}")
        return None

    # Sample data
    n_samples = min(sample_size, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    print(f"Computing attention weights for {n_samples} samples...")
    attention_weights = model.get_attention_weights(X_sample)
    print(f"Attention weights shape: {attention_weights.shape}")

    # Analyze
    mean_attention = attention_weights.mean(axis=0)
    seq_len = len(mean_attention)

    # Temporal analysis
    recent_6h = mean_attention[-12:].sum()  # Last 6 hours
    historical = mean_attention[:-12].sum()

    print("\n" + "="*60)
    print("ATTENTION ANALYSIS RESULTS")
    print("="*60)

    print(f"\nTemporal Distribution:")
    print(f"  Recent 6h (last 12 steps): {recent_6h*100:.1f}%")
    print(f"  Historical (earlier):      {historical*100:.1f}%")
    print(f"  Recent/Historical ratio:   {recent_6h/historical:.2f}x")

    # Quarter analysis
    quarter = seq_len // 4
    q1 = mean_attention[:quarter].sum()
    q2 = mean_attention[quarter:2*quarter].sum()
    q3 = mean_attention[2*quarter:3*quarter].sum()
    q4 = mean_attention[3*quarter:].sum()

    print(f"\nQuarter Distribution:")
    print(f"  Q1 (oldest 25%):  {q1*100:.1f}%")
    print(f"  Q2 (25-50%):      {q2*100:.1f}%")
    print(f"  Q3 (50-75%):      {q3*100:.1f}%")
    print(f"  Q4 (recent 25%):  {q4*100:.1f}%")

    # Peak detection
    peak_threshold = mean_attention.mean() + mean_attention.std()
    peak_indices = np.where(mean_attention > peak_threshold)[0]
    print(f"\nPeak Attention Timesteps:")
    for idx in peak_indices:
        hours_before = (seq_len - idx) * 0.5
        print(f"  Step {idx}: {hours_before:.1f}h before prediction (weight: {mean_attention[idx]:.4f})")

    # Class-specific analysis
    normal_mask = y_sample == 0
    incident_mask = y_sample == 1

    if np.sum(normal_mask) > 5 and np.sum(incident_mask) > 5:
        normal_attention = attention_weights[normal_mask].mean(axis=0)
        incident_attention = attention_weights[incident_mask].mean(axis=0)

        print(f"\nClass-Specific Analysis:")
        print(f"  Normal samples: {np.sum(normal_mask)}")
        print(f"    Recent 6h focus: {normal_attention[-12:].sum()*100:.1f}%")
        print(f"  Incident samples: {np.sum(incident_mask)}")
        print(f"    Recent 6h focus: {incident_attention[-12:].sum()*100:.1f}%")

        if incident_attention[-12:].sum() > normal_attention[-12:].sum():
            print("  -> Model focuses MORE on recent history for incidents")
        else:
            print("  -> Model focuses MORE on historical patterns for incidents")

    # Interpretation
    print("\n" + "-"*60)
    print("INTERPRETATION:")
    if recent_6h / historical > 2:
        print("  [!] HIGH RECENCY BIAS: Model heavily focuses on last 6 hours")
        print("      This may cause it to miss slow degradation patterns")
        print("      Model acts more like instant classifier than sequence processor")
    elif q4 > q1 + q2:
        print("  [~] MODERATE RECENCY BIAS: Model prefers recent timesteps")
        print("      Some historical context used but limited")
    else:
        print("  [OK] BALANCED ATTENTION: Model uses full time window")
        print("      Good sequence processing behavior")

    # Generate visualizations
    print("\n" + "-"*60)
    print("Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean attention bar chart
    colors = ['steelblue'] * (seq_len - 12) + ['coral'] * 12
    axes[0, 0].bar(range(seq_len), mean_attention, color=colors, alpha=0.7)
    axes[0, 0].set_xlabel('Timestep (30min intervals)')
    axes[0, 0].set_ylabel('Mean Attention Weight')
    axes[0, 0].set_title('Average Attention Distribution\n(Red = Last 6 hours)')
    axes[0, 0].axvline(x=seq_len - 12, color='red', linestyle='--', alpha=0.5)

    # 2. Attention heatmap
    n_show = min(50, len(attention_weights))
    # Sort by y_sample for better visualization
    sort_idx = np.argsort(y_sample[:n_show])
    sns.heatmap(attention_weights[:n_show][sort_idx], ax=axes[0, 1], cmap='YlOrRd',
                cbar_kws={'label': 'Attention'})
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Sample (sorted by class)')
    axes[0, 1].set_title(f'Attention Heatmap (n={n_show})')

    # 3. Class comparison
    if np.sum(normal_mask) > 5 and np.sum(incident_mask) > 5:
        x = np.arange(seq_len)
        width = 0.4
        axes[1, 0].bar(x - width/2, normal_attention, width, label='Normal', color='green', alpha=0.6)
        axes[1, 0].bar(x + width/2, incident_attention, width, label='Incident', color='red', alpha=0.6)
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Mean Attention')
        axes[1, 0].set_title('Attention by Class')
        axes[1, 0].legend()
        axes[1, 0].axvline(x=seq_len - 12, color='black', linestyle='--', alpha=0.3)

    # 4. Cumulative attention
    cumsum = np.cumsum(mean_attention)
    axes[1, 1].plot(range(seq_len), cumsum, 'b-', linewidth=2)
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[1, 1].axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')

    idx_50 = np.searchsorted(cumsum, 0.5)
    idx_90 = np.searchsorted(cumsum, 0.9)
    hours_50 = (seq_len - idx_50) * 0.5
    hours_90 = (seq_len - idx_90) * 0.5

    axes[1, 1].axvline(x=idx_50, color='red', linestyle=':', alpha=0.5)
    axes[1, 1].axvline(x=idx_90, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Cumulative Attention')
    axes[1, 1].set_title(f'Cumulative (50% at -{hours_50:.1f}h, 90% at -{hours_90:.1f}h)')
    axes[1, 1].legend()

    plt.suptitle(f'Attention Analysis: {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = ANALYSIS_DIR / f"attention_{experiment_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    plt.close()

    return {
        'mean_attention': mean_attention.tolist(),
        'recent_6h_focus': float(recent_6h),
        'historical_focus': float(historical),
        'quarter_distribution': [float(q1), float(q2), float(q3), float(q4)],
        'peak_timesteps': peak_indices.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Attention Analysis for LSTM Models')
    parser.add_argument('--model', '-m', required=True,
                        help='Experiment name (e.g., f13-SMOTE-Attention)')
    parser.add_argument('--strategy', '-s', type=str, default=None,
                        help='Feature strategy (auto-detected if not specified)')
    parser.add_argument('--window', '-w', type=int, default=48,
                        help='Window size (default: 48)')
    parser.add_argument('--samples', '-n', type=int, default=200,
                        help='Number of samples to analyze (default: 200)')
    parser.add_argument('--data', '-d', type=str,
                        default='data/raw_dataset.csv',
                        help='Path to raw dataset')

    args = parser.parse_args()

    # Find model
    model_path = find_model(args.model)
    print(f"Found model: {model_path}")

    # Infer feature strategy
    if args.strategy:
        strategy = args.strategy
    else:
        strategy = infer_feature_strategy(args.model)
    print(f"Using feature strategy: {strategy}")

    # Prepare data
    print(f"\nPreparing data with window={args.window}, strategy={strategy}...")
    data_path = PROJECT_ROOT / args.data

    X, y, timestamps, feature_cols = prepare_training_data(
        str(data_path),
        feature_strategy=strategy,
        sampling_period='30min',
        window_size=args.window
    )

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Split to get test set (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")

    # Run attention analysis
    results = analyze_attention(model_path, X_test, y_test, args.model, args.samples)

    if results:
        import json
        results_path = ANALYSIS_DIR / f"attention_{args.model}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
