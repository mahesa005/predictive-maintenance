# Predictive Maintenance with Custom LSTM

A from-scratch LSTM implementation for time-series anomaly prediction in industrial maintenance systems, built with CuPy for GPU acceleration.

## Project Structure

```
predictive-maintenance/
├── deployment/
│   └── nlstm.html              # Interactive Nested LSTM architecture explanation
├── notebooks/
│   └── train_lstm_optimized_experiments.ipynb  # Main training notebook
├── src/
│   ├── model/                  # LSTM model implementations
│   │   ├── lstm_cupy_optimized.py         # Base optimized LSTM (Adam)
│   │   ├── lstm_attention_optimized.py    # Attention LSTM
│   │   ├── lstm_bidirectional.py          # Bidirectional LSTM
│   │   ├── lstm_bidirectional_attention.py
│   │   ├── lstm_nested.py                 # Nested LSTM
│   │   ├── lstm_nested_attention.py
│   │   ├── lstm_peephole.py               # Peephole LSTM
│   │   ├── lstm_cifg.py                   # Coupled Input-Forget Gate
│   │   ├── lstm_cifg_attention.py
│   │   ├── lstm_glu.py                    # Gated Linear Unit
│   │   ├── lstm_glu_attention.py
│   │   ├── lstm_layernorm.py              # Layer Normalization
│   │   ├── lstm_residual.py               # Residual connections
│   │   └── checkpoints/                   # Saved model weights (.pkl)
│   ├── data/
│   │   └── pipeline.py         # Data preprocessing pipeline
│   └── utils/
│       └── model_io.py         # Model save/load utilities
├── data/
│   ├── output/                 # Experiment outputs (plots, metrics)
│   └── splits/                 # Train/val/test data splits
└── scripts/                    # Utility scripts
```

## Nested LSTM Explanation (Interactive HTML)

The `deployment/nlstm.html` file contains an interactive visual explanation of the Nested LSTM architecture.

**To view it with Live Server in VS Code:**
1. Install the "Live Server" extension in VS Code
2. Right-click on `deployment/nlstm.html`
3. Select "Open with Live Server"
4. The page will open in your browser with hot-reload enabled

Alternatively, you can simply double-click the HTML file to open it directly in your browser.

## Main Notebook: train_lstm_optimized_experiments.ipynb

This is the primary notebook for running experiments. **Important: The notebook is designed for flexible experimentation, requiring you to comment/uncomment certain sections to enable different workflows.**

### Notebook Cell Structure

| Cell | Section | Description |
|------|---------|-------------|
| 0 | Header | Markdown title |
| 1 | Imports | Load all libraries and model classes |
| 2 | Load Data | Load the incident dataset |
| 3 | Preprocess Data | Apply feature engineering and scaling |
| 4 | Check Ratios | Inspect class balance |
| 5 | Base Functions | Helper functions for evaluation |
| 6 | Training | `train_eval()` function - main training logic |
| 7 | Save Model | Save trained model to checkpoint |
| 8 | Feature Grid Search Function | Grid search over feature strategies |
| 9 | Feature Grid Search | Execute grid search (disabled by default) |
| 10 | **Run Experiment** | **Main entry point - configure and run here** |
| 11 | Visualization | Plot final results comparison |
| 12 | Results Export | Export results to CSV |

### How to Use the Notebook

#### Step 1: Run Setup Cells (Cells 1-6)
Run cells 1 through 6 sequentially to load libraries, data, and define functions.

#### Step 2: Configure Your Experiment (Cell 10)
Cell 10 is where you configure your experiment. Key settings to modify:

```python
# 1. SELECT FEATURE STRATEGY - uncomment ONE:
# FEATURE_STRATEGY = 'baseline'           # 2 features (Priority, Impact only)
# FEATURE_STRATEGY = 'quick_win'          # 7 features (+ interactions)
# FEATURE_STRATEGY = 'service_moderate'   # 13 features (+ service features)
# FEATURE_STRATEGY = 'full'               # 15 features (+ temporal)
FEATURE_STRATEGY = 'ultimate'             # 19 features (all features)

# 2. SELECT MODEL TYPE - modify in the train_eval() call:
# Options: 'Optimized', 'Attention', 'Nested', 'BiDirectional',
#          'Peephole', 'CIFG', 'GLU', 'LayerNorm', 'Residual',
#          'BiDirectional-Attention', 'Nested-Attention', etc.

# 3. EXPERIMENT PARAMETERS - modify these variables:
SAMPLING_PERIOD = '30min'  # '30min', '1h', '2h'
WINDOW_SIZE = 48           # Lookback window (48 = 24 hours at 30min)
```

#### Step 3: Comment/Uncomment Workflows

**For Single Model Training:**
- Uncomment the single `train_eval()` call in cell 10
- Comment out any loop-based experiments

**For Grid Search Over Features:**
- Run cell 8 to define the grid search function
- Run cell 9 to execute (uncomment the execution line)

**For Model Comparison Loop:**
- Uncomment the MODEL_REGISTRY loop in cell 10
- This trains multiple model types sequentially

#### Step 4: Run Training (Cell 10)
Execute cell 10 to start training. Progress will be printed with:
- Epoch-by-epoch loss and validation metrics
- Early stopping notifications
- Final test set evaluation

#### Step 5: Save Results (Cells 7, 11-12)
- Cell 7: Save trained model checkpoint
- Cell 11: Generate visualization plots
- Cell 12: Export metrics to CSV

### Output Locations

All experiment outputs are saved to `data/output/<experiment_name>/`:

```
data/output/30min_win48_f13-SMOTE-Attention/
├── cm_opt_30min_win48_f13-SMOTE-Attention.png     # Confusion matrix
├── pr_curve_30min_win48_f13-SMOTE-Attention.png   # Precision-recall curve
└── metrics_30min_win48_f13-SMOTE-Attention.csv    # Metrics (F1, AUC, etc.)
```

Model checkpoints are saved to `src/model/checkpoints/`:
```
src/model/checkpoints/
└── lstm_attention_30min_win48_f13-SMOTE-Attention.pkl
```

### Naming Convention

Experiment names follow this pattern:
```
<sampling>_<window>_<features>-<balancing>-<model>
```
Example: `30min_win48_f13-SMOTE-Attention`
- `30min`: 30-minute sampling period
- `win48`: Window size of 48 timesteps (24 hours lookback)
- `f13`: Feature strategy with 13 features
- `SMOTE`: Class balancing method used
- `Attention`: Model architecture

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 12.x)
pip install cupy-cuda12x
```

## Best Performing Models

| Model | F1 Score | ROC-AUC |
|-------|----------|---------|
| f19-ultimate-Nested | 82.94% | 90.94% |
| f15-SMOTE-Attention | 82.31% | 91.17% |
| f13-service_moderate-Attention | 82.53% | 90.90% |
