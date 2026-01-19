# Solusi Underfitting di Akurasi 0.6883

## Diagnosis Masalah

Model **stuck di 68.83%** karena:
1. ✗ **Model capacity terlalu kecil** (hidden_size=128)
2. ✗ **Learning rate terlalu rendah** (0.001)
3. ✗ **Batch size terlalu besar** (512)
4. ✗ **SMOTE dinonaktifkan** (class imbalance tidak ditangani)
5. ✗ **Tidak ada hidden layers tambahan**

---

## REKOMENDASI PERBAIKAN

### 1️⃣ TINGKATKAN MODEL CAPACITY

**Ubah hyperparameter di notebook:**

```python
# BEFORE (Underfitting)
HIDDEN_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 512
PATIENCE = 10
LAMBDA_L2 = 0.0005

# AFTER (Improved)
HIDDEN_SIZE = 256        # ↑ dari 128 → 256
EPOCHS = 150             # ↑ training lebih lama
LEARNING_RATE = 0.005    # ↑ dari 0.001 → 0.005 (5x lebih cepat)
BATCH_SIZE = 128         # ↓ dari 512 → 128 (update lebih sering)
PATIENCE = 15            # ↑ toleransi lebih lama
LAMBDA_L2 = 0.001        # ↑ lebih kuat regularisasi
```

**Impact:**
- `HIDDEN_SIZE=256`: Model 2x lebih powerful, bisa learn pattern lebih kompleks
- `LEARNING_RATE=0.005`: Update weights lebih agresif
- `BATCH_SIZE=128`: Lebih frequent gradient updates → faster convergence

---

### 2️⃣ AKTIFKAN SMOTE KEMBALI

**Di notebook, uncomment SMOTE section:**

```python
# Apply SMOTE on sequences (flatten -> SMOTE -> reshape)
print("⚠️ Applying SMOTE on training sequences (not validation)")

# Step 1: Flatten 3D sequences to 2D for SMOTE
nsamples, nsteps, nfeatures = X_train_seq.shape
X_train_flattened = X_train_seq.reshape((nsamples, nsteps * nfeatures))

# Step 2: Apply SMOTE on flattened sequences
smote = SMOTE(random_state=42)
X_resampled_flat, y_train_resampled = smote.fit_resample(X_train_flattened, y_train_seq)

# Step 3: Reshape back to 3D for LSTM
X_train_seq_resampled = X_resampled_flat.reshape((X_resampled_flat.shape[0], nsteps, nfeatures))

print(f"✅ Training balanced: {np.sum(y_train_resampled == 0)} vs {np.sum(y_train_resampled == 1)}")
```

**Impact:**
- Model belajar dari failure cases lebih baik
- Akurasi pada class minority meningkat

---

### 3️⃣ TAMBAH DROPOUT LAYERS (Opsional - Jika Overfitting Muncul)

Edit [lstm_cupy_optimized.py](src/model/lstm_cupy_optimized.py):

```python
class LSTMModelGPUOptimized:
    def __init__(self, input_size, hidden_size, output_size=1, dropout_rate=0.2):
        self.dropout_rate = dropout_rate
        # ... existing init code ...
    
    def forward_batch(self, X_batch, training=True):
        # ... existing code ...
        
        # Apply dropout before output
        if training and self.dropout_rate > 0:
            mask = cp.random.binomial(1, 1-self.dropout_rate, h.shape) / (1-self.dropout_rate)
            h = h * mask
        
        # ... rest of code ...
```

---

### 4️⃣ QUICK TEST - RUN DENGAN HYPERPARAMETER BARU

```python
# Hyperparameters
HIDDEN_SIZE = 256
EPOCHS = 150
LEARNING_RATE = 0.005
BATCH_SIZE = 128
PATIENCE = 15
LAMBDA_L2 = 0.001

# Training dengan SMOTE aktif
print("Starting improved training...\n")

history = model.train(
    X_train_seq_resampled, y_train_resampled,  # ← SMOTE data
    X_val_seq, y_val_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    print_every=1,
    patience=PATIENCE,
    lambda_l2=LAMBDA_L2
)
```

---

## EXPECTED RESULTS

| Metrik | Before | After |
|--------|--------|-------|
| Val Accuracy | **0.6883** | **0.75-0.85** (estimated) |
| Training Curve | Flat | Steep convergence |
| Class 1 Recall | Low | Higher |

---

## TROUBLESHOOTING

### Jika masih stuck:
- ↑ Increase `HIDDEN_SIZE` ke 512
- ↓ Decrease `LEARNING_RATE` ke 0.01 (lebih aggressive)
- ↓ Decrease `BATCH_SIZE` ke 64 (lebih frequent updates)

### Jika overfitting muncul:
- ↑ Increase `LAMBDA_L2` ke 0.01
- ↓ Decrease `HIDDEN_SIZE` ke 128
- Add dropout layers (lihat step 3)
- ↓ Decrease `EPOCHS`

---

## PRIORITY

1. **HIGH**: Ubah hyperparameter (batasan 5 menit)
2. **HIGH**: Aktifkan SMOTE kembali (2 menit)
3. **MEDIUM**: Eksperi learning rate decay (opsional)
4. **LOW**: Add dropout layers (hanya jika overfitting)
