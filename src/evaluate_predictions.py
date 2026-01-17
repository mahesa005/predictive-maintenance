"""
Script untuk mengevaluasi prediksi dengan berbagai metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
"""
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt


def evaluate_predictions(input_file: str, threshold: float = 0.5, save_plot: bool = True):
    """
    Mengevaluasi prediksi dengan berbagai metrik.
    
    Args:
        input_file: Path ke file predictions CSV (harus memiliki kolom: actual, predicted, probability)
        threshold: Threshold untuk binary classification (default: 0.5)
        save_plot: Apakah menyimpan plot ROC curve (default: True)
    """
    # Baca file predictions
    df = pd.read_csv(input_file)
    
    print("=" * 60)
    print("EVALUASI MODEL PREDIKSI")
    print("=" * 60)
    print(f"\nFile: {input_file}")
    print(f"Jumlah data: {len(df)}")
    print(f"Threshold: {threshold}")
    
    # Ambil nilai actual dan probability
    y_true = df['actual'].values
    y_prob = df['probability'].values
    
    # Buat prediksi berdasarkan threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Hitung metrik
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Print hasil
    print("\n" + "=" * 60)
    print("METRIK EVALUASI")
    print("=" * 60)
    print(f"\n{'Metrik':<20} {'Nilai':>15}")
    print("-" * 35)
    print(f"{'Accuracy':<20} {accuracy:>15.4f} ({accuracy*100:.2f}%)")
    print(f"{'Precision':<20} {precision:>15.4f} ({precision*100:.2f}%)")
    print(f"{'Recall':<20} {recall:>15.4f} ({recall*100:.2f}%)")
    print(f"{'F1-Score':<20} {f1:>15.4f} ({f1*100:.2f}%)")
    print(f"{'ROC AUC':<20} {roc_auc:>15.4f} ({roc_auc*100:.2f}%)")
    
    # Confusion Matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"\n{'':>15} {'Predicted 0':>15} {'Predicted 1':>15}")
    print("-" * 45)
    print(f"{'Actual 0':<15} {tn:>15} {fp:>15}")
    print(f"{'Actual 1':<15} {fn:>15} {tp:>15}")
    
    # Distribusi
    print("\n" + "=" * 60)
    print("DISTRIBUSI DATA")
    print("=" * 60)
    print(f"\nActual 0: {(y_true == 0).sum()} ({(y_true == 0).mean()*100:.2f}%)")
    print(f"Actual 1: {(y_true == 1).sum()} ({(y_true == 1).mean()*100:.2f}%)")
    print(f"\nPredicted 0: {(y_pred == 0).sum()} ({(y_pred == 0).mean()*100:.2f}%)")
    print(f"Predicted 1: {(y_pred == 1).sum()} ({(y_pred == 1).mean()*100:.2f}%)")
    
    # Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
    
    # Plot ROC Curve
    if save_plot:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Tambahkan titik optimal (Youden's J statistic)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='green', s=100, 
                   label=f'Optimal Threshold = {best_threshold:.4f}')
        plt.legend(loc='lower right', fontsize=10)
        
        output_plot = input_file.replace('.csv', '_roc_curve.png')
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nROC Curve disimpan ke: {output_plot}")
        print(f"Optimal Threshold (Youden's J): {best_threshold:.4f}")
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def compare_thresholds(input_file: str, thresholds: list = None):
    """
    Membandingkan performa model pada berbagai threshold.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    df = pd.read_csv(input_file)
    y_true = df['actual'].values
    y_prob = df['probability'].values
    
    print("\n" + "=" * 80)
    print("PERBANDINGAN THRESHOLD")
    print("=" * 80)
    print(f"\n{'Threshold':>10} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'ROC AUC':>12}")
    print("-" * 70)
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        
        print(f"{thresh:>10.2f} {acc:>12.4f} {prec:>12.4f} {rec:>12.4f} {f1:>12.4f} {auc:>12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions with multiple metrics")
    parser.add_argument("--input", "-i", type=str, 
                        default="../data/predictions_gpu_optimized.csv",
                        help="Path ke file predictions CSV")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Threshold untuk binary classification (default: 0.5)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Jangan simpan plot ROC curve")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Bandingkan berbagai threshold")
    
    args = parser.parse_args()
    
    # Evaluasi utama
    evaluate_predictions(args.input, args.threshold, save_plot=not args.no_plot)
    
    # Perbandingan threshold jika diminta
    if args.compare:
        compare_thresholds(args.input)
