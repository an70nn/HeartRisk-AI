import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, f1_score

# Mostra heatmap del False Positive Rate incrociando due feature categoriche
def plot_fpr_heatmap(df, y_true, y_pred, feature_row, feature_col):
    """
    Calcola e mostra una heatmap con il False Positive Rate (FPR) incrociando due feature categoriche.
    
    Args:
        df (pd.DataFrame): Dataframe con le features originali.
        y_true (pd.Series or np.array): Etichette vere (0/1).
        y_pred (np.array): Predizioni del modello (0/1).
        feature_row (str): Nome feature categoriale per le righe della heatmap.
        feature_col (str): Nome feature categoriale per le colonne della heatmap.
    Returns:
        Mostra il grafico a schermo.
    """

    # Costruiamo una matrice pivot per FPR
    unique_rows = df[feature_row].unique()
    unique_cols = df[feature_col].unique()

    fpr_matrix = pd.DataFrame(index=unique_rows, columns=unique_cols, dtype=float)

    for r in unique_rows:
        for c in unique_cols:
            mask = (df[feature_row] == r) & (df[feature_col] == c)
            if mask.sum() == 0:
                fpr_matrix.loc[r, c] = np.nan
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]

            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            if cm.shape != (2, 2):
                fpr_matrix.loc[r, c] = np.nan
                continue
            tn, fp, fn, tp = cm.ravel()

            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            fpr_matrix.loc[r, c] = fpr

    plt.figure(figsize=(8, 6))
    sns.heatmap(fpr_matrix, annot=True, fmt=".2f", cmap="Reds", cbar_kws={'label': 'FPR'})
    plt.title(f"Heatmap FPR: '{feature_row}' vs '{feature_col}'")
    plt.xlabel(feature_col)
    plt.ylabel(feature_row)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """
    Calcola e mostra la matrice di confusione tra valori veri e predetti.

    Args:
        y_true (array-like): Etichette vere.
        y_pred (array-like): Etichette predette dal modello.
    Returns:
        Mostra il grafico a schermo.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Matrice di Confusione")
    plt.tight_layout()
    plt.show()

def plot_model_comparison(accuracy, f1_logistic_regression, baseline_accuracy, baseline_f1_score):
    """
    Genera e mostra un grafico a barre che confronta le prestazioni
    del modello di Regressione Logistica ottimizzato con una baseline.

    Args:
        accuracy (float): Accuratezza del modello di Regressione Logistica.
        f1_logistic_regression (float): F1-score del modello di Regressione Logistica.
        baseline_accuracy (float): Accuratezza del modello baseline.
        baseline_f1_score (float): F1-score del modello baseline (per la classe positiva).
    Returns:
        Mostra il grafico a schermo.
    """
    comparison_data = {
        'Modello': ['Regressione Logistica', 'Baseline (Maggiore Frequenza)'],
        'Accuracy': [accuracy, baseline_accuracy],
        'F1 Score': [f1_logistic_regression, baseline_f1_score]
    }
    df_comparison = pd.DataFrame(comparison_data)

    df_plot = df_comparison.melt('Modello', var_name='Metrica', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Modello', y='Score', hue='Metrica', data=df_plot, palette='viridis')
    plt.title('Confronto delle Prestazioni: Modello Ottimizzato vs Baseline')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.0)
    plt.legend(title='Metrica')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def save_metrics_to_file(output_dir, accuracy, f1_score_model, baseline_accuracy, baseline_f1_score, classification_report_text, cm):
    """
    Salva le metriche del modello in un file di testo.

    Args:
        output_dir (str): Directory in cui salvare il file.
        accuracy (float): Accuratezza del modello di Regressione Logistica.
        f1_score_model (float): F1-score del modello di Regressione Logistica (per la classe positiva).
        baseline_accuracy (float): Accuratezza del modello baseline.
        baseline_f1_score (float): F1-score del modello baseline (per la classe positiva).
        classification_report_text (str): Report classificazione in formato stringa.
        cm (np.ndarray): Matrice di confusione.
    Returns:
        Salva il file e stampa il percorso di salvataggio.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "model_metrics.txt")

    with open(report_path, "w") as f:
        f.write(f"Accuratezza Modello: {accuracy:.4f}\n")
        f.write(f"F1 Score Modello (Classe 1): {f1_score_model:.4f}\n")
        f.write(f"Accuratezza Baseline: {baseline_accuracy:.4f}\n")
        f.write(f"F1 Score Baseline (Classe 1): {baseline_f1_score:.4f}\n\n")
        f.write("Report di Classificazione:\n")
        f.write(classification_report_text)
        f.write("\nMatrice di Confusione:\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")

    print("\nMetriche salvate in: ")
    print(f"{report_path}")