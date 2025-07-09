import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_clean_data(csv_path):
    """
    Carica un file CSV e restituisce un DataFrame pandas.

    Args:
        csv_path (str): Percorso del file CSV da caricare.
    Returns:
        pd.DataFrame: DataFrame contenente i dati caricati.
    """
    return pd.read_csv(csv_path)
    
def show_distribution(df, column):
    """
    Mostra la distribuzione di una variabile numerica tramite istogramma con KDE.

    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        column (str): Nome della colonna numerica da visualizzare.
    Returns:
        Mostra un grafico a schermo.
    """
    if column not in df.columns:
        print(f"La colonna '{column}' non è presente nel dataset.")
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribuzione della variabile '{column}'")
    plt.xlabel(column)
    plt.ylabel("Frequenza")
    plt.tight_layout()
    plt.show()

def plot_feature_distribution(df, column):
    """
    Mostra un istogramma semplice della distribuzione di una variabile numerica.

    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        column (str): Nome della colonna numerica da visualizzare.
    Returns:
        None: Mostra un grafico a schermo.
    """
    if column not in df.columns:
        print(f"La colonna '{column}' non è presente nel dataset.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Distribuzione della variabile '{column}'")
    plt.xlabel(column)
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_target_distribution(df, feature):
    """
    Visualizza la distribuzione di una feature categorica rispetto alla variabile target.

    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        feature (str): Nome della feature categorica da analizzare.
    Returns:
        None: Mostra un grafico a schermo.
    """
    if feature not in df.columns:
        print(f"La colonna '{feature}' non è presente nel dataset.")
        return
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=feature, hue='target', palette='pastel')
    plt.title(f"Distribuzione di '{feature}' rispetto al target")
    plt.xlabel(feature)
    plt.ylabel("Conteggio")
    plt.tight_layout()
    plt.show()
