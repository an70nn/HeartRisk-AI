import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_perturbation(preprocessor, model, X_test, y_test, feature_name):
    """
    Esegue un test di robustezza sul modello analizzando la sensibilità rispetto a perturbazioni su una singola feature.

    La funzione permette di selezionare un esempio specifico dal set di test e di variare manualmente il valore di una feature specificata,
    osservando come cambiano la predizione del modello e la probabilità associata alla classe positiva (malattia cardiaca).

    Args:
        preprocessor : sklearn.pipeline.Pipeline
        model        : modello sklearn compatibile (es. LogisticRegression)
        X_test       : pd.DataFrame contenente le features originali (non processate)
        y_test       : pd.Series contenente le etichette reali corrispondenti a X_test
        feature_name : Il nome della feature su cui applicare le perturbazioni
    Return:
        Mostra un grafico della sensibilità della probabilità rispetto alle perturbazioni.
    """

    print("\nTEST DI ROBUSTEZZA – PERTURBAZIONI SU INPUT")
    
    # Dizionario con suggerimenti sui range per ogni feature
    feature_ranges = {
        'age':        "Età (anni): valori tipici da 29 a 77",
        'trestbps':   "Pressione a riposo (mm Hg): 94 – 200",
        'chol':       "Colesterolo (mg/dl): 126 – 564",
        'thalach':    "Frequenza cardiaca max (bpm): 71 – 202",
        'oldpeak':    "Oldpeak (depressione ST): 0.0 – 6.2",
        'sex':        "Sesso: 0 = Donna, 1 = Uomo",
        'cp':         "Tipo di dolore toracico (cp): 0 – 3",
        'fbs':        "Glicemia a digiuno > 120 (fbs): 0 = No, 1 = Sì",
        'restecg':    "ECG a riposo: 0 – 2",
        'exang':      "Angina da sforzo: 0 = No, 1 = Sì",
        'slope':      "Pendenza tratto ST: 0 – 2",
        'ca':         "Numero di vasi colorati: 0 – 3",
        'thal':       "Tipo di thalassemia: 1 = normale, 2 = difetto fisso, 3 = reversibile"
    }

    print(f"Analizzando la sensibilità rispetto alla feature: '{feature_name}'")

    # Mostra indice massimo disponibile
    print(f"\nSono disponibili {len(X_test)} esempi.")
    
    try:
        sample_index = int(input(f"Inserisci l’indice del paziente da analizzare (0–{len(X_test)-1}): "))
        sample = X_test.iloc[sample_index].copy()
        true_label = y_test.iloc[sample_index]
    except (ValueError, IndexError):
        print("Indice non valido.")
        return

    # Stampa valori originali
    print(f"\nEsempio selezionato (indice {sample_index})")
    print("Valori originali:")
    print(sample)
    print(f"Etichetta reale: {true_label}")

    if feature_name not in sample:
        print(f"La feature '{feature_name}' non è presente nei dati.")
        return

    # Definisci colonne categoriche per sicurezza
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Calcola predizione e probabilità originali
    original_df = pd.DataFrame([sample.values], columns=sample.index)
    original_df[categorical_cols] = original_df[categorical_cols].astype(str)
    original_processed = preprocessor.transform(original_df)
    original_pred = model.predict(original_processed)[0]
    original_proba = model.predict_proba(original_processed)[0][1]

    print(f"Classe predetta originale: {original_pred}")
    print(f"Probabilità di malattia (originale): {original_proba:.2%}")

    try:
        if feature_name in feature_ranges:
            print()
            print(f"Suggerimento: {feature_ranges[feature_name]}")
        else:
            print("Nessun suggerimento disponibile per questa feature.")

        perturb_values_input = input("Inserisci un valori di perturbazione separati da virgole (es. -1,0,+1,+2): ")
        perturb_values = [float(val.strip()) for val in perturb_values_input.split(",")]
    except ValueError:
        print("Input non valido. Inserire numeri separati da virgola.")
        return

    predictions = []
    probabilities = []

    print("\nVariazioni applicate:")
    for delta in perturb_values:
        sample_copy = sample.copy()
        sample_copy[feature_name] += delta  # Applica perturbazione

        sample_df = pd.DataFrame([sample_copy.values], columns=sample_copy.index)
        sample_df[categorical_cols] = sample_df[categorical_cols].astype(str)

        sample_processed = preprocessor.transform(sample_df)
        proba = model.predict_proba(sample_processed)[0][1]
        pred = model.predict(sample_processed)[0]

        predictions.append(pred)
        probabilities.append(proba)

        print(f"[{sample_copy[feature_name]:.2f}] (Δ{delta:+}) → Pred: {pred} | Probabilità: {proba:.2%}")

    # Plot
    perturbed_values = [sample[feature_name] + d for d in perturb_values]

    plt.figure(figsize=(10, 5))
    plt.plot(perturbed_values, probabilities, marker='o', linestyle='-', color='dodgerblue')
    plt.axhline(0.5, color='red', linestyle='--', label='Soglia decisione (0.5)')
    plt.title(f"Sensibilità del modello rispetto a '{feature_name}'")
    plt.xlabel(f"Valore modificato di '{feature_name}'")
    plt.ylabel("Probabilità di malattia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

   # Analisi variazione
    initial_pred = predictions[0]
    change_detected = False

    for delta, pred in zip(perturb_values, predictions):
        if pred != initial_pred:
            print(f"\nIl modello ha cambiato la previsione almeno una volta! (ad esempio a Δ{delta:+})")
            change_detected = True
            break

    if not change_detected:
        print("\nIl modello è robusto a queste variazioni (previsione costante).")
