import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from preprocessing import load_and_preprocess_data, get_preprocessor
from data_visualization import *
import prediction_cli
import robustness_test
import model_metrics

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FILE_PATH   = os.path.join(os.path.dirname(BASE_DIR), "dataset", "processed.cleveland.data")
OUTPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), "output")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(f"\nINFO: Creazione directory di output:")
    print(f"{os.path.abspath(OUTPUT_PATH)}")
else:
    print(f"\nINFO: Directory di output già esistente")
    print(f"{os.path.abspath(OUTPUT_PATH)}")

print("\nFASE 1:\nCaricamento e pre-elaborazione del dataset")
df = load_and_preprocess_data(FILE_PATH)

cleaned_data_file = "cleaned_data.csv"
df.to_csv(os.path.join(OUTPUT_PATH, cleaned_data_file), index=False)
print(f"\nDati puliti salvati in")
print(f"{os.path.join(OUTPUT_PATH, cleaned_data_file)}")

X = df.drop('target', axis=1)
y = df['target']

numerical_features   = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

preprocessor = get_preprocessor(numerical_features, categorical_features)

print("\nFASE 2:\nDivisione e pre-elaborazione dei dati")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

X_train_file = "X_train_processed.csv"
y_train_file = "y_train.csv"
X_test_file  = "X_test_processed.csv"
y_test_file  = "y_test.csv"

pd.DataFrame(X_train_processed).to_csv(os.path.join(OUTPUT_PATH, X_train_file), index=False)
pd.DataFrame(X_test_processed).to_csv(os.path.join(OUTPUT_PATH, X_test_file), index=False)
y_train.to_csv(os.path.join(OUTPUT_PATH, y_train_file), index=False)
y_test.to_csv(os.path.join(OUTPUT_PATH, y_test_file), index=False)
print(f"Salvataggio dei datasets 'Preprocessati' in")
print(f"{OUTPUT_PATH}")

print("\nFASE 3:\nAddestramento e Valutazione del Modello")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)

# Calcola l'F1 score per il modello (classe 1)
f1_logistic_regression = f1_score(y_test, y_pred, pos_label=1)

report_text = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuratezza: {accuracy:.4f}")
print("\nReport di Classificazione:\n", report_text)
print("Matrice di Confusione:\n", cm)

baseline_accuracy = y_test.value_counts(normalize=True).max()

# Calcola l'F1 score per la baseline 
baseline_f1_score = 0.0 

print(f"\nBaseline accuracy: {baseline_accuracy:.4f}")
if accuracy > baseline_accuracy:
    print("Il modello supera la baseline.")
else:
    print("Il modello NON supera la baseline.")

model_metrics.save_metrics_to_file(OUTPUT_PATH, accuracy, f1_logistic_regression, baseline_accuracy, baseline_f1_score, report_text, cm)

def menu():
    while True:
        print("\nMENU INTERATTIVO")
        print("1. Visualizza grafici EDA")
        print("2. Testa il modello con dati manuali (CLI)")
        print("3. Valuta la robustezza del modello (AML)")
        print("4. Visualizza metriche grafiche del modello")
        print("0. Esci")
        main_choice = input("Scegli un'opzione: ")

        if main_choice == '1':
            df_clean = pd.read_csv(os.path.join(OUTPUT_PATH, cleaned_data_file))
            print("\nTipo di grafico:")
            print("a. Distribuzione di una variabile numerica")
            print("b. Target vs Feature categorica")
            print("c. Heatmap delle correlazioni tra feature numeriche")
            sub = input("Scegli un'opzione: ")

            if sub == 'a':
                col = input("Inserisci nome colonna numerica [age / trestbps / chol / thalach / oldpeak]: ")
                show_distribution(df_clean, col)
            elif sub == 'b':
                feat = input("Inserisci nome feature categorica [sex / cp / fbs / restecg / exang / slope / ca / thal]: ")
                plot_target_distribution(df_clean, feat)
            elif sub == 'c':
                df_clean = pd.read_csv(os.path.join(OUTPUT_PATH, cleaned_data_file))
    
                # Calcola e mostra la heatmap delle correlazioni numeriche
                correlation_matrix = df_clean.corr(numeric_only=True)

                plt.figure(figsize=(12, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                plt.title("Heatmap delle Correlazioni tra le Feature Numeriche")
                plt.tight_layout()
                plt.show()
            else:
                print("Opzione non valida.")

        elif main_choice == '2':
            prediction_cli.test_user_input(preprocessor, model)

        elif main_choice == '3':
            feature_name = input("Inserisci nome feature da perturbare [age / trestbps / chol / thalach / oldpeak / sex / cp / fbs / restecg / exang / slope / ca / thal]: ")
            robustness_test.test_perturbation(preprocessor, model, X_test, y_test, feature_name)
        
        elif main_choice == '4':
            y_pred = model.predict(X_test_processed)

            # Heatmap FPR
            print("\nHeatmap FPR: Inserisci due feature categoriche [sex / cp / fbs / restecg / exang / slope / ca / thal]")
            feature_row = input("Feature riga: ")
            feature_col = input("Feature colonna: ")

            valid_categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
            if feature_row in valid_categorical and feature_col in valid_categorical:
                df_clean = pd.read_csv(os.path.join(OUTPUT_PATH, cleaned_data_file))
                X_test_original = X_test.copy()
                model_metrics.plot_fpr_heatmap(X_test_original, y_test.values, y_pred, feature_row, feature_col)

            else:
                print("Feature non valide.")

            # Calcolo dell'F1-score per la Regressione Logistica
            f1_logistic_regression = f1_score(y_test, y_pred) 
            # Calcolo dell'F1-score per la baseline 
            baseline_f1_score = 0.0 
            model_metrics.plot_model_comparison(accuracy, f1_logistic_regression, baseline_accuracy, baseline_f1_score)

            # Calcolo della Matrice di confusione
            model_metrics.plot_confusion_matrix(y_test, y_pred)

        elif main_choice == '0':
            print("Uscita dal programma.")
            break
        else:
            print("Scelta non valida. Riprova.")

menu()