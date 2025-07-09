Sistema predittivo per la diagnosi precoce delle malattie cardiache 

Progetto realizzato per il corso di MMSA 2024/2025 – Università degli Studi di Bari Aldo Moro

# Introduzione
HeartRisk AI è un progetto di machine learning che mira a predire la presenza di malattie cardiache attraverso tecniche di apprendimento supervisionato, spiegabilità (XAI) e test di robustezza.
Il modello si basa sul Cleveland Heart Disease dataset della UCI Machine Learning Repository, scelto per la sua rilevanza clinica, struttura consolidata e ampia diffusione in ambito accademico.

# 🎯 Obiettivo
Predizione della malattia cardiaca
    Utilizzare un classificatore per stimare la presenza o assenza di patologie sulla base di parametri clinici e demografici.
    Spiegabilità delle predizioni
    Applicare tecniche di Explainable AI (XAI) per rilevare possibili bias nei sottogruppi (es. sesso, tipo di dolore toracico).
    Robustezza e stabilità
    Valutare la stabilità delle predizioni tramite perturbazioni controllate delle feature (Adversarial ML).
    Supporto decisionale
    Offrire uno strumento semplice ed efficace per screening preliminari o supporto clinico a basso costo.

|  |  |
|----------|--------|
| **Nome** | Cleveland Heart Disease |
| **Fonte** | UCI ML Repository - Dataset #45 |
| **Record** | 303 (con 14 feature) |
| **Target** | Presenza (1) o assenza (0) di malattia cardiaca |

# Struttura del Progetto
<pre> 
HeartRisk-AI/
│
├── dataset/
│   └── processed.cleveland.data
│
├── output/
│   ├── X_train_processed.csv
│   ├── y_train.csv
│   ├── model_metrics.txt
│   └── ...
│
├── report/
│   ├── Analisi/
│   ├── confusion_matrix_logistic_regression.png
│   ├── heatmap_correlation_features.png
│   ├── f1_score.png
│   └── Report.pdf
│
├── src/
│   ├── data_visualization.py
│   ├── preprocessing.py
│   ├── model_metrics.py
│   ├── robustness_test.py
│   └── prediction_cli.py
│
├── main.py
├── requirements.txt
└── README.md
 </pre>

# Installazione
1. Clona il repository
`git clone https://github.com/an70nn/HeartRisk-AI.git`
2. Sposta nella directory del progetto
`cd HeartRisk-AI`
3. Crea ed attiva l’ambiente virtuale
`python -m venv venv`
(Per Windows  venv\Scripts\activate)
(Per Linux/macOS  source venv/bin/activate)
4. Installa le dipendenze
`pip install -r requirements.txt`
5. Esegui il progetto
`python .\src\main.py`

# 📌 Autore
|  |  |
|----------|--------|
|Progetto sviluppato da | an70nn |
|Università degli Studi di Bari | Corso MMSA |
|Anno Accademico | 2024/2025 |
