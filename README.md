 Progetto ML: Predizione Malattie Cardiache con il Dataset di Cleveland

🎯 Obiettivo
Il progetto ha l’obiettivo di sviluppare un sistema di classificazione binaria in grado di prevedere la presenza o assenza di malattie cardiache a partire da un set di esami clinici, utilizzando il dataset di Cleveland.

📁 Struttura dei file
Modulo  - Descrizione 
main.py - Script principale: carica i dati, addestra il modello, avvia menu interattivo
preprocessing.py - Funzioni per la pulizia e trasformazione dei dati
data_visualization.py -	Funzioni per analisi esplorativa (grafici)
prediction_cli.py -	Permette l’inserimento manuale di dati da terminale per test pratici
robustness_test.py - Test di robustezza adversariale (AML) su input perturbati
Dataset/ - Contiene il file originale del dataset (processed.cleveland.data)
Process_data/ -	Contiene i file generati dopo preprocessing e split

⚙️ Come eseguire il progetto
1. Requisiti
Installa i pacchetti richiesti: 
    pip install pandas scikit-learn matplotlib seaborn
2. Avvia il progetto
Esegui da terminale:
    python main.py


📊 Menu interattivo

Durante l’esecuzione di main.py, apparirà un menu che consente di:
    1.  Visualizzare grafici EDA
        (distribuzioni, correlazioni, boxplot…)
    2.  Inserire dati manuali da terminale
        per ottenere una previsione "in tempo reale"
    3.  Test di robustezza del modello
        con perturbazioni su feature numeriche (Adversarial Machine Learning)
    4.  Uscire


🧠 Modello utilizzato
    - Regressione Logistica
    - Feature numeriche standardizzate, categoriche codificate One-Hot
    - Divisione training/test stratificata
    - Valutazione tramite Accuracy, Report di Classificazione, Confusion Matrix
    - Confronto con Baseline

🔐 Considerazioni su sicurezza e affidabilità
    Il progetto include un modulo di testing adversariale che mostra come piccole modifiche nei dati in input possano cambiare l’output del modello.
    Serve a valutare la robustezza del sistema predittivo, fondamentale in ambito medico.

📚 Dataset
Il dataset utilizzato è il Cleveland Heart Disease dataset (UCI repository), contenente 14 variabili tra cui:
    - Età, sesso, colesterolo, glicemia, tipo di dolore toracico, ECG, etc.
    - L’output target è binarizzato (0 = assenza, 1 = presenza di malattia)

📌 Autore
    Progetto sviluppato da Tizio
    Università: [NOME CORSO]
    Anno Accademico: 2024/2025
    Consegna: 10 Luglio 2025