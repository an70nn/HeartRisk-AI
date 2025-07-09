import pandas as pd

def test_user_input(preprocessor, model):
    """
    Permette l'inserimento interattivo dei dati di un paziente e restituisce la predizione del modello.

    Args:
        preprocessor (ColumnTransformer) : Il preprocessore usato per trasformare i dati.
        model (sklearn-like estimator)   : Il modello addestrato che supporta predict e predict_proba.
    Returns:
        Stampa il risultato.
    """


    print("\nINSERIMENTO DATI PAZIENTE")

    try:
        # Input numerici
        age      = float(input("Età: "))
        trestbps = float(input("Pressione a riposo (mm Hg): "))
        chol     = float(input("Colesterolo (mg/dl): "))
        thalach  = float(input("Frequenza cardiaca max: "))
        oldpeak  = float(input("Oldpeak (depressione ST): "))

        # Input categorici (numeri interi)
        sex      = int(input("Sesso (0 = Donna, 1 = Uomo): "))
        cp       = int(input("Tipo di dolore al petto (0–3): "))
        fbs      = int(input("Glicemia a digiuno > 120 mg/dl (0 = No, 1 = Sì): "))
        restecg  = int(input("Risultato ECG a riposo (0–2): "))
        exang    = int(input("Angina da sforzo (0 = No, 1 = Sì): "))
        slope    = int(input("Pendenza tratto ST (0–2): "))
        ca       = int(input("Numero di vasi colorati (0–3): "))
        thal     = int(input("Tipo di thal (1 = normale, 2 = difetto fisso, 3 = reversibile): "))

        # Costruzione input come array 2D
        input_dict = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach,
            'oldpeak': oldpeak,
            'fbs': fbs,
            'restecg': restecg,
            'exang': exang,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        input_data = pd.DataFrame([input_dict])

        # Forza i tipi delle colonne categoriche a stringa per evitare conflitti di tipo con SimpleImputer
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        input_data[categorical_cols] = input_data[categorical_cols].astype(str)


        # Applica lo stesso preprocessing
        input_processed = preprocessor.transform(input_data)

        # Predizione
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][prediction]

        print("\nRISULTATO DELLA PREDIZIONE:")
        if prediction == 1:
            print(f"Probabile presenza di malattia cardiaca ({probability:.2%} di probabilità).")
        else:
            print(f"Nessun segnale di malattia cardiaca rilevato ({probability:.2%} di probabilità).")

    except Exception as e:
        print(f"\nErrore nell'inserimento dei dati: {e}")
