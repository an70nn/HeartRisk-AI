import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(file_path):
    """
        Carica il dataset e applica la pre-elaborazione dei dati iniziale

        Args:
            file_path (str)  : Il percorso del file del dataset
        Returns:
            pandas.DataFrame : Il DataFrame caricato e parzialmente pre-elaborato
    """

    columns_name = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    # Carica il dataset, convertendo i '?' in NaN
    df = pd.read_csv(file_path, header=None, sep=",", na_values=['?'])
    df.columns = columns_name
    
    # Converte i valori Nan per 'ca' e 'thal' con 0
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    df[['ca', 'thal']] = imputer.fit_transform(df[['ca', 'thal']])

    # Binarizza la colonna target
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    return df

def get_preprocessor(numerical_features, categorical_features):
    """
        Crea e restituisce un ColumnTransformer per la pre-elaborazione dei dati
        Questa fase si occuperà della standardizzazione delle features numeriche
        e della codifica one-hot delle features categoriche

        Args:
            numerical_features (list)   : Lista di nomi delle colonne numeriche
            categorical_features (list) : Lista di nomi delle colonne categoriche
        Returns:
            ColumnTransformer: Un oggetto ColumnTransformer configurato
    """

    # Pipeline per la standardizzazione delle features numeriche
    numerical_transformer = StandardScaler()

    # Pipeline per la codifica one-hot delle features categoriche
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer per applicare le trasformazioni
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)
        ])

    return preprocessor