import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path):
    df = pd.read_csv(file_path, sep=',', encoding='latin1')
    
    # Eliminar columnas que no son útiles para la predicción
    cols_to_drop = ['N', 'REGION', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'COD', 'ESTAB', 'FECHA_REGISTRO', 'APEL', 'NOMB', 'SEXO', 'MESES', 'PESO_G', 'TALLA_CM']
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')

    # Rellenar valores faltantes (un ejemplo sencillo)
    df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

    # Definir las variables objetivo y de entrada
    X = df_clean.drop('ANEMIA', axis=1)
    y = df_clean['ANEMIA']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test