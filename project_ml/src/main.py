# script para coordinar todo el proceso
# src/main.py
from preprocess import load_and_split_data
from train_models import train_base_models
# Importa aquí las funciones de evaluación y stacking cuando las crees

def run_anemia_prediction():
    print("Iniciando la predicción de anemia...")
    file_path = '../data/Niños AREQUIPA.csv'

    # Paso 1: Cargar y preparar datos
    X_train, X_test, y_train, y_test = load_and_split_data(file_path)

    # Paso 2: Entrenar modelos base
    rf_model, xgb_model = train_base_models(X_train, y_train)

    # Aquí es donde continuarías con los siguientes pasos
    # 3. Generar predicciones de los modelos base
    # 4. Elegir y entrenar el meta-modelo
    # 5. Evaluar el modelo de stacking

if __name__ == '__main__':
    run_anemia_prediction()