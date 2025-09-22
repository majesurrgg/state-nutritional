# contendra el código para entrenar modelos base
# posteriormente el meta-modelo
# src/train_models.py
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from preprocess import load_and_split_data

def train_base_models(X_train, y_train):
    # Entrenar Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Entrenar XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model

if __name__ == '__main__':
    # Ruta del archivo CSV
    file_path = '../data/Niños AREQUIPA.csv'

    # Cargar y dividir los datos
    X_train, X_test, y_train, y_test = load_and_split_data(file_path)

    # Entrenar los modelos
    rf_model, xgb_model = train_base_models(X_train, y_train)
    print("Modelos base entrenados correctamente.")