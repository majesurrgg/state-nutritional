import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset de estado nutricional
df_nutricion = pd.read_csv('ruta/a/Pt/Niños AREQUIPA.csv')

# Identificar variables de entrada (X) y objetivo (y)
# Reemplaza 'nombre_variable_nutricion' y 'columna_excluida' con los nombres reales
X_nutricion = df_nutricion.drop(['nombre_variable_nutricion'], axis=1) 
y_nutricion = df_nutricion['nombre_variable_nutricion']

# Dividir los datos en entrenamiento y prueba
X_train_nutricion, X_test_nutricion, y_train_nutricion, y_test_nutricion = train_test_split(X_nutricion, y_nutricion, test_size=0.2, random_state=42)

# entrenar modelos base
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Inicializar los modelos
rf_nutricion = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_nutricion = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Entrenar Random Forest
rf_nutricion.fit(X_train_nutricion, y_train_nutricion)

# Entrenar XGBoost
xgb_nutricion.fit(X_train_nutricion, y_train_nutricion)