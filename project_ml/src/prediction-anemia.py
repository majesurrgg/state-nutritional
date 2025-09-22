#1er paso escoger 2 modelos ml (xgboost y randomForest)
# hay 2: entrenamiento para predicción de anemia
#        entreamiento para predecir estado nutricional
# 1. Cargar y Dividir Datos: Carga el archivo Niños AREQUIPA.csv de la carpeta Hb. 
# Identifica tu variable objetivo (la columna que indica si hay anemia o no) y tus 
# variables de entrada (todas las demás columnas que usarás para predecir). Luego, 
# divide tus datos en conjuntos de entrenamiento y prueba.

import pandas as pd
from sklearn.model_selection import train_test_split
# Cargar el dataset de anemia
df_anemia = pd.read_csv('ruta/a/Hb/Niños AREQUIPA.csv')

# Identificar variables de entrada (X) y objetivo (y)
# Reemplaza 'nombre_variable_anemia' y 'columna_excluida' con los nombres reales
X_anemia = df_anemia.drop(['nombre_variable_anemia'], axis=1) 
y_anemia = df_anemia['nombre_variable_anemia']

# Dividir los datos en entrenamiento y prueba
X_train_anemia, X_test_anemia, y_train_anemia, y_test_anemia = train_test_split(X_anemia, y_anemia, test_size=0.2, random_state=42)


# 2.Entrenar los Modelos Base: Ahora, entrena cada modelo por separado usando los 
# datos de entrenamiento.
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Inicializar los modelos
rf_anemia = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_anemia = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Entrenar Random Forest
rf_anemia.fit(X_train_anemia, y_train_anemia)

# Entrenar XGBoost
xgb_anemia.fit(X_train_anemia, y_train_anemia)