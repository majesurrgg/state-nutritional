# funciones:
# entrenamiento de modelos base (random_forest, xgboost, svm)
# implementacion de stacking ensemble
# guardado de modelos

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import os

class StackingEnsemble:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.meta_model = None
        self.is_fitted = False
        
        # Inicializar modelos base (Nivel 0)
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Inicializar modelos base con parámetros optimizados"""
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Importante para obtener probabilidades
                random_state=self.random_state
            )
        }
        
        # Metamodelo (Nivel 1)
        self.meta_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
    
    def _generate_base_predictions(self, X_train, y_train, X_val=None):
        """Generar predicciones de los modelos base usando validación cruzada"""
        print("Generando predicciones de modelos base...")
        
        # Configurar validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Almacenar predicciones de entrenamiento (para entrenar meta-modelo)
        train_meta_features = np.zeros((X_train.shape[0], len(self.base_models)))
        
        # Almacenar predicciones de validación (si se proporciona)
        val_meta_features = None
        if X_val is not None:
            val_meta_features = np.zeros((X_val.shape[0], len(self.base_models)))
        
        # Generar predicciones para cada modelo base
        for i, (model_name, model) in enumerate(self.base_models.items()):
            print(f"  Procesando {model_name}...")
            
            # Predicciones de validación cruzada para entrenamiento
            train_pred = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')
            train_meta_features[:, i] = train_pred[:, 1]  # Probabilidad de clase positiva
            
            # Entrenar modelo en todos los datos de entrenamiento
            model.fit(X_train, y_train)
            
            # Predicciones para validación (si se proporciona)
            if X_val is not None:
                val_pred = model.predict_proba(X_val)
                val_meta_features[:, i] = val_pred[:, 1]
        
        return train_meta_features, val_meta_features
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrenar el modelo de stacking ensemble"""
        print("Iniciando entrenamiento del Stacking Ensemble...")
        
        # Paso 1: Generar predicciones de modelos base
        train_meta_features, val_meta_features = self._generate_base_predictions(
            X_train, y_train, X_val
        )
        
        # Paso 2: Entrenar meta-modelo
        print("Entrenando meta-modelo...")
        self.meta_model.fit(train_meta_features, y_train)
        
        # Evaluación en validación (si se proporciona)
        if X_val is not None and y_val is not None:
            val_predictions = self.meta_model.predict(val_meta_features)
            val_proba = self.meta_model.predict_proba(val_meta_features)[:, 1]
            
            print(f"\nRendimiento en validación:")
            print(f"Accuracy: {accuracy_score(y_val, val_predictions):.4f}")
            if len(np.unique(y_val)) == 2:  # Solo para clasificación binaria
                print(f"AUC-ROC: {roc_auc_score(y_val, val_proba):.4f}")
        
        self.is_fitted = True
        print("¡Entrenamiento completado!")
    
    def predict(self, X):
        """Realizar predicciones con el ensemble"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Generar predicciones de modelos base
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (model_name, model) in enumerate(self.base_models.items()):
            pred_proba = model.predict_proba(X)
            meta_features[:, i] = pred_proba[:, 1]
        
        # Predicción final con meta-modelo
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """Obtener probabilidades de predicción"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Generar predicciones de modelos base
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (model_name, model) in enumerate(self.base_models.items()):
            pred_proba = model.predict_proba(X)
            meta_features[:, i] = pred_proba[:, 1]
        
        # Probabilidades finales con meta-modelo
        return self.meta_model.predict_proba(meta_features)
    
    def evaluate_base_models(self, X_test, y_test):
        """Evaluar modelos base individualmente"""
        print("\n" + "="*50)
        print("EVALUACIÓN DE MODELOS BASE")
        print("="*50)
        
        results = {}
        
        for model_name, model in self.base_models.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print("-" * 30)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            
            # AUC solo para clasificación binaria
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_proba)
                results[model_name]['auc'] = auc
                print(f"AUC-ROC: {auc:.4f}")
            
            # Reporte de clasificación
            print("\nReporte de clasificación:")
            print(classification_report(y_test, y_pred, zero_division=0))
        
        return results
    
    def evaluate_stacking(self, X_test, y_test):
        """Evaluar el modelo de stacking"""
        print("\n" + "="*50)
        print("EVALUACIÓN DEL STACKING ENSEMBLE")
        print("="*50)
        
        # Predicciones del ensemble
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # AUC solo para clasificación binaria
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba)
            results['auc'] = auc
            print(f"AUC-ROC: {auc:.4f}")
        
        # Reporte de clasificación
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Matriz de confusión
        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        
        return results
    
    def save_models(self, base_path='models/'):
        """Guardar todos los modelos"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Crear directorios
        os.makedirs(f'{base_path}/base_models', exist_ok=True)
        os.makedirs(f'{base_path}/stacking', exist_ok=True)
        
        # Guardar modelos base
        for model_name, model in self.base_models.items():
            joblib.dump(model, f'{base_path}/base_models/{model_name}.pkl')
        
        # Guardar meta-modelo
        joblib.dump(self.meta_model, f'{base_path}/stacking/meta_model.pkl')
        
        # Guardar el ensemble completo
        joblib.dump(self, f'{base_path}/stacking/stacking_ensemble.pkl')
        
        print(f"Modelos guardados en {base_path}")
    
    @classmethod
    def load_models(cls, base_path='models/'):
        """Cargar modelo completo"""
        ensemble = joblib.load(f'{base_path}/stacking/stacking_ensemble.pkl')
        print("Modelos cargados exitosamente")
        return ensemble

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.stacking_ensemble = None
    
    def load_processed_data(self):
        """Cargar datos procesados"""
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            val_data = pd.read_csv('data/processed/val_data.csv')
            test_data = pd.read_csv('data/processed/test_data.csv')
            
            # Separar características y target (asumiendo que target es la última columna)
            X_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]
            
            X_val = val_data.iloc[:, :-1]
            y_val = val_data.iloc[:, -1]
            
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]
            
            print("Datos cargados exitosamente:")
            print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            print(f"Distribución target - Train: {y_train.value_counts().to_dict()}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return None
    
    def train_stacking_ensemble(self):
        """Entrenar el modelo de stacking ensemble completo"""
        # Cargar datos
        data = self.load_processed_data()
        if data is None:
            return None
        
        X_train, X_val, X_test, y_train, y_val, y_test = data
        
        # Inicializar y entrenar ensemble
        self.stacking_ensemble = StackingEnsemble(random_state=self.random_state)
        self.stacking_ensemble.fit(X_train, y_train, X_val, y_val)
        
        # Evaluaciones
        print("\n" + "="*80)
        print("EVALUACIÓN COMPLETA DEL MODELO")
        print("="*80)
        
        # Evaluar modelos base
        base_results = self.stacking_ensemble.evaluate_base_models(X_test, y_test)
        
        # Evaluar stacking ensemble
        stacking_results = self.stacking_ensemble.evaluate_stacking(X_test, y_test)
        
        # Comparación de resultados
        self._compare_results(base_results, stacking_results)
        
        # Guardar modelos
        self.stacking_ensemble.save_models()
        
        return self.stacking_ensemble, base_results, stacking_results
    
    def _compare_results(self, base_results, stacking_results):
        """Comparar resultados de modelos base vs stacking"""
        print("\n" + "="*50)
        print("COMPARACIÓN DE RESULTADOS")
        print("="*50)
        
        # Crear tabla de comparación
        comparison_data = []
        
        for model_name, results in base_results.items():
            row = {
                'Modelo': model_name.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy']:.4f}"
            }
            if 'auc' in results:
                row['AUC-ROC'] = f"{results['auc']:.4f}"
            comparison_data.append(row)
        
        # Agregar stacking
        stacking_row = {
            'Modelo': 'Stacking Ensemble',
            'Accuracy': f"{stacking_results['accuracy']:.4f}"
        }
        if 'auc' in stacking_results:
            stacking_row['AUC-ROC'] = f"{stacking_results['auc']:.4f}"
        comparison_data.append(stacking_row)
        
        # Mostrar tabla
        df_comparison = pd.DataFrame(comparison_data)
        print("\nTabla de Comparación:")
        print(df_comparison.to_string(index=False))
        
        # Determinar el mejor modelo
        best_accuracy = max([results['accuracy'] for results in base_results.values()] + [stacking_results['accuracy']])
        
        if stacking_results['accuracy'] == best_accuracy:
            print(f"\n🎉 ¡El Stacking Ensemble obtuvo el mejor resultado!")
            improvement = stacking_results['accuracy'] - max([results['accuracy'] for results in base_results.values()])
            print(f"Mejora en accuracy: +{improvement:.4f}")
        else:
            print(f"\nEl mejor modelo individual fue mejor que el stacking.")
        
        # Guardar resultados
        os.makedirs('results/metrics', exist_ok=True)
        df_comparison.to_csv('results/metrics/model_comparison.csv', index=False)

def main():
    """Función principal para entrenar los modelos"""
    print("INICIANDO ENTRENAMIENTO DE MODELOS")
    print("="*50)
    
    # Inicializar trainer
    trainer = ModelTrainer(random_state=42)
    
    # Entrenar modelos
    results = trainer.train_stacking_ensemble()
    
    if results:
        ensemble, base_results, stacking_results = results
        print("\n✅ Entrenamiento completado exitosamente!")
        print("📁 Modelos guardados en la carpeta 'models/'")
        print("📊 Resultados guardados en la carpeta 'results/'")
    else:
        print("\n❌ Error en el entrenamiento")

if __name__ == "__main__":
    main()