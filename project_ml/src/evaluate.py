# funciones:
# carga el modelo final y los datos de prueba
# realiza predicciones y calcula métricas detalladas
# evaluacion de modelos individuales y stacking
# calculo de métricas (precisión, recall, F1, AUC)
# generación de reportes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns      

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

from sklearn.calibration import calibration_curve
import joblib
import os

class ModelEvaluator:
    # define la ruta del modelo a evaluar y variables para almacenar resultados
    def __init__(self, model_path='models/stacking/stacking_ensemble.pkl'):
        self.model_path = model_path
        self.model = None
        self.results = {}
    ###########################################################################    
     
    # carga el modelo entrenado desde disco       
    def load_model(self):
        """Cargar modelo entrenado"""
        try:
            self.model = joblib.load(self.model_path)
            print("Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    # carga de modelo y datos de prueba desde un archivo CSV
    def load_test_data(self):
        """Cargar datos de prueba"""
        try:
            test_data = pd.read_csv('data/processed/test_data.csv')
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]
            
            print(f"Datos de prueba cargados: {X_test.shape}")
            return X_test, y_test
            
        except Exception as e:
            print(f"Error cargando datos de prueba: {e}")
            return None, None
    
    
    # calcula la especificidad como accuracy, precision, recall, F1, AUC, matriz de confusión
    def calculate_detailed_metrics(self, y_true, y_pred, y_proba):
        """Calcular métricas detalladas"""
        metrics = {}
        
        # metricas básicas
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Para clasificación binaria
        if len(np.unique(y_true)) == 2:
            metrics['precision_binary'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall_binary'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1_binary'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['specificity'] = self._calculate_specificity(y_true, y_pred)
            
            # Métricas con probabilidades
            if y_proba is not None:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
                metrics['avg_precision'] = average_precision_score(y_true, y_proba)
        
        # Matriz de confusión
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    # calcular la especificidad (verdaderos negativos)
    def _calculate_specificity(self, y_true, y_pred):
        """Calcular especificidad (tasa de verdaderos negativos)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # calcula métricas clínicas como sensibilidad, especificidad, Valores predictivos, VPN, razones de verosimilitud y las interpreta
    def evaluate_clinical_relevance(self, y_true, y_pred, y_proba):
        """Evaluación desde perspectiva clínica"""
        clinical_metrics = {}
        
        # Calcular métricas clínicas básicas
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        clinical_metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        clinical_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        clinical_metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Valor predictivo positivo
        clinical_metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Valor predictivo negativo
        
        # Razones de verosimilitud
        clinical_metrics['lr_positive'] = clinical_metrics['sensitivity'] / (1 - clinical_metrics['specificity']) if clinical_metrics['specificity'] != 1 else float('inf')
        clinical_metrics['lr_negative'] = (1 - clinical_metrics['sensitivity']) / clinical_metrics['specificity'] if clinical_metrics['specificity'] != 0 else float('inf')
        
        # Interpretación clínica
        clinical_metrics['interpretation'] = self._interpret_clinical_metrics(clinical_metrics)
        
        return clinical_metrics
    
    # genera interpretaciones textuales de las métricas clínicas
    def _interpret_clinical_metrics(self, metrics):
        """Interpretar métricas desde perspectiva clínica"""
        interpretation = {}
        
        # Interpretación de sensibilidad
        if metrics['sensitivity'] >= 0.95:
            interpretation['sensitivity'] = "Excelente - Muy pocos casos de anemia pasarán desapercibidos"
        elif metrics['sensitivity'] >= 0.85:
            interpretation['sensitivity'] = "Buena - Detecta la mayoría de casos de anemia"
        elif metrics['sensitivity'] >= 0.70:
            interpretation['sensitivity'] = "Aceptable - Algunos casos de anemia no serán detectados"
        else:
            interpretation['sensitivity'] = "Baja - Muchos casos de anemia no serán detectados"
        
        # Interpretación de especificidad
        if metrics['specificity'] >= 0.95:
            interpretation['specificity'] = "Excelente - Muy pocos falsos positivos"
        elif metrics['specificity'] >= 0.85:
            interpretation['specificity'] = "Buena - Pocos diagnósticos incorrectos de anemia"
        elif metrics['specificity'] >= 0.70:
            interpretation['specificity'] = "Aceptable - Algunos niños sanos serán diagnosticados con anemia"
        else:
            interpretation['specificity'] = "Baja - Muchos diagnósticos incorrectos de anemia"
        
        return interpretation
    
    # genera y guarda gráficos de evaluación como matriz de confusión, curvas ROC y Precision-Recall, distribución de predicciones y probabilidades, curva de calibración y métricas por clase
    def create_visualizations(self, y_true, y_pred, y_proba):
        """Crear visualizaciones de evaluación"""
        os.makedirs('results/plots', exist_ok=True)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Matriz de confusión
        plt.subplot(3, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        
        # 2. Distribución de predicciones
        plt.subplot(3, 3, 2)
        plt.hist(y_pred, bins=len(np.unique(y_pred)), alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribución de Predicciones')
        plt.xlabel('Clase Predicha')
        plt.ylabel('Frecuencia')
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            # 3. Curva ROC
            plt.subplot(3, 3, 3)
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 4. Curva Precision-Recall
            plt.subplot(3, 3, 4)
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)
            plt.plot(recall, precision, linewidth=2, label=f'PR (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Curva Precision-Recall')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 5. Distribución de probabilidades
            plt.subplot(3, 3, 5)
            plt.hist(y_proba[y_true == 0], bins=30, alpha=0.6, label='Clase 0', color='lightcoral')
            plt.hist(y_proba[y_true == 1], bins=30, alpha=0.6, label='Clase 1', color='lightblue')
            plt.xlabel('Probabilidad Predicha')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Probabilidades por Clase')
            plt.legend()
            
            # 6. Curva de calibración
            plt.subplot(3, 3, 6)
            fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
            plt.plot(mean_pred, fraction_pos, marker='o', linewidth=2, label='Modelo')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfectamente calibrado')
            plt.xlabel('Probabilidad Predicha')
            plt.ylabel('Fracción de Positivos')
            plt.title('Curva de Calibración')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Métricas por clase (si es multiclase)
        if len(np.unique(y_true)) > 2:
            plt.subplot(3, 3, 7)
            report = classification_report(y_true, y_pred, output_dict=True)
            classes = [k for k in report.keys() if k.isdigit() or k in ['0', '1', '2']]
            metrics_names = ['precision', 'recall', 'f1-score']
            
            metrics_data = []
            for metric in metrics_names:
                values = [report[cls][metric] for cls in classes]
                metrics_data.append(values)
            
            x = np.arange(len(classes))
            width = 0.25
            
            for i, metric in enumerate(metrics_names):
                plt.bar(x + i*width, metrics_data[i], width, label=metric)
            
            plt.xlabel('Clases')
            plt.ylabel('Score')
            plt.title('Métricas por Clase')
            plt.xticks(x + width, classes)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/plots/evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizaciones guardadas en results/plots/evaluation_comprehensive.png")
    
    # crea un reporte textual con todas las métricas, interpretaciones y recomendaciones 
    def generate_detailed_report(self, metrics, clinical_metrics):
        """Generar reporte detallado"""
        report = []
        
        report.append("="*80)
        report.append("REPORTE DETALLADO DE EVALUACIÓN DEL MODELO")
        report.append("Predicción de Estado Nutricional Infantil - Arequipa")
        report.append("="*80)
        
        # Métricas generales
        report.append("\n📊 MÉTRICAS GENERALES:")
        report.append(f"  • Accuracy (Exactitud):     {metrics['accuracy']:.4f}")
        report.append(f"  • Precision (Precisión):    {metrics['precision']:.4f}")
        report.append(f"  • Recall (Sensibilidad):    {metrics['recall']:.4f}")
        report.append(f"  • F1-Score:                 {metrics['f1_score']:.4f}")
        
        if 'auc_roc' in metrics:
            report.append(f"  • AUC-ROC:                  {metrics['auc_roc']:.4f}")
        if 'avg_precision' in metrics:
            report.append(f"  • Average Precision:        {metrics['avg_precision']:.4f}")
        
        # Métricas clínicas
        if clinical_metrics:
            report.append("\n🏥 MÉTRICAS CLÍNICAS:")
            report.append(f"  • Sensibilidad (Sensitivity): {clinical_metrics['sensitivity']:.4f}")
            report.append(f"  • Especificidad (Specificity): {clinical_metrics['specificity']:.4f}")
            report.append(f"  • Valor Predictivo Positivo:   {clinical_metrics['ppv']:.4f}")
            report.append(f"  • Valor Predictivo Negativo:   {clinical_metrics['npv']:.4f}")
            
            report.append("\n💡 INTERPRETACIÓN CLÍNICA:")
            for metric, interpretation in clinical_metrics['interpretation'].items():
                report.append(f"  • {metric.capitalize()}: {interpretation}")
        
        # Matriz de confusión
        report.append("\n📋 MATRIZ DE CONFUSIÓN:")
        cm = metrics['confusion_matrix']
        report.append(f"    Predicción →  0    1")
        report.append(f"  Real ↓")
        for i, row in enumerate(cm):
            report.append(f"    {i}           {row[0]:>4} {row[1]:>4}")
        
        # Recomendaciones
        report.append("\n🎯 RECOMENDACIONES PARA USO CLÍNICO:")
        
        if clinical_metrics and clinical_metrics['sensitivity'] >= 0.85:
            report.append("  ✅ El modelo tiene buena sensibilidad para detectar casos de anemia")
        else:
            report.append("  ⚠️  El modelo podría no detectar todos los casos de anemia")
        
        if clinical_metrics and clinical_metrics['specificity'] >= 0.85:
            report.append("  ✅ El modelo tiene baja tasa de falsos positivos")
        else:
            report.append("  ⚠️  El modelo podría generar diagnósticos incorrectos")
        
        report.append("\n📈 SUGERENCIAS DE MEJORA:")
        if metrics['accuracy'] < 0.85:
            report.append("  • Considerar más datos de entrenamiento")
            report.append("  • Revisar ingeniería de características")
            report.append("  • Probar otros algoritmos")
        
        if clinical_metrics and clinical_metrics['sensitivity'] < 0.85:
            report.append("  • Ajustar threshold para aumentar sensibilidad")
            report.append("  • Considerar costo de falsos negativos en contexto clínico")
        
        # Guardar reporte
        os.makedirs('results/reports', exist_ok=True)
        with open('results/reports/evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)
    
    # ejecuta todo el flujo, carga modelos y datos, realiza predicciones, calcula métricas, genera visualiazciones y reportes, guarda resultados en disco
    def comprehensive_evaluation(self):
        """Evaluación completa del modelo"""
        # Cargar modelo y datos
        if not self.load_model():
            return None
        
        X_test, y_test = self.load_test_data()
        if X_test is None or y_test is None:
            return None
        
        print("Iniciando evaluación completa...")
        
        # Realizar predicciones
        y_pred = self.model.predict(X_test)
        y_proba = None
        
        try:
            y_proba_full = self.model.predict_proba(X_test)
            if len(y_proba_full.shape) == 2 and y_proba_full.shape[1] == 2:
                y_proba = y_proba_full[:, 1]  # Probabilidad de clase positiva
        except:
            print("No se pudieron obtener probabilidades")
        
        # Calcular métricas detalladas
        metrics = self.calculate_detailed_metrics(y_test, y_pred, y_proba)
        
        # Métricas clínicas (solo para clasificación binaria)
        clinical_metrics = None
        if len(np.unique(y_test)) == 2:
            clinical_metrics = self.evaluate_clinical_relevance(y_test, y_pred, y_proba)
        
        # Crear visualizaciones
        self.create_visualizations(y_test, y_pred, y_proba)
        
        # Generar reporte
        report = self.generate_detailed_report(metrics, clinical_metrics)
        print(report)
        
        # Guardar métricas en CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('results/metrics/detailed_metrics.csv', index=False)
        
        self.results = {
            'metrics': metrics,
            'clinical_metrics': clinical_metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_test
        }
        
        return self.results

# ejecuta la evaluacion completa del modelo y muestra mensajes de exito o error
def main():
    """Función principal para evaluación"""
    print("INICIANDO EVALUACIÓN DETALLADA DEL MODELO")
    print("="*50)
    
    evaluator = ModelEvaluator()
    results = evaluator.comprehensive_evaluation()
    
    if results:
        print("\n✅ Evaluación completada exitosamente!")
        print("📁 Resultados guardados en la carpeta 'results/'")
    else:
        print("\n❌ Error en la evaluación")

if __name__ == "__main__":
    main()