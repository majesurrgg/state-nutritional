# funciones:
# orquestacion de pipeline completo
# configuracion de parámetros
# ejecución secuencial

import os
import sys
import argparse
import time
from datetime import datetime
from eda import ExploratoryDataAnalysis

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar módulos del proyecto
from preprocess import DataPreprocessor
from train_models import ModelTrainer
from evaluate import ModelEvaluator

class MLPipeline:
    def step_0_eda(self):
        """Paso 0: Análisis Exploratorio de Datos"""
        print("\n" + "="*60)
        print("PASO 0: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("="*60)
    
        start_time = time.time()
    
        try:
            eda = ExploratoryDataAnalysis()
            success = eda.run_complete_eda()
        
            if not success:
                raise Exception("Error en el EDA")
        
            print(f"\n✅ EDA completado exitosamente!")
            print(f"⏱️  Tiempo: {time.time() - start_time:.2f} segundos")
        
            return True
        
        except Exception as e:
            print(f"\n❌ Error en EDA: {e}")
            return False
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.preprocessor = None
        self.trainer = None
        self.evaluator = None
        
        # Crear directorios necesarios
        self.create_directories()
    
    def get_default_config(self):
        """Configuración por defecto del pipeline"""
        return {
            'data_path': 'data/raw/',
            'target_variable': 'anemia',  # opciones: 'anemia', 'desnutricion', 'riesgo'
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'save_processed': True,
            'save_models': True,
            'create_plots': True
        }
    
    def create_directories(self):
        """Crear estructura de directorios necesaria"""
        directories = [
            'data/processed',
            'models/base_models',
            'models/stacking',
            'results/metrics',
            'results/plots',
            'results/reports'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def step_1_preprocess(self):
        """Paso 1: Preprocesamiento de datos"""
        print("\n" + "="*60)
        print("PASO 1: PREPROCESAMIENTO DE DATOS")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Inicializar preprocessor
            self.preprocessor = DataPreprocessor(data_path=self.config['data_path'])
            
            # Procesar datos
            data = self.preprocessor.process_data(
                target=self.config['target_variable'],
                save_processed=self.config['save_processed']
            )
            
            if data is None:
                raise Exception("Error en el preprocesamiento de datos")
            
            X_train, X_val, X_test, y_train, y_val, y_test = data
            
            print(f"\n✅ Preprocesamiento completado exitosamente!")
            print(f"⏱️  Tiempo: {time.time() - start_time:.2f} segundos")
            print(f"📊 Datos finales:")
            print(f"   - Entrenamiento: {X_train.shape}")
            print(f"   - Validación: {X_val.shape}")
            print(f"   - Prueba: {X_test.shape}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error en preprocesamiento: {e}")
            return False
    
    def step_2_train(self):
        """Paso 2: Entrenamiento de modelos"""
        print("\n" + "="*60)
        print("PASO 2: ENTRENAMIENTO DE MODELOS")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Inicializar trainer
            self.trainer = ModelTrainer(random_state=self.config['random_state'])
            
            # Entrenar modelos
            results = self.trainer.train_stacking_ensemble()
            
            if results is None:
                raise Exception("Error en el entrenamiento de modelos")
            
            ensemble, base_results, stacking_results = results
            
            print(f"\n✅ Entrenamiento completado exitosamente!")
            print(f"⏱️  Tiempo: {time.time() - start_time:.2f} segundos")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error en entrenamiento: {e}")
            return False
    
    def step_3_evaluate(self):
        """Paso 3: Evaluación detallada"""
        print("\n" + "="*60)
        print("PASO 3: EVALUACIÓN DETALLADA")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Inicializar evaluador
            self.evaluator = ModelEvaluator()
            
            # Realizar evaluación completa
            results = self.evaluator.comprehensive_evaluation()
            
            if results is None:
                raise Exception("Error en la evaluación")
            
            print(f"\n✅ Evaluación completada exitosamente!")
            print(f"⏱️  Tiempo: {time.time() - start_time:.2f} segundos")
            
            return True, results
            
        except Exception as e:
            print(f"\n❌ Error en evaluación: {e}")
            return False, None
    
    def generate_summary_report(self, evaluation_results=None):
        """Generar reporte resumen del proyecto"""
        print("\n" + "="*60)
        print("GENERANDO REPORTE RESUMEN")
        print("="*60)
        
        try:
            report_lines = []
            
            # Encabezado
            report_lines.extend([
                "="*80,
                "REPORTE FINAL DEL PROYECTO",
                "Aplicación de Algoritmos de Machine Learning para la",
                "Predicción del Estado Nutricional Infantil en Arequipa",
                "="*80,
                f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Variable objetivo: {self.config['target_variable']}",
                ""
            ])
            
            # Configuración utilizada
            report_lines.extend([
                "📋 CONFIGURACIÓN DEL EXPERIMENTO:",
                f"  • Tamaño de prueba: {self.config['test_size']*100:.1f}%",
                f"  • Tamaño de validación: {self.config['val_size']*100:.1f}%",
                f"  • Semilla aleatoria: {self.config['random_state']}",
                f"  • Variable objetivo: {self.config['target_variable']}",
                ""
            ])
            
            # Metodología
            report_lines.extend([
                "🔬 METODOLOGÍA:",
                "  • Fuente de datos: SIEN-HIS (Sistema de Información del Estado Nutricional)",
                "  • Población: Niños de 5-11 años en Arequipa, 2024",
                "  • Técnica: Stacking Ensemble Learning",
                "  • Modelos base: Random Forest, XGBoost, SVM",
                "  • Meta-modelo: Regresión Logística",
                "  • Validación: Validación cruzada estratificada (5 folds)",
                ""
            ])
            
            # Resultados (si están disponibles)
            if evaluation_results:
                metrics = evaluation_results['metrics']
                clinical_metrics = evaluation_results.get('clinical_metrics')
                
                report_lines.extend([
                    "📊 RESULTADOS PRINCIPALES:",
                    f"  • Exactitud (Accuracy): {metrics['accuracy']:.4f}",
                    f"  • Precisión: {metrics['precision']:.4f}",
                    f"  • Sensibilidad (Recall): {metrics['recall']:.4f}",
                    f"  • F1-Score: {metrics['f1_score']:.4f}",
                ])
                
                if 'auc_roc' in metrics:
                    report_lines.append(f"  • AUC-ROC: {metrics['auc_roc']:.4f}")
                
                if clinical_metrics:
                    report_lines.extend([
                        "",
                        "🏥 MÉTRICAS CLÍNICAS:",
                        f"  • Sensibilidad clínica: {clinical_metrics['sensitivity']:.4f}",
                        f"  • Especificidad clínica: {clinical_metrics['specificity']:.4f}",
                        f"  • Valor Predictivo Positivo: {clinical_metrics['ppv']:.4f}",
                        f"  • Valor Predictivo Negativo: {clinical_metrics['npv']:.4f}",
                    ])
                
                report_lines.append("")
            
            # Archivos generados
            report_lines.extend([
                "  ARCHIVOS GENERADOS:",
                "  Datos procesados:",
                "    • data/processed/merged_data.csv - Dataset unificado",
                "    • data/processed/train_data.csv - Datos de entrenamiento",
                "    • data/processed/val_data.csv - Datos de validación", 
                "    • data/processed/test_data.csv - Datos de prueba",
                "",
                "  Modelos entrenados:",
                "    • models/base_models/ - Modelos individuales (RF, XGB, SVM)",
                "    • models/stacking/ - Modelo de stacking ensemble",
                "",
                "  Resultados y evaluación:",
                "    • results/metrics/ - Métricas detalladas",
                "    • results/plots/ - Visualizaciones",
                "    • results/reports/ - Reportes detallados",
                ""
            ])
            
            # Conclusiones y recomendaciones
            report_lines.extend([
                "🎯 CONCLUSIONES:",
                "  • Se implementó exitosamente un modelo de stacking ensemble",
                "  • El modelo combina las fortalezas de tres algoritmos diferentes",
                "  • Los datos del SIEN-HIS proporcionan información valiosa para la predicción",
                "  • La corrección por altitud es crucial en el contexto de Arequipa",
                "",
                "💡 RECOMENDACIONES PARA USO CLÍNICO:",
                "  • Validar el modelo con datos de otras regiones",
                "  • Considerar actualización periódica con nuevos datos",
                "  • Integrar con sistemas de información hospitalaria",
                "  • Capacitar al personal de salud en interpretación de resultados",
                "",
                "📝 PARA EL ARTÍCULO DE INVESTIGACIÓN:",
                "  • Los resultados demuestran la viabilidad del stacking ensemble",
                "  • La metodología es replicable y escalable",
                "  • Los datos reales de Arequipa aportan validez externa",
                "  • Se puede comparar con enfoques tradicionales de diagnóstico",
                ""
            ])
            
            # Limitaciones
            report_lines.extend([
                "⚠️  LIMITACIONES:",
                "  • Datos limitados a un año (2024) y una región (Arequipa)",
                "  • Posible sesgo de selección en establecimientos de salud",
                "  • Requiere validación prospectiva",
                "  • Dependencia de calidad de datos del SIEN-HIS",
                "",
                "="*80
            ])
            
            # Guardar reporte
            report_text = '\n'.join(report_lines)
            
            os.makedirs('results/reports', exist_ok=True)
            with open('results/reports/project_summary.txt', 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print("✅ Reporte resumen generado: results/reports/project_summary.txt")
            return report_text
            
        except Exception as e:
            print(f"❌ Error generando reporte: {e}")
            return None
    
    def run_complete_pipeline(self):
        """Ejecutar pipeline completo"""
        print("🚀 INICIANDO PIPELINE COMPLETO DE MACHINE LEARNING")
        #print("Predicción del Estado Nutricional Infantil - Arequipa")
        #print("="*80)
        
        pipeline_start = time.time()
        
        # Paso 0: EDA (NUEVO)
        if not self.step_0_eda():
            print("\n⚠️  EDA falló, pero continuando...")
        
        # Paso 1: Preprocesamiento
        if not self.step_1_preprocess():
            print("\n💥 Pipeline interrumpido en el preprocesamiento")
            return False
        
        # Paso 2: Entrenamiento
        if not self.step_2_train():
            print("\n💥 Pipeline interrumpido en el entrenamiento")
            return False
        
        # Paso 3: Evaluación
        success, evaluation_results = self.step_3_evaluate()
        if not success:
            print("\n💥 Pipeline interrumpido en la evaluación")
            return False
        
        # Generar reporte final
        self.generate_summary_report(evaluation_results)
        
        # Tiempo total
        total_time = time.time() - pipeline_start
        
        print(f"\n🎉 ¡PIPELINE COMPLETADO EXITOSAMENTE!")
        print(f"⏱️  Tiempo total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"📂 Revisa la carpeta 'results/' para todos los outputs")
        
        return True
    
    def run_single_step(self, step):
        """Ejecutar un paso individual"""
        if step == 'preprocess':
            return self.step_1_preprocess()
        elif step == 'train':
            return self.step_2_train()
        elif step == 'evaluate':
            success, results = self.step_3_evaluate()
            return success
        else:
            print(f"❌ Paso '{step}' no reconocido")
            return False

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Pipeline de Machine Learning para Predicción de Estado Nutricional Infantil'
    )
    
    parser.add_argument(
        '--step', 
        choices=['preprocess', 'train', 'evaluate', 'complete'],
        default='complete',
        help='Paso específico a ejecutar (default: complete)'
    )
    
    parser.add_argument(
        '--target',
        choices=['anemia', 'desnutricion', 'riesgo'],
        default='anemia',
        help='Variable objetivo a predecir (default: anemia)'
    )
    
    parser.add_argument(
        '--data-path',
        default='data/raw/',
        help='Ruta a los datos originales (default: data/raw/)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Semilla para reproducibilidad (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Configuración personalizada
    config = {
        'data_path': args.data_path,
        'target_variable': args.target,
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': args.random_state,
        'save_processed': True,
        'save_models': True,
        'create_plots': True
    }
    
    # Inicializar pipeline
    pipeline = MLPipeline(config)
    
    # Ejecutar según el argumento
    if args.step == 'complete':
        success = pipeline.run_complete_pipeline()
    else:
        success = pipeline.run_single_step(args.step)
    
    if success:
        print(f"\n✅ Proceso '{args.step}' completado exitosamente")
    else:
        print(f"\n❌ Error en el proceso '{args.step}'")
        sys.exit(1)

if __name__ == "__main__":
    main()