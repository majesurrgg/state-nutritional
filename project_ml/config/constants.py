# Constantes y configuraciones del proyecto
# Define valores fijos utilizados en todo el sistema

# ==================== INFORMACIÓN DEL PROYECTO ====================
PROJECT_NAME = "Predicción de Estado Nutricional Infantil"
PROJECT_VERSION = "2.0"
PROJECT_DESCRIPTION = "Sistema de ML para detección temprana de anemia y desnutrición en menores de 5 años"
AUTHOR = "Tu Nombre"
INSTITUTION = "Tu Universidad"

# ==================== CONFIGURACIÓN DE DATOS ====================
# Años de datos a procesar
DATA_YEARS = ['2022', '2023', '2024']

# Rango de edad objetivo (meses)
MIN_AGE_MONTHS = 0
MAX_AGE_MONTHS = 59  # Menores de 5 años

# ==================== PUNTOS DE CORTE CLÍNICOS ====================
# Anemia (hemoglobina en g/dL)
HB_CUTOFF_ANEMIA = {
    '0-5': 11.0,    # 0-5 meses
    '6-59': 11.0,   # 6-59 meses
}

# Desnutrición (Z-scores)
Z_SCORE_SEVERE = -3
Z_SCORE_MODERATE = -2
Z_SCORE_NORMAL = -1

# IMC (kg/m²)
IMC_BAJO = 13.0
IMC_NORMAL_MIN = 13.0
IMC_NORMAL_MAX = 17.0
IMC_SOBREPESO = 17.0

# ==================== RANGOS DE VALORES VÁLIDOS ====================
# Para filtrado de datos
VALID_RANGES = {
    'peso': (2.0, 25.0),           # kg
    'talla': (45.0, 120.0),        # cm
    'hemoglobina': (5.0, 20.0),    # g/dL
    'edad_meses': (0, 59),         # meses
}

# ==================== CATEGORÍAS ====================
# Grupos de edad
AGE_GROUPS = {
    '0-5m': (0, 5),
    '6-11m': (6, 11),
    '12-23m': (12, 23),
    '24-35m': (24, 35),
    '36-59m': (36, 59)
}

AGE_GROUP_LABELS = ['0-5m', '6-11m', '12-23m', '24-35m', '36-59m']

# Categorías de altitud (metros sobre el nivel del mar)
ALTITUDE_CATEGORIES = {
    'Baja': (0, 2500),
    'Media': (2500, 3500),
    'Alta': (3500, 5000)
}

# ==================== VARIABLES DEL MODELO ====================
# Variables numéricas básicas
NUMERIC_FEATURES = [
    'EdadMeses_hb',
    'Hemoglobina',
    'Hbc',
    'Peso',
    'Talla',
    'ZPE',
    'ZTE',
    'ZIMCE',
    'AlturaREN_hb',
    'IMC',
    'Edad_años'
]

# Variables categóricas
CATEGORICAL_FEATURES = [
    'Sexo_hb',
    'Grupo_edad',
    'Categoria_altitud'
]

# Programas sociales
PROGRAMAS_SOCIALES = [
    'Juntos_hb',
    'SIS_hb',
    'Pin_hb',
    'Qaliwarma_hb'
]

# Servicios de salud
SERVICIOS_SALUD = [
    'Cred_hb',
    'Suplementacion_hb',
    'Consejeria_hb'
]

# Variables objetivo
TARGET_VARIABLES = {
    'anemia': 'Anemia',
    'desnutricion': 'Estado_nutricional_alterado',
    'riesgo': 'Riesgo_nutricional'
}

# ==================== CONFIGURACIÓN DE MODELOS ====================
# Hiperparámetros de Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'n_jobs': -1
}

# Hiperparámetros de XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss'
}

# Hiperparámetros de SVM
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': True
}

# Configuración de validación cruzada
CV_FOLDS = 5
CV_SHUFFLE = True

# ==================== DIVISIÓN DE DATOS ====================
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

# ==================== RUTAS DE ARCHIVOS ====================
# Estructura de directorios
DIRS = {
    'data_raw': 'data/raw/',
    'data_processed': 'data/processed/',
    'models_base': 'models/base_models/',
    'models_stacking': 'models/stacking/',
    'results_metrics': 'results/metrics/',
    'results_plots': 'results/plots/',
    'results_reports': 'results/reports/',
    'results_eda': 'results/eda/'
}

# Nombres de archivos
FILES = {
    'merged_data': 'data/processed/merged_data.csv',
    'train_data': 'data/processed/train_data.csv',
    'val_data': 'data/processed/val_data.csv',
    'test_data': 'data/processed/test_data.csv',
    'model_rf': 'models/base_models/random_forest.pkl',
    'model_xgb': 'models/base_models/xgboost.pkl',
    'model_svm': 'models/base_models/svm.pkl',
    'model_ensemble': 'models/stacking/stacking_ensemble.pkl',
    'eda_report': 'results/reports/eda_report.txt',
    'eval_report': 'results/reports/evaluation_report.txt',
    'project_summary': 'results/reports/project_summary.txt'
}

# ==================== CONFIGURACIÓN DE VISUALIZACIÓN ====================
# Estilo de gráficos
PLOT_STYLE = 'seaborn-v0_8'
PLOT_DPI = 300
FIGURE_SIZE_DEFAULT = (15, 10)
FIGURE_SIZE_LARGE = (20, 15)

# Paleta de colores
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ffbb00',
    'danger': '#d62728',
    'info': '#17a2b8',
    'anemia': '#ff6b6b',
    'normal': '#51cf66',
    'desnutricion': '#ffa94d'
}

# ==================== MENSAJES Y TEXTOS ====================
MESSAGES = {
    'loading_data': "Cargando datos...",
    'processing': "Procesando...",
    'training': "Entrenando modelos...",
    'evaluating': "Evaluando modelo...",
    'success': "✅ Completado exitosamente",
    'error': "❌ Error en el proceso",
    'warning': "⚠️ Advertencia"
}

# Interpretaciones clínicas
INTERPRETATIONS = {
    'sensitivity': {
        'excellent': "Excelente - Muy pocos casos pasarán desapercibidos",
        'good': "Buena - Detecta la mayoría de casos",
        'acceptable': "Aceptable - Algunos casos no serán detectados",
        'low': "Baja - Muchos casos no serán detectados"
    },
    'specificity': {
        'excellent': "Excelente - Muy pocos falsos positivos",
        'good': "Buena - Pocos diagnósticos incorrectos",
        'acceptable': "Aceptable - Algunos diagnósticos incorrectos",
        'low': "Baja - Muchos diagnósticos incorrectos"
    }
}

# Umbrales de interpretación
INTERPRETATION_THRESHOLDS = {
    'excellent': 0.95,
    'good': 0.85,
    'acceptable': 0.70
}

# ==================== CONFIGURACIÓN DE REPORTES ====================
REPORT_ENCODING = 'utf-8'
REPORT_LINE_WIDTH = 80

# ==================== AJUSTE POR ALTITUD ====================
# Factores de corrección de hemoglobina por altitud (g/dL)
ALTITUDE_CORRECTION = {
    (0, 1000): 0.0,
    (1000, 1500): -0.2,
    (1500, 2000): -0.5,
    (2000, 2500): -0.8,
    (2500, 3000): -1.3,
    (3000, 3500): -1.9,
    (3500, 4000): -2.7,
    (4000, 4500): -3.5
}

# ==================== METADATA ====================
METADATA = {
    'dataset_source': 'SIEN-HIS (Sistema de Información del Estado Nutricional)',
    'region': 'Arequipa, Perú',
    'years': '2022-2024',
    'target_population': 'Menores de 5 años (0-59 meses)',
    'main_outcomes': ['Anemia', 'Desnutrición'],
    'algorithms': ['Random Forest', 'XGBoost', 'SVM', 'Stacking Ensemble']
}

# ==================== FUNCIONES AUXILIARES ====================
def get_age_group(edad_meses):
    """Obtener grupo de edad según meses"""
    for label, (min_age, max_age) in AGE_GROUPS.items():
        if min_age <= edad_meses <= max_age:
            return label
    return 'Unknown'

def get_altitude_category(altitude):
    """Obtener categoría de altitud"""
    for category, (min_alt, max_alt) in ALTITUDE_CATEGORIES.items():
        if min_alt <= altitude < max_alt:
            return category
    return 'Alta'  # Por defecto si está muy alto

def get_hb_cutoff(edad_meses):
    """Obtener punto de corte de hemoglobina según edad"""
    if edad_meses <= 5:
        return HB_CUTOFF_ANEMIA['0-5']
    else:
        return HB_CUTOFF_ANEMIA['6-59']

def get_altitude_correction(altitude):
    """Obtener corrección de hemoglobina por altitud"""
    for (min_alt, max_alt), correction in ALTITUDE_CORRECTION.items():
        if min_alt <= altitude < max_alt:
            return correction
    return -3.5  # Máxima corrección para altitudes muy altas

# ==================== VALIDACIÓN ====================
def validate_age(edad_meses):
    """Validar que la edad está en el rango permitido"""
    return MIN_AGE_MONTHS <= edad_meses <= MAX_AGE_MONTHS

def validate_weight(peso):
    """Validar peso"""
    return VALID_RANGES['peso'][0] <= peso <= VALID_RANGES['peso'][1]

def validate_height(talla):
    """Validar talla"""
    return VALID_RANGES['talla'][0] <= talla <= VALID_RANGES['talla'][1]

def validate_hemoglobin(hemoglobina):
    """Validar hemoglobina"""
    return VALID_RANGES['hemoglobina'][0] <= hemoglobina <= VALID_RANGES['hemoglobina'][1]