# config/constants.py
CONSTANTS_PY = """
'''
Constantes del proyecto
'''

# Rangos válidos para variables clínicas
VALID_RANGES = {
    'Hemoglobina': (5, 20),      # g/dL
    'Peso': (10, 80),             # kg
    'Talla': (80, 160),           # cm
    'EdadMeses': (60, 132),       # 5-11 años
    'AlturaREN': (0, 5000),       # msnm
    'IMC': (10, 30)               # kg/m²
}

# Puntos de corte para diagnóstico (OMS)
ANEMIA_CUTOFFS = {
    'sin_anemia': 11.0,           # Hb >= 11.0 g/dL
    'leve': (10.0, 10.9),         # 10.0 <= Hb < 11.0
    'moderada': (7.0, 9.9),       # 7.0 <= Hb < 10.0
    'severa': 7.0                 # Hb < 7.0
}

# Corrección de hemoglobina por altitud (MINSA)
ALTITUDE_CORRECTION = {
    (0, 1000): 0.0,
    (1000, 2000): -0.2,
    (2000, 2500): -0.5,
    (2500, 3000): -0.8,
    (3000, 3500): -1.3,
    (3500, 4000): -1.9,
    (4000, 4500): -2.7
}

# Categorías de altitud
ALTITUDE_CATEGORIES = {
    'baja': (0, 2500),
    'media': (2500, 3500),
    'alta': (3500, 5000)
}

# Grupos de edad
AGE_GROUPS = {
    '5-6': (60, 83),    # meses
    '7-8': (84, 107),
    '9-11': (108, 132)
}

# Colores para gráficos
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'anemia': '#C73E1D',
    'no_anemia': '#06A77D'
}

# Nombres de variables para el paper
VARIABLE_NAMES_ES = {
    'Hemoglobina': 'Hemoglobina (g/dL)',
    'Peso': 'Peso (kg)',
    'Talla': 'Talla (cm)',
    'IMC': 'IMC (kg/m²)',
    'EdadMeses': 'Edad (meses)',
    'Sexo': 'Sexo',
    'AlturaREN': 'Altitud (msnm)'
}

# Años del estudio
STUDY_YEARS = [2022, 2023, 2024]

# Semilla para reproducibilidad
RANDOM_STATE = 42
"""