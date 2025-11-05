README_MD = """
# 🏥 Predicción Automática de Anemia Infantil en Arequipa, Perú

## 📖 Descripción

Proyecto de Machine Learning para la predicción automática de anemia infantil en niños de 5-11 años en Arequipa, Perú, utilizando datos del Sistema de Información del Estado Nutricional (SIEN-HIS) de los años 2022, 2023 y 2024.

**Método**: Stacking Ensemble combinando Random Forest, XGBoost y SVM.

## 🎯 Objetivos

1. Desarrollar un modelo de IA para predecir anemia infantil
2. Lograr alta sensibilidad (>90%) para no perder casos
3. Superar métodos tradicionales de screening
4. Identificar factores de riesgo más importantes
5. Proporcionar herramienta útil para personal de salud

## 📊 Datos

- **Fuente**: SIEN-HIS (Sistema de Información del Estado Nutricional)
- **Periodo**: 2022-2024 (3 años)
- **Población**: Niños de 5-11 años, Arequipa
- **Variables**: 
  - Hemoglobina
  - Medidas antropométricas (peso, talla, IMC)
  - Datos demográficos (edad, sexo)
  - Factores geográficos (altitud)
  - Programas sociales (Juntos, SIS, Qali Warma, PIN)
  - Servicios de salud (CRED, suplementación, consejería)

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/anemia-prediction-arequipa.git
cd anemia-prediction-arequipa

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\\Scripts\\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete
pip install -e .
```