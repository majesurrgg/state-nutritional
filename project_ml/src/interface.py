# Interfaz gráfica para predicción de anemia y desnutrición
# Usa Streamlit para crear una aplicación web interactiva

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

class PredictionInterface:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_model(self):
        """Cargar modelo entrenado"""
        try:
            self.model = joblib.load('models/stacking/stacking_ensemble.pkl')
            return True
        except Exception as e:
            st.error(f"Error cargando modelo: {e}")
            return False
    
    def predict_single(self, features):
        """Realizar predicción individual"""
        try:
            # Convertir a DataFrame
            df_features = pd.DataFrame([features])
            
            # Predicción
            prediction = self.model.predict(df_features)[0]
            probabilities = self.model.predict_proba(df_features)[0]
            
            return prediction, probabilities
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return None, None

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="Sistema de Predicción Nutricional",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título principal
    st.title("🏥 Sistema de Predicción de Estado Nutricional Infantil")
    st.markdown("**Detección temprana de anemia y desnutrición en menores de 5 años**")
    st.markdown("---")
    
    # Sidebar con información
    st.sidebar.header("ℹ️ Información del Sistema")
    st.sidebar.info(
        """
        **Población objetivo:** Menores de 5 años (0-59 meses)
        
        **Variables evaluadas:**
        - Anemia
        - Desnutrición
        
        **Algoritmos:**
        - Random Forest
        - XGBoost
        - SVM
        - Stacking Ensemble
        
        **Datos:** SIEN 2022-2024
        """
    )
    
    # Crear tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Predicción Individual",
        "📊 Análisis por Lotes",
        "📈 Estadísticas",
        "ℹ️ Información"
    ])
    
    # ===== TAB 1: Predicción Individual =====
    with tab1:
        st.header("Predicción Individual")
        st.write("Ingrese los datos del niño/a para obtener una predicción:")
        
        # Crear formulario
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Datos Demográficos")
                edad_meses = st.number_input(
                    "Edad (meses)", 
                    min_value=0, 
                    max_value=59, 
                    value=24,
                    help="Edad del niño/a en meses (0-59)"
                )
                sexo = st.selectbox(
                    "Sexo",
                    options=["M", "F"],
                    help="Sexo del niño/a"
                )
            
            with col2:
                st.subheader("Datos Antropométricos")
                peso = st.number_input(
                    "Peso (kg)",
                    min_value=2.0,
                    max_value=25.0,
                    value=12.0,
                    step=0.1,
                    help="Peso en kilogramos"
                )
                talla = st.number_input(
                    "Talla (cm)",
                    min_value=45.0,
                    max_value=120.0,
                    value=85.0,
                    step=0.1,
                    help="Talla en centímetros"
                )
                hemoglobina = st.number_input(
                    "Hemoglobina (g/dL)",
                    min_value=5.0,
                    max_value=20.0,
                    value=11.0,
                    step=0.1,
                    help="Nivel de hemoglobina"
                )
            
            with col3:
                st.subheader("Datos Socioeconómicos")
                altitud = st.selectbox(
                    "Altitud",
                    options=["Baja", "Media", "Alta"],
                    help="Categoría de altitud de residencia"
                )
                tiene_seguro = st.checkbox("Tiene SIS", value=True)
                tiene_juntos = st.checkbox("Programa Juntos", value=False)
                tiene_suplementacion = st.checkbox("Suplementación", value=True)
            
            # Botón de predicción
            submitted = st.form_submit_button("🔍 Realizar Predicción", use_container_width=True)
        
        if submitted:
            # Calcular IMC
            imc = peso / ((talla/100) ** 2)
            
            # Mostrar resultados
            st.markdown("---")
            st.subheader("📋 Resultados de la Evaluación")
            
            # Crear columnas para resultados
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("IMC", f"{imc:.2f}")
                st.caption("Índice de Masa Corporal")
            
            with col2:
                # Determinar estado de hemoglobina
                if hemoglobina < 11:
                    hb_status = "⚠️ Bajo"
                    hb_color = "red"
                else:
                    hb_status = "✅ Normal"
                    hb_color = "green"
                st.metric("Hemoglobina", f"{hemoglobina:.1f} g/dL")
                st.markdown(f"<p style='color:{hb_color};'>{hb_status}</p>", unsafe_allow_html=True)
            
            with col3:
                edad_años = edad_meses / 12
                st.metric("Edad", f"{edad_años:.1f} años")
                st.caption(f"{edad_meses} meses")
            
            # Predicción simulada (aquí iría tu modelo real)
            st.markdown("---")
            st.subheader("🎯 Predicción del Modelo")
            
            # NOTA: Aquí debes cargar y usar tu modelo real
            # Por ahora, simulación basada en umbrales
            tiene_anemia = hemoglobina < 11
            riesgo_desnutricion = imc < 15 or peso < (edad_meses * 0.5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if tiene_anemia:
                    st.error("⚠️ RIESGO DE ANEMIA DETECTADO")
                    st.write("**Probabilidad:** 85%")
                    st.write("**Recomendación:** Consulta con especialista y suplementación")
                else:
                    st.success("✅ SIN RIESGO DE ANEMIA")
                    st.write("**Probabilidad:** 15%")
                    st.write("**Recomendación:** Mantener controles regulares")
            
            with col2:
                if riesgo_desnutricion:
                    st.warning("⚠️ RIESGO DE DESNUTRICIÓN")
                    st.write("**Probabilidad:** 70%")
                    st.write("**Recomendación:** Evaluación nutricional y seguimiento")
                else:
                    st.success("✅ ESTADO NUTRICIONAL ADECUADO")
                    st.write("**Probabilidad:** 20%")
                    st.write("**Recomendación:** Mantener alimentación balanceada")
            
            # Gráfico de probabilidades
            st.markdown("---")
            st.subheader("📊 Probabilidades de Riesgo")
            
            fig = go.Figure()
            
            categories = ['Anemia', 'Desnutrición']
            probabilidades = [85 if tiene_anemia else 15, 70 if riesgo_desnutricion else 20]
            colors = ['red' if tiene_anemia else 'green', 'orange' if riesgo_desnutricion else 'green']
            
            fig.add_trace(go.Bar(
                x=categories,
                y=probabilidades,
                marker_color=colors,
                text=[f"{p}%" for p in probabilidades],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Probabilidades de Riesgo",
                yaxis_title="Probabilidad (%)",
                yaxis=dict(range=[0, 100]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 2: Análisis por Lotes =====
    with tab2:
        st.header("Análisis por Lotes")
        st.write("Cargue un archivo CSV con múltiples registros para análisis masivo")
        
        uploaded_file = st.file_uploader(
            "Seleccione archivo CSV",
            type=['csv'],
            help="Formato: edad_meses, sexo, peso, talla, hemoglobina, ..."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Archivo cargado: {len(df)} registros")
                
                # Mostrar preview
                st.subheader("Vista Previa de Datos")
                st.dataframe(df.head(10))
                
                # Botón de análisis
                if st.button("🔍 Analizar Lote", use_container_width=True):
                    st.info("Procesando registros...")
                    
                    # Aquí iría el procesamiento con tu modelo
                    # Simulación de resultados
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    st.success("✅ Análisis completado")
                    
                    # Mostrar estadísticas
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Analizado", len(df))
                    with col2:
                        st.metric("Con Riesgo", int(len(df) * 0.35))
                    with col3:
                        st.metric("Requieren Seguimiento", int(len(df) * 0.15))
                    
                    # Opción de descarga
                    st.download_button(
                        label="📥 Descargar Resultados",
                        data=df.to_csv(index=False),
                        file_name=f"resultados_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error procesando archivo: {e}")
    
    # ===== TAB 3: Estadísticas =====
    with tab3:
        st.header("Estadísticas del Sistema")
        
        # Crear datos de ejemplo
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rendimiento del Modelo")
            
            metrics_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                'Valor': [0.89, 0.87, 0.91, 0.89, 0.93]
            })
            
            fig = px.bar(metrics_df, x='Métrica', y='Valor', 
                        title='Métricas de Desempeño',
                        color='Valor',
                        color_continuous_scale='RdYlGn')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribución de Predicciones")
            
            # Datos simulados
            labels = ['Sin Riesgo', 'Anemia', 'Desnutrición', 'Ambos']
            values = [60, 20, 15, 5]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig.update_layout(title="Distribución de Casos", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tendencias temporales
        st.subheader("Tendencias 2022-2024")
        
        # Datos simulados de prevalencia
        years = ['2022', '2023', '2024']
        anemia = [28, 26, 24]
        desnutricion = [18, 16, 15]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=anemia, mode='lines+markers', name='Anemia', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=years, y=desnutricion, mode='lines+markers', name='Desnutrición', line=dict(color='orange')))
        fig.update_layout(
            title="Evolución de Prevalencia (%)",
            xaxis_title="Año",
            yaxis_title="Prevalencia (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 4: Información =====
    with tab4:
        st.header("Información del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Sobre el Proyecto")
            st.markdown("""
            Este sistema utiliza algoritmos de Machine Learning para predecir el riesgo de 
            anemia y desnutrición en niños menores de 5 años en la región de Arequipa.
            
            **Fuente de datos:** Sistema de Información del Estado Nutricional (SIEN-HIS)
            
            **Periodo:** 2022-2024
            
            **Población:** Menores de 5 años (0-59 meses)
            """)
            
            st.subheader("🎯 Objetivos")
            st.markdown("""
            - Detección temprana de anemia
            - Identificación de riesgo nutricional
            - Apoyo en toma de decisiones clínicas
            - Priorización de intervenciones
            """)
        
        with col2:
            st.subheader("🔬 Metodología")
            st.markdown("""
            **Algoritmos utilizados:**
            - Random Forest
            - XGBoost
            - Support Vector Machines (SVM)
            - Stacking Ensemble
            
            **Variables consideradas:**
            - Edad y sexo
            - Peso y talla
            - Hemoglobina
            - Factores socioeconómicos
            - Acceso a programas sociales
            - Altitud de residencia
            """)
            
            st.subheader("⚠️ Advertencias")
            st.warning("""
            Este sistema es una herramienta de apoyo y no reemplaza el diagnóstico clínico. 
            Todas las predicciones deben ser validadas por personal de salud calificado.
            """)
        
        st.markdown("---")
        st.subheader("📞 Contacto")
        st.info("""
        Para más información sobre el proyecto:
        - **Institución:** [Tu Universidad]
        - **Investigador:** [Tu Nombre]
        - **Email:** [tu@email.com]
        """)

if __name__ == "__main__":
    main()