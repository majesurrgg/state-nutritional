# Análisis Exploratorio de Datos (EDA)
# Genera visualizaciones y estadísticas descriptivas
# para comprender los datos antes del modelado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class ExploratoryDataAnalysis:
    def __init__(self, data_path='data/processed/merged_data.csv'):
        self.data_path = data_path
        self.df = None
        self.report = []
        
    def load_data(self):
        """Cargar datos procesados"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Datos cargados: {self.df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def general_statistics(self):
        """Estadísticas generales del dataset"""
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS GENERALES")
        print("="*60)
        
        self.report.append("="*60)
        self.report.append("REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS")
        self.report.append("="*60)
        self.report.append(f"\n1. INFORMACIÓN GENERAL DEL DATASET")
        self.report.append(f"   • Total de registros: {len(self.df):,}")
        self.report.append(f"   • Total de variables: {len(self.df.columns)}")
        self.report.append(f"   • Periodo: 2022-2024")
        self.report.append(f"   • Población: Menores de 5 años (0-59 meses)")
        
        # Información por año
        if 'año' in self.df.columns:
            year_counts = self.df['año'].value_counts().sort_index()
            self.report.append(f"\n   Distribución por año:")
            for year, count in year_counts.items():
                self.report.append(f"   • {year}: {count:,} registros ({count/len(self.df)*100:.1f}%)")
        
        # Valores faltantes
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        if missing.sum() > 0:
            self.report.append(f"\n   Variables con valores faltantes:")
            for col in missing[missing > 0].index:
                self.report.append(f"   • {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")
        
        print('\n'.join(self.report[-10:]))
    
    def demographic_analysis(self):
        """Análisis demográfico"""
        print("\n" + "="*60)
        print("👶 ANÁLISIS DEMOGRÁFICO")
        print("="*60)
        
        self.report.append(f"\n2. ANÁLISIS DEMOGRÁFICO")
        
        # Distribución por sexo
        if 'Sexo_hb' in self.df.columns:
            sex_dist = self.df['Sexo_hb'].value_counts()
            self.report.append(f"\n   Distribución por sexo:")
            for sex, count in sex_dist.items():
                self.report.append(f"   • {sex}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        # Distribución por grupo de edad
        if 'Grupo_edad' in self.df.columns:
            age_dist = self.df['Grupo_edad'].value_counts().sort_index()
            self.report.append(f"\n   Distribución por grupo de edad:")
            for age, count in age_dist.items():
                self.report.append(f"   • {age}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        # Estadísticas de edad
        if 'EdadMeses_hb' in self.df.columns:
            edad = self.df['EdadMeses_hb']
            self.report.append(f"\n   Estadísticas de edad (meses):")
            self.report.append(f"   • Media: {edad.mean():.1f}")
            self.report.append(f"   • Mediana: {edad.median():.1f}")
            self.report.append(f"   • Desviación estándar: {edad.std():.1f}")
            self.report.append(f"   • Rango: {edad.min():.0f} - {edad.max():.0f}")
        
        print('\n'.join(self.report[-15:]))
    
    def nutritional_analysis(self):
        """Análisis de indicadores nutricionales"""
        print("\n" + "="*60)
        print("🍎 ANÁLISIS NUTRICIONAL")
        print("="*60)
        
        self.report.append(f"\n3. ANÁLISIS DE INDICADORES NUTRICIONALES")
        
        # Prevalencia de anemia
        if 'Anemia' in self.df.columns:
            anemia_dist = self.df['Anemia'].value_counts()
            prevalencia_anemia = (anemia_dist.get(1, 0) / len(self.df) * 100)
            self.report.append(f"\n   ANEMIA:")
            self.report.append(f"   • Prevalencia: {prevalencia_anemia:.2f}%")
            self.report.append(f"   • Con anemia: {anemia_dist.get(1, 0):,}")
            self.report.append(f"   • Sin anemia: {anemia_dist.get(0, 0):,}")
        
        # Prevalencia de desnutrición
        if 'Estado_nutricional_alterado' in self.df.columns:
            desnut_dist = self.df['Estado_nutricional_alterado'].value_counts()
            prevalencia_desnut = (desnut_dist.get(1, 0) / len(self.df) * 100)
            self.report.append(f"\n   DESNUTRICIÓN:")
            self.report.append(f"   • Prevalencia: {prevalencia_desnut:.2f}%")
            self.report.append(f"   • Con desnutrición: {desnut_dist.get(1, 0):,}")
            self.report.append(f"   • Sin desnutrición: {desnut_dist.get(0, 0):,}")
        
        # Estadísticas de hemoglobina
        if 'Hemoglobina' in self.df.columns:
            hb = self.df['Hemoglobina']
            self.report.append(f"\n   HEMOGLOBINA (g/dL):")
            self.report.append(f"   • Media: {hb.mean():.2f}")
            self.report.append(f"   • Mediana: {hb.median():.2f}")
            self.report.append(f"   • Desviación estándar: {hb.std():.2f}")
            self.report.append(f"   • Rango: {hb.min():.1f} - {hb.max():.1f}")
        
        # Estadísticas antropométricas
        if 'Peso' in self.df.columns and 'Talla' in self.df.columns:
            self.report.append(f"\n   ANTROPOMETRÍA:")
            self.report.append(f"   • Peso medio: {self.df['Peso'].mean():.2f} kg")
            self.report.append(f"   • Talla media: {self.df['Talla'].mean():.2f} cm")
            if 'IMC' in self.df.columns:
                self.report.append(f"   • IMC medio: {self.df['IMC'].mean():.2f}")
        
        print('\n'.join(self.report[-20:]))
    
    def socioeconomic_analysis(self):
        """Análisis de factores socioeconómicos"""
        print("\n" + "="*60)
        print("🏘️ ANÁLISIS SOCIOECONÓMICO")
        print("="*60)
        
        self.report.append(f"\n4. ANÁLISIS SOCIOECONÓMICO")
        
        # Programas sociales
        programas = ['Juntos_hb', 'SIS_hb', 'Pin_hb', 'Qaliwarma_hb']
        programas_existentes = [p for p in programas if p in self.df.columns]
        
        if programas_existentes:
            self.report.append(f"\n   COBERTURA DE PROGRAMAS SOCIALES:")
            for programa in programas_existentes:
                cobertura = self.df[programa].sum()
                pct = (cobertura / len(self.df) * 100)
                nombre = programa.replace('_hb', '')
                self.report.append(f"   • {nombre}: {cobertura:,} ({pct:.1f}%)")
        
        # Servicios de salud
        servicios = ['Cred_hb', 'Suplementacion_hb', 'Consejeria_hb']
        servicios_existentes = [s for s in servicios if s in self.df.columns]
        
        if servicios_existentes:
            self.report.append(f"\n   ACCESO A SERVICIOS DE SALUD:")
            for servicio in servicios_existentes:
                acceso = self.df[servicio].sum()
                pct = (acceso / len(self.df) * 100)
                nombre = servicio.replace('_hb', '')
                self.report.append(f"   • {nombre}: {acceso:,} ({pct:.1f}%)")
        
        # Distribución por altitud
        if 'Categoria_altitud' in self.df.columns:
            altitud_dist = self.df['Categoria_altitud'].value_counts()
            self.report.append(f"\n   DISTRIBUCIÓN POR ALTITUD:")
            for cat, count in altitud_dist.items():
                self.report.append(f"   • {cat}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        print('\n'.join(self.report[-15:]))
    
    def create_visualizations(self):
        """Crear visualizaciones del EDA"""
        print("\n📈 Generando visualizaciones...")
        
        os.makedirs('results/eda', exist_ok=True)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        
        # === FIGURA 1: Distribuciones demográficas ===
        fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribución por edad
        if 'EdadMeses_hb' in self.df.columns:
            axes[0, 0].hist(self.df['EdadMeses_hb'], bins=30, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Edad (meses)')
            axes[0, 0].set_ylabel('Frecuencia')
            axes[0, 0].set_title('Distribución de Edad')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribución por sexo
        if 'Sexo_hb' in self.df.columns:
            sex_counts = self.df['Sexo_hb'].value_counts()
            axes[0, 1].bar(sex_counts.index, sex_counts.values, color=['lightblue', 'lightpink'])
            axes[0, 1].set_xlabel('Sexo')
            axes[0, 1].set_ylabel('Frecuencia')
            axes[0, 1].set_title('Distribución por Sexo')
        
        # 3. Distribución por grupo de edad
        if 'Grupo_edad' in self.df.columns:
            age_counts = self.df['Grupo_edad'].value_counts().sort_index()
            axes[1, 0].bar(range(len(age_counts)), age_counts.values, color='lightgreen')
            axes[1, 0].set_xticks(range(len(age_counts)))
            axes[1, 0].set_xticklabels(age_counts.index, rotation=45)
            axes[1, 0].set_xlabel('Grupo de Edad')
            axes[1, 0].set_ylabel('Frecuencia')
            axes[1, 0].set_title('Distribución por Grupo de Edad')
        
        # 4. Distribución por año
        if 'año' in self.df.columns:
            year_counts = self.df['año'].value_counts().sort_index()
            axes[1, 1].bar(year_counts.index.astype(str), year_counts.values, color='lightsalmon')
            axes[1, 1].set_xlabel('Año')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].set_title('Distribución por Año')
        
        plt.tight_layout()
        plt.savefig('results/eda/01_demografia.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # === FIGURA 2: Indicadores nutricionales ===
        fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribución de hemoglobina
        if 'Hemoglobina' in self.df.columns:
            axes[0, 0].hist(self.df['Hemoglobina'], bins=30, color='coral', edgecolor='black')
            axes[0, 0].axvline(11, color='red', linestyle='--', label='Punto de corte anemia')
            axes[0, 0].set_xlabel('Hemoglobina (g/dL)')
            axes[0, 0].set_ylabel('Frecuencia')
            axes[0, 0].set_title('Distribución de Hemoglobina')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Prevalencia de anemia
        if 'Anemia' in self.df.columns:
            anemia_counts = self.df['Anemia'].value_counts()
            labels = ['Sin Anemia', 'Con Anemia']
            colors = ['lightgreen', 'lightcoral']
            axes[0, 1].pie(anemia_counts.values, labels=labels, colors=colors, 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Prevalencia de Anemia')
        
        # 3. Distribución de peso
        if 'Peso' in self.df.columns:
            axes[1, 0].hist(self.df['Peso'], bins=30, color='lightblue', edgecolor='black')
            axes[1, 0].set_xlabel('Peso (kg)')
            axes[1, 0].set_ylabel('Frecuencia')
            axes[1, 0].set_title('Distribución de Peso')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribución de talla
        if 'Talla' in self.df.columns:
            axes[1, 1].hist(self.df['Talla'], bins=30, color='lightgreen', edgecolor='black')
            axes[1, 1].set_xlabel('Talla (cm)')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].set_title('Distribución de Talla')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/eda/02_nutricional.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # === FIGURA 3: Análisis de anemia por grupos ===
        fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if 'Anemia' in self.df.columns:
            # 1. Anemia por grupo de edad
            if 'Grupo_edad' in self.df.columns:
                anemia_edad = pd.crosstab(self.df['Grupo_edad'], self.df['Anemia'], normalize='index') * 100
                anemia_edad.plot(kind='bar', stacked=True, ax=axes[0, 0], color=['lightgreen', 'lightcoral'])
                axes[0, 0].set_xlabel('Grupo de Edad')
                axes[0, 0].set_ylabel('Porcentaje')
                axes[0, 0].set_title('Prevalencia de Anemia por Grupo de Edad')
                axes[0, 0].legend(['Sin Anemia', 'Con Anemia'])
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Anemia por sexo
            if 'Sexo_hb' in self.df.columns:
                anemia_sexo = pd.crosstab(self.df['Sexo_hb'], self.df['Anemia'], normalize='index') * 100
                anemia_sexo.plot(kind='bar', ax=axes[0, 1], color=['lightgreen', 'lightcoral'])
                axes[0, 1].set_xlabel('Sexo')
                axes[0, 1].set_ylabel('Porcentaje')
                axes[0, 1].set_title('Prevalencia de Anemia por Sexo')
                axes[0, 1].legend(['Sin Anemia', 'Con Anemia'])
                axes[0, 1].tick_params(axis='x', rotation=0)
            
            # 3. Anemia por año
            if 'año' in self.df.columns:
                anemia_year = pd.crosstab(self.df['año'], self.df['Anemia'], normalize='index') * 100
                anemia_year.plot(kind='bar', ax=axes[1, 0], color=['lightgreen', 'lightcoral'])
                axes[1, 0].set_xlabel('Año')
                axes[1, 0].set_ylabel('Porcentaje')
                axes[1, 0].set_title('Evolución de Anemia 2022-2024')
                axes[1, 0].legend(['Sin Anemia', 'Con Anemia'])
                axes[1, 0].tick_params(axis='x', rotation=0)
            
            # 4. Anemia por altitud
            if 'Categoria_altitud' in self.df.columns:
                anemia_alt = pd.crosstab(self.df['Categoria_altitud'], self.df['Anemia'], normalize='index') * 100
                anemia_alt.plot(kind='bar', ax=axes[1, 1], color=['lightgreen', 'lightcoral'])
                axes[1, 1].set_xlabel('Altitud')
                axes[1, 1].set_ylabel('Porcentaje')
                axes[1, 1].set_title('Prevalencia de Anemia por Altitud')
                axes[1, 1].legend(['Sin Anemia', 'Con Anemia'])
                axes[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('results/eda/03_anemia_grupos.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # === FIGURA 4: Correlaciones ===
        numeric_cols = ['EdadMeses_hb', 'Hemoglobina', 'Peso', 'Talla', 'IMC']
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(numeric_cols) > 1:
            fig4, ax = plt.subplots(figsize=(10, 8))
            correlation = self.df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, ax=ax)
            ax.set_title('Matriz de Correlación - Variables Numéricas')
            plt.tight_layout()
            plt.savefig('results/eda/04_correlaciones.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Visualizaciones guardadas en results/eda/")
    
    def save_report(self):
        """Guardar reporte del EDA"""
        os.makedirs('results/reports', exist_ok=True)
        
        with open('results/reports/eda_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        print("✅ Reporte guardado en results/reports/eda_report.txt")
    
    def run_complete_eda(self):
        """Ejecutar análisis exploratorio completo"""
        print("🔍 INICIANDO ANÁLISIS EXPLORATORIO DE DATOS")
        print("="*60)
        
        if not self.load_data():
            return False
        
        # Análisis
        self.general_statistics()
        self.demographic_analysis()
        self.nutritional_analysis()
        self.socioeconomic_analysis()
        
        # Visualizaciones
        self.create_visualizations()
        
        # Guardar reporte
        self.save_report()
        
        print("\n✅ Análisis exploratorio completado")
        print("📁 Resultados en results/eda/ y results/reports/")
        
        return True

def main():
    """Función principal"""
    eda = ExploratoryDataAnalysis()
    eda.run_complete_eda()

if __name__ == "__main__":
    main()