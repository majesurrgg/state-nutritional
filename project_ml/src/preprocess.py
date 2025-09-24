# preprocesamiento de datos
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class DataPreprocessor:
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_datasets(self):
        """Cargar datasets de HB y PT"""
        try:
            # Cargar datos de hemoglobina
            hb_files = [f for f in os.listdir(os.path.join(self.data_path, 'hb/')) if f.endswith('.csv')]
            pt_files = [f for f in os.listdir(os.path.join(self.data_path, 'pt/')) if f.endswith('.csv')]
            
            # Cargar y concatenar archivos HB
            hb_dfs = []
            for file in hb_files:
                df = pd.read_csv(os.path.join(self.data_path, 'hb/', file))
                hb_dfs.append(df)
            hb_data = pd.concat(hb_dfs, ignore_index=True)
            
            # Cargar y concatenar archivos PT
            pt_dfs = []
            for file in pt_files:
                df = pd.read_csv(os.path.join(self.data_path, 'pt/', file))
                pt_dfs.append(df)
            pt_data = pd.concat(pt_dfs, ignore_index=True)
            
            print(f"HB data shape: {hb_data.shape}")
            print(f"PT data shape: {pt_data.shape}")
            
            return hb_data, pt_data
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return None, None
    
    def merge_datasets(self, hb_data, pt_data):
        """Unir datasets de HB y PT"""
        # Crear clave única para el merge
        hb_data['merge_key'] = (hb_data['Renipress'].astype(str) + '_' + 
                               hb_data['FechaAtencion'].astype(str) + '_' + 
                               hb_data['EdadMeses'].astype(str) + '_' + 
                               hb_data['Sexo'].astype(str))
        
        pt_data['merge_key'] = (pt_data['Renipress'].astype(str) + '_' + 
                               pt_data['FechaAtencion'].astype(str) + '_' + 
                               pt_data['EdadMeses'].astype(str) + '_' + 
                               pt_data['Sexo'].astype(str))
        
        # Merge de datasets
        merged_data = pd.merge(hb_data, pt_data, on='merge_key', how='inner', suffixes=('_hb', '_pt'))
        
        print(f"Merged data shape: {merged_data.shape}")
        print(f"Merge efficiency: {len(merged_data)/min(len(hb_data), len(pt_data))*100:.1f}%")
        
        return merged_data
    
    def clean_data(self, df):
        """Limpieza de datos"""
        print("Iniciando limpieza de datos...")
        
        # Eliminar duplicados
        df = df.drop_duplicates(subset=['merge_key'])
        
        # Limpiar valores faltantes críticos
        df = df.dropna(subset=['Hemoglobina', 'Peso', 'Talla', 'EdadMeses_hb'])
        
        # Convertir a numérico las columnas críticas
        numeric_columns = ['Hemoglobina', 'Peso', 'Talla', 'EdadMeses_hb', 'AlturaREN_hb']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar filas con valores nulos después de conversión
        df = df.dropna(subset=numeric_columns)
        
        # Convertir fechas
        if 'FechaAtencion_hb' in df.columns:
            df['FechaAtencion_hb'] = pd.to_datetime(df['FechaAtencion_hb'], errors='coerce')
        if 'FechaNacimiento_hb' in df.columns:
            df['FechaNacimiento_hb'] = pd.to_datetime(df['FechaNacimiento_hb'], errors='coerce')
        
        # Filtros de calidad (solo si las columnas existen y son numéricas)
        if 'Hemoglobina' in df.columns:
            df = df[(df['Hemoglobina'] > 5) & (df['Hemoglobina'] < 20)]  # Valores realistas
        if 'Peso' in df.columns:
            df = df[(df['Peso'] > 10) & (df['Peso'] < 80)]  # Peso realista para 5-11 años
        if 'Talla' in df.columns:
            df = df[(df['Talla'] > 80) & (df['Talla'] < 160)]  # Talla realista
        if 'EdadMeses_hb' in df.columns:
            df = df[(df['EdadMeses_hb'] >= 60) & (df['EdadMeses_hb'] <= 132)]  # 5-11 años
        
        print(f"Datos después de limpieza: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        """Ingeniería de características"""
        print("Creando nuevas características...")
        
        # Verificar qué columnas existen
        print("Columnas disponibles:", df.columns.tolist())
        
        # Características demográficas
        if 'EdadMeses_hb' in df.columns:
            df['Edad_años'] = pd.to_numeric(df['EdadMeses_hb'], errors='coerce') / 12
            df['Grupo_edad'] = pd.cut(df['Edad_años'], 
                                     bins=[5, 7, 9, 11], 
                                     labels=['5-6años', '7-8años', '9-11años'],
                                     include_lowest=True)
        
        # Características geográficas
        if 'AlturaREN_hb' in df.columns:
            altitud_values = pd.to_numeric(df['AlturaREN_hb'], errors='coerce')
            df['Categoria_altitud'] = pd.cut(altitud_values, 
                                            bins=[0, 2500, 3500, 5000], 
                                            labels=['Baja', 'Media', 'Alta'],
                                            include_lowest=True)
        
        # Características socioeconómicas
        programas_sociales = []
        for col in ['Juntos_hb', 'SIS_hb', 'Pin_hb', 'Qaliwarma_hb']:
            if col in df.columns:
                # Convertir a numérico, tratando valores no numéricos como 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                programas_sociales.append(col)
        
        if programas_sociales:
            df['Num_programas'] = df[programas_sociales].sum(axis=1)
            df['Tiene_programas'] = (df['Num_programas'] > 0).astype(int)
        else:
            df['Num_programas'] = 0
            df['Tiene_programas'] = 0
        
        # Características de servicios de salud
        servicios = []
        for col in ['Cred_hb', 'Suplementacion_hb', 'Consejeria_hb']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                servicios.append(col)
        
        if servicios:
            df['Num_servicios'] = df[servicios].sum(axis=1)
        else:
            df['Num_servicios'] = 0
        
        # IMC
        if 'Peso' in df.columns and 'Talla' in df.columns:
            peso = pd.to_numeric(df['Peso'], errors='coerce')
            talla = pd.to_numeric(df['Talla'], errors='coerce')
            df['IMC'] = peso / (talla/100)**2
        
        # Variables target
        # Variable de anemia - vamos a investigar qué valores tiene
        if 'Dx_anemia' in df.columns:
            print(f"Valores únicos en Dx_anemia: {df['Dx_anemia'].unique()}")
            print(f"Conteo de valores en Dx_anemia: {df['Dx_anemia'].value_counts()}")
            
            # Ser más flexible en la detección de anemia
            anemia_keywords = ['SI', 'SÍ', 'SÍ', 'S', 'YES', '1', 'TRUE', 'LEVE', 'MODERADA', 'SEVERA']
            df['Anemia'] = (
                df['Dx_anemia'].astype(str).str.upper().str.strip().isin(anemia_keywords) |
                df['Dx_anemia'].astype(str).str.upper().str.contains('ANEMIA', na=False)
            ).astype(int)
        else:
            print("⚠️  Columna Dx_anemia no encontrada. Creando variable dummy.")
            df['Anemia'] = 0
        
        # Variable de desnutrición crónica
        if 'Dx_TE' in df.columns:
            print(f"Valores únicos en Dx_TE: {df['Dx_TE'].unique()}")
            df['Desnutricion_cronica'] = (
                df['Dx_TE'].astype(str).str.upper().str.contains('BAJA|DESNU|TALLA BAJA', na=False)
            ).astype(int)
        else:
            df['Desnutricion_cronica'] = 0
        
        # Variable de estado nutricional alterado
        if 'Dx_IMCE' in df.columns:
            print(f"Valores únicos en Dx_IMCE: {df['Dx_IMCE'].unique()}")
            alteraciones = ['DESNUTRICION', 'SOBREPESO', 'OBESIDAD', 'AGUDA']
            df['Estado_nutricional_alterado'] = (
                df['Dx_IMCE'].astype(str).str.upper().str.contains('|'.join(alteraciones), na=False) |
                (df['Desnutricion_cronica'] == 1)
            ).astype(int)
        else:
            df['Estado_nutricional_alterado'] = df['Desnutricion_cronica']
        
        # Variable target combinada
        df['Riesgo_nutricional'] = 0  # Normal
        df.loc[df['Anemia'] == 1, 'Riesgo_nutricional'] = 1  # Solo anemia
        df.loc[df['Estado_nutricional_alterado'] == 1, 'Riesgo_nutricional'] = 1  # Solo desnutrición
        df.loc[(df['Anemia'] == 1) & (df['Estado_nutricional_alterado'] == 1), 'Riesgo_nutricional'] = 2  # Ambos
        
        # Mostrar distribución de variables target
        print(f"Distribución Anemia: {df['Anemia'].value_counts().to_dict()}")
        print(f"Distribución Desnutrición: {df['Estado_nutricional_alterado'].value_counts().to_dict()}")
        print(f"Distribución Riesgo: {df['Riesgo_nutricional'].value_counts().to_dict()}")
        
        return df
    
    def prepare_features(self, df):
        """Preparar características para el modelo"""
        # Características numéricas básicas que deberían estar siempre
        numeric_features = []
        
        # Verificar y agregar características numéricas disponibles
        possible_numeric = [
            'EdadMeses_hb', 'Hemoglobina', 'Hbc', 'Peso', 'Talla', 
            'ZPE', 'ZTE', 'ZIMCE', 'AlturaREN_hb', 'Num_programas', 
            'Num_servicios', 'IMC', 'Edad_años'
        ]
        
        for col in possible_numeric:
            if col in df.columns:
                # Convertir a numérico si no lo está
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_features.append(col)
        
        # Características categóricas
        categorical_features = []
        possible_categorical = ['Sexo_hb', 'Grupo_edad', 'Categoria_altitud']
        
        for col in possible_categorical:
            if col in df.columns:
                # Convertir a string y manejar valores faltantes ANTES de crear categorías
                df[col] = df[col].astype(str).replace('nan', 'Unknown')
                categorical_features.append(col)
        
        print(f"Características numéricas: {numeric_features}")
        print(f"Características categóricas: {categorical_features}")
        
        # Crear DataFrame de características
        feature_cols = numeric_features + categorical_features
        X = df[feature_cols].copy()
        
        # Llenar valores faltantes
        for col in numeric_features:
            X[col] = X[col].fillna(X[col].median())
        
        for col in categorical_features:
            # Para variables categóricas, convertir a string y manejar NaN
            X[col] = X[col].astype(str).replace('nan', 'Unknown').replace('NaN', 'Unknown')
        
        # Encoding de variables categóricas
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                # Manejar categorías nuevas que no estaban en el entrenamiento
                unique_values = X[col].unique()
                known_values = self.label_encoders[col].classes_
                new_values = set(unique_values) - set(known_values)
                
                if new_values:
                    print(f"⚠️  Nuevas categorías encontradas en {col}: {new_values}")
                    # Agregar nuevas categorías al encoder
                    all_categories = np.concatenate([known_values, list(new_values)])
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(all_categories)
                
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Variables target
        y_anemia = df['Anemia']
        y_desnutricion = df['Estado_nutricional_alterado'] 
        y_riesgo = df['Riesgo_nutricional']
        
        print(f"Forma final de X: {X.shape}")
        print(f"Características finales: {X.columns.tolist()}")
        
        return X, y_anemia, y_desnutricion, y_riesgo
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """Dividir datos en train/val/test"""
        # Primero dividir en train+val y test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Luego dividir train+val en train y val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Escalar características numéricas"""
        # Identificar columnas numéricas
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Ajustar scaler en train y transformar todos los sets
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_val_scaled[numeric_cols] = self.scaler.transform(X_val[numeric_cols])
        X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def process_data(self, target='anemia', save_processed=True):
        """Pipeline completo de procesamiento"""
        # 1. Cargar datos
        hb_data, pt_data = self.load_datasets()
        if hb_data is None or pt_data is None:
            return None
        
        # 2. Unir datasets
        merged_data = self.merge_datasets(hb_data, pt_data)
        
        # 3. Limpiar datos
        clean_data = self.clean_data(merged_data)
        
        # 4. Ingeniería de características
        featured_data = self.feature_engineering(clean_data)
        
        # 5. Preparar características
        X, y_anemia, y_desnutricion, y_riesgo = self.prepare_features(featured_data)
        
        # Seleccionar target apropiado
        if target == 'anemia':
            y = y_anemia
        elif target == 'desnutricion':
            y = y_desnutricion
        elif target == 'riesgo':
            y = y_riesgo
        else:
            y = y_anemia  # Por defecto
        
        # 6. Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 7. Escalar características
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        # 8. Guardar datos procesados
        if save_processed:
            os.makedirs('data/processed', exist_ok=True)
            
            # Guardar datos originales
            featured_data.to_csv('data/processed/merged_data.csv', index=False)
            
            # Guardar splits
            train_df = pd.concat([X_train_scaled, y_train], axis=1)
            val_df = pd.concat([X_val_scaled, y_val], axis=1) 
            test_df = pd.concat([X_test_scaled, y_test], axis=1)
            
            train_df.to_csv('data/processed/train_data.csv', index=False)
            val_df.to_csv('data/processed/val_data.csv', index=False)
            test_df.to_csv('data/processed/test_data.csv', index=False)
        
        print(f"Procesamiento completado:")
        print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        print(f"Distribución target - Train: {y_train.value_counts().to_dict()}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.process_data(target='anemia')
    if data:
        print("¡Preprocesamiento exitoso!")