# completar
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

"""Carga de datasets desde archivos CSV"""
class DataLoader:
    """ Cargar hemoglobina """
    def load_hb_data(self):  
        try:
            hb_files = [f for f in os.listdir(os.path.join(
                self.data_path, 'hb/')) if f.endswith('.csv')]

            # Cargar y concatenar archivos HB
            hb_dfs = []
            for file in hb_files:
                df = pd.read_csv(os.path.join(self.data_path, 'hb/', file))
                hb_dfs.append(df)
            hb_data = pd.concat(hb_dfs, ignore_index=True)

            print(f"HB data shape: {hb_data.shape}")
            return hb_data

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    """ Cargar datos de peso/talla """
    def load_pt_data(self):
        try:
            pt_files = [f for f in os.listdir(os.path.join(
                self.data_path, 'pt/')) if f.endswith('.csv')]

            # Cargar y concatenar archivos PT
            pt_dfs = []
            for file in pt_files:
                df = pd.read_csv(os.path.join(self.data_path, 'pt/', file))
                pt_dfs.append(df)
            pt_data = pd.concat(pt_dfs, ignore_index=True)

            print(f"PT data shape: {pt_data.shape}")
            return pt_data

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    """ Unir datasets de hemoglobina y peso/talla """
    def merge_datasets(self, hb_data, pt_data):
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
        merged_data = pd.merge(
            hb_data, pt_data, on='merge_key', how='inner', suffixes=('_hb', '_pt'))

        print(f"Merged data shape: {merged_data.shape}")
        print(
            f"Merge efficiency: {len(merged_data)/min(len(hb_data), len(pt_data))*100:.1f}%")

        return merged_data
