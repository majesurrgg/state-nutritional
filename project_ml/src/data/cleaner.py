# limpieza de datos
class DataCleaner:
    def remove_duplicates(self, df): # eliminar duplicados
        # Eliminar duplicados
        df = df.drop_duplicates(subset=['merge_key'])
    def handle_missing(self, df): # manejar valores faltantes
        # Limpiar valores faltantes críticos
        df = df.dropna(subset=['Hemoglobina', 'Peso', 'Talla', 'EdadMeses_hb'])
        
    def remove_outliers() # eliminar outliers 
    def validate_ranges() # validar rangos médicos
    
        
    