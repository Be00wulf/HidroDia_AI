import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- RUTA DE ARCHIVOS ---
DATA_DIR = 'data/'
MAIN_DATA_FILE = os.path.join(DATA_DIR, 'dataset.csv')         # Archivo de síntomas y enfermedades
SEVERITY_FILE = os.path.join(DATA_DIR, 'Symptom-severity.csv') # Archivo de pesos de síntomas
MODEL_DIR = 'models/'

# --- FUNCIONES DE PREPROCESAMIENTO ---

def load_and_map_severity(df_main, df_severity):
    """
    Carga la escala de gravedad y transforma el dataset principal a un formato
    One-Hot Encoding con pesos, donde cada columna es un síntoma único (132 columnas).
    """
    
    # 1. Limpiar y crear el mapa de gravedad (Symptom -> Weight)
    df_severity.columns = ['Symptom', 'Weight']
    df_severity['Symptom'] = df_severity['Symptom'].str.replace('_', ' ').str.strip().str.lower()
    severity_map = df_severity.set_index('Symptom')['Weight'].to_dict()
    
    # 2. Preparar el DataFrame principal
    df_main.columns = df_main.columns.str.replace('_', ' ').str.strip().str.lower()
    disease_column = df_main.columns[0]
    symptom_columns = df_main.columns[1:]
    
    # 3. Extraer el vocabulario completo de síntomas del dataset principal
    # Aplanar el dataframe de síntomas para obtener la lista de todos los síntomas reales presentes
    all_symptoms_unique = pd.Series(df_main[symptom_columns].values.ravel()).dropna().unique()
    
    # Normalizar esta lista de síntomas 
    all_symptoms_unique_cleaned = [s.replace('_', ' ').strip().lower() for s in all_symptoms_unique]
    
    # 4. Crear el DataFrame final con formato One-Hot Ponderado
    # Las columnas serán todos los síntomas únicos.
    X_data_weighted = pd.DataFrame(0, index=df_main.index, columns=all_symptoms_unique_cleaned)
    
    # Rellenar la matriz ponderada (X_data_weighted)
    for index, row in df_main.iterrows():
        for col in symptom_columns:
            symptom = str(row[col]).replace('_', ' ').strip().lower()
            if symptom in severity_map:
                weight = severity_map.get(symptom, 1) # Obtener el peso, default 1
                
                # Asignar el peso en la columna correspondiente al síntoma real
                if symptom in X_data_weighted.columns:
                    X_data_weighted.loc[index, symptom] = weight

    X = X_data_weighted.values
    
    print(f"Número de síntomas ÚNICOS para el entrenamiento: {X.shape[1]}")
    
    # Devolvemos el DataFrame mapeado (con la columna de enfermedad) y la lista de columnas (todos los síntomas)
    df_mapped = df_main[[disease_column]].copy()
    df_mapped = pd.concat([df_mapped, X_data_weighted], axis=1)

    return df_mapped, X, severity_map, all_symptoms_unique_cleaned

def encode_labels(df, disease_col):
    """Codifica la columna de enfermedades (y) a números y aplica One-Hot Encoding."""
    
    le = LabelEncoder()
    # Aseguramos que la columna de enfermedad esté limpia
    df[disease_col] = df[disease_col].str.strip().str.lower()
    
    y_encoded = le.fit_transform(df[disease_col])
    y_one_hot = to_categorical(y_encoded)
    
    print(f"Número de Clases de Enfermedades: {len(le.classes_)}")
    print(f"Dimensión de Y (One-Hot Encoded): {y_one_hot.shape}")
    
    return y_one_hot, le

# Lógica principal de ejecución
def prepare_data():
    if not os.path.exists(MAIN_DATA_FILE) or not os.path.exists(SEVERITY_FILE):
        print("ERROR: Asegúrate de que los archivos 'dataset.csv' y 'Symptom-severity.csv' estén en la carpeta 'data/'.")
        return None, None, None, None, None, None
    
    print("Iniciando la carga, transformación y ponderación de datos a 132 características...")
    
    # 1. Cargar datos
    df_main = pd.read_csv(MAIN_DATA_FILE)
    df_severity = pd.read_csv(SEVERITY_FILE)
    
    # 2. Ponderar síntomas (X_data ahora tiene 132 columnas)
    disease_column = df_main.columns[0].strip().lower().replace('_', ' ')
    df_mapped, X_data, severity_map, symptom_cols_list = load_and_map_severity(df_main, df_severity)
    
    # 3. Codificar la etiqueta (Y)
    Y_data, label_encoder = encode_labels(df_mapped, disease_column)
    
    # 4. Dividir para Entrenamiento/Prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    
    # 5. Guardar objetos esenciales para la predicción
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
        
    # Guardamos la lista de los 132 síntomas reales
    with open(os.path.join(MODEL_DIR, 'symptom_cols_list.pkl'), 'wb') as f:
        pickle.dump(symptom_cols_list, f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(MODEL_DIR, 'severity_map.pkl'), 'wb') as f:
        pickle.dump(severity_map, f)

    print("\n--- Resultados de División ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Objetos guardados. El modelo Keras debe ser reentrenado con input_shape={X_train.shape[1]}.")
    
    return X_train, X_test, Y_train, Y_test, label_encoder, severity_map

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, le, severity_map = prepare_data()
    
    if X_train is not None:
        print("\nPreprocesamiento de datos completado")