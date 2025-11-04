import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_DIR = 'data/'
MAIN_DATA_FILE = os.path.join(DATA_DIR, 'dataset.csv')         #síntomas y enfermedades
SEVERITY_FILE = os.path.join(DATA_DIR, 'Symptom-severity.csv') #pesos de síntomas
MODEL_DIR = 'models/'


def load_and_map_severity(df_main, df_severity):
    """
    Carga la escala de gravedad y la mapea al dataset principal.
    Crea un diccionario de mapeo de Symptom -> Weight.
    """
    # LIMPIAR la columna de síntomas y crear el mapa de gravedad
    df_severity.columns = ['Symptom', 'Weight']
    # normalizar los nombres de los síntomas -> sin minúsculas - guiones bajos - espacios extras
    df_severity['Symptom'] = df_severity['Symptom'].str.replace('_', ' ').str.strip().str.lower()
    severity_map = df_severity.set_index('Symptom')['Weight'].to_dict()
    
    # PREPARAR el DataFrame principal => normalizar los nombres de las columnas
    df_main.columns = df_main.columns.str.replace('_', ' ').str.strip().str.lower()
    
    # la primera columna es la enfermedad
    disease_column = df_main.columns[0]
    symptom_columns = df_main.columns[1:]
    
    # MAPEAR síntomas a pesos
    mapped_df = pd.DataFrame()
    mapped_df[disease_column] = df_main[disease_column].str.strip().str.lower()
    
    # RELLENAR los valores nulos en las columnas de síntomas con una cadena vacía 
    # para poder limpiar los síntomas antes de mapearlos
    df_temp = df_main[symptom_columns].fillna('')

    for col in symptom_columns:
        # limpiar el nombre del síntoma en la columna de datos
        symptoms_cleaned = df_temp[col].str.replace('_', ' ').str.strip().str.lower()
        
        # reemplazar el nombre del síntoma por su peso. Si no se encuentra, el peso es 0
        mapped_df[col] = symptoms_cleaned.map(severity_map).fillna(0).astype(int)

    # CREAR la matriz final de síntomas ponderados (X)
    # eliminamos la columna de enfermedad y usamos el resto como características (X)
    X = mapped_df.drop(columns=[disease_column]).values
    
    # imprimiendo estadísticas de limpieza
    print(f"Número de síntomas después de mapeo: {X.shape[1]}")
    
    return mapped_df, X, severity_map

def encode_labels(df, disease_col):
    """Codifica la columna de enfermedades (y) a números y aplica One-Hot Encoding."""
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[disease_col])
    
    # APLICAR one hot encoding: se necesita para la capa de salida Softmax en Keras
    y_one_hot = to_categorical(y_encoded)
    
    print(f"Número de Clases de Enfermedades: {len(le.classes_)}")
    print(f"Dimensión Y one hot encoded: {y_one_hot.shape}")
    
    return y_one_hot, le

# --- Lógica principal de ejecución ---
def prepare_data():
    if not os.path.exists(MAIN_DATA_FILE) or not os.path.exists(SEVERITY_FILE):
        print("ERROR: revisar archivos csv en -> data ")
        return None, None, None, None, None, None
    
    print("\nIniciando la carga y ponderación de datos")
    
    # CARGAMOS datos
    df_main = pd.read_csv(MAIN_DATA_FILE)
    df_severity = pd.read_csv(SEVERITY_FILE)
    
    # PONDERAMOS síntomas con la escala de gravedad
    df_mapped, X_data, severity_map = load_and_map_severity(df_main, df_severity)
    
    # CODIFICAMOS la etiqueta (Y)
    disease_column = df_mapped.columns[0]
    Y_data, label_encoder = encode_labels(df_mapped, disease_column)
    
    # DIVIDIMOS para entrenamiento de prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    
    # GUARDAMOS objetos esenciales para la predicción en el futuro
    # GUARDAMOS el label encoder y el mapa de gravedad.
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(MODEL_DIR, 'severity_map.pkl'), 'wb') as f:
        pickle.dump(severity_map, f)

    print("\nResultados de División")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("Objetos label encoder y severity map guardados en la carpeta models")
    
    return X_train, X_test, Y_train, Y_test, label_encoder, severity_map

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    X_train, X_test, Y_train, Y_test, le, severity_map = prepare_data()
    
    if X_train is not None:
        print("\npreprocesamiento de datos completado")

