import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

# rutas
MODEL_DIR = 'models/'

# Archivos esenciales del modelo y preprocesamiento
MODEL_PATH = os.path.join(MODEL_DIR, 'disease_predictor_model.keras')
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
# SEV_MAP_PATH = os.path.join(MODEL_DIR, 'severity_map.pkl')
SEV_MAP_PATH = os.path.join(MODEL_DIR, 'severity_map.pkl')
SYMPTOM_COLS_PATH = os.path.join(MODEL_DIR, 'symptom_cols_list.pkl') 

# --- OBJETOS GLOBALES que se cargarán una vez al iniciar la app
# Usamos un diccionario para cargar y almacenar los objetos
MODEL_ASSETS = {}

def load_prediction_assets():
    """Carga el modelo Keras, el Label Encoder y el mapa de gravedad."""
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Modelo Keras no encontrado. hay que ejecutar train_model.py primero")
        return False
    
    try:
        MODEL_ASSETS['model'] = load_model(MODEL_PATH)

        with open(SYMPTOM_COLS_PATH, 'rb') as f:
            MODEL_ASSETS['symptom_cols_list'] = pickle.load(f)

        with open(LE_PATH, 'rb') as f:
            MODEL_ASSETS['label_encoder'] = pickle.load(f)
            
        with open(SEV_MAP_PATH, 'rb') as f:
            MODEL_ASSETS['severity_map'] = pickle.load(f)
            
        print("Activos del modelo cargados exitosamente.")
        return True
    except Exception as e:
        print(f"Error al cargar los activos del modelo: {e}")
        return False

def preprocess_symptoms_for_prediction(symptoms_text):
    """
    Transforma el texto libre de síntomas a un vector ponderado usando la plantilla
    de 17 síntomas con la que se entrenó el modelo.
    """
    if not MODEL_ASSETS:
        load_prediction_assets()
        
    severity_map = MODEL_ASSETS.get('severity_map')
    # la lista de 17 síntomas que el modelo conoce
    symptom_cols_list = MODEL_ASSETS.get('symptom_cols_list') 
    
    if severity_map is None or symptom_cols_list is None:
        return None, None
    
    # 1. Normalizar la entrada del usuario
    # Tomamos la entrada y la separamos por comas
    input_list = [s.strip().replace('_', ' ').lower() for s in symptoms_text.split(',')]
    
    # 2. Crear el vector de características ponderado (X)
    # Inicializar el vector con ceros. SU TAMAÑO ES FIJO: 17
    symptom_vector = np.zeros(len(symptom_cols_list))
    
    detected_symptoms = []
    
    # Mapeo y ponderación:
    for input_symptom in input_list:
        # Buscar si el síntoma está en la lista de 17 síntomas CONOCIDOS por el modelo
        if input_symptom in symptom_cols_list:
            
            # 1. Encontrar el índice dentro del vector de 17 elementos
            index = symptom_cols_list.index(input_symptom)
            
            # 2. Obtener el peso real del mapa de gravedad (que tiene todos los 132 pesos)
            symptom_weight = severity_map.get(input_symptom, 1) # Usar 1 si el peso no se encuentra
            
            # 3. Asignar el peso en la posición correcta (0 a 16) del vector de 17 elementos
            symptom_vector[index] = symptom_weight
            detected_symptoms.append(input_symptom)
        
        # Opcional: Si el usuario pone un síntoma que existe en el severity_map
        # pero que no está entre los 17 originales, lo ignoramos.
    
    # El modelo espera una entrada 2D (1 fila, 17 columnas)
    X_predict = symptom_vector.reshape(1, -1)
    
    return X_predict, ",".join(detected_symptoms)


def apply_symbolic_hydrocephalus_check(detected_symptoms_list):
    """
    Implementa la IA Simbólica (basada en reglas) para alerta de Hidrocefalia.
    Se basa en la Triada clásica y síntomas severos/atípicos.
    """
    symptoms = set(detected_symptoms_list)
    
    # Síntomas clave de urgencia (Ejemplos basados en tu experiencia):
    key_severe_symptoms = {
        "dolor de cabeza",
        "vomiting", 
        "blurred and distorted vision", 
        "dizziness", 
        "unsteadiness" 
    }
    
    # Contar cuántos síntomas severos clave están presentes
    severity_count = len(symptoms.intersection(key_severe_symptoms))
    
    if severity_count >= 3:
        return "ALERTA URGENTE (Hidrocefalia, Triada Sospechosa)", "danger"
    elif severity_count >= 1 and ("vomiting" in symptoms or "blurred and distorted vision" in symptoms):
        return "ALERTA MODERADA (Requiere Seguimiento)", "warning"
    else:
        return "NORMAL", "success"


def get_prediction(symptoms_text):
    """Ejecuta la predicción del modelo y la verificación simbólica"""
    
    if not MODEL_ASSETS:
        if not load_prediction_assets():
            return None, "Error de sistema al cargar el modelo", "danger"

    # 1. Preprocesar la entrada
    X_predict, detected_symptoms_str = preprocess_symptoms_for_prediction(symptoms_text)
    
    if X_predict is None or detected_symptoms_str == "":
        return None, "No se detectaron síntomas válidos conocidos por el sistema", "info"
        
    detected_symptoms_list = detected_symptoms_str.split(',')

    # 2. Predecir con el modelo Keras
    model = MODEL_ASSETS['model']
    le = MODEL_ASSETS['label_encoder']
    
    probabilities = model.predict(X_predict, verbose=0)[0]
    
    # 3. Obtener las top 5 predicciones
    top_indices = np.argsort(probabilities)[::-1][:5]
    top_predictions = []
    
    for i in top_indices:
        disease = le.inverse_transform([i])[0].capitalize()
        probability = probabilities[i] * 100
        top_predictions.append((disease, probability))

    # 4. Aplicar la verificación simbólica de Hidrocefalia
    hydro_alert, alert_level = apply_symbolic_hydrocephalus_check(detected_symptoms_list)
    
    return top_predictions, hydro_alert, alert_level

# Asegurar que los activos se carguen al iniciar
if __name__ == "__main__":
    if load_prediction_assets():
        print("\n--- Prueba de predicción ---")
        # NOTA: Los síntomas DEBEN coincidir con los de tu severity_map.
        # Usa síntomas como 'vomiting', 'headache', 'skin rash'
        test_symptoms = "vomiting, headache, joint pain" 
        predictions, alert, level = get_prediction(test_symptoms)
        
        print(f"Entrada: {test_symptoms}")
        print(f"Alerta de Hidrocefalia: {alert} ({level})")
        print("Predicciones del Modelo:")
        for disease, prob in predictions:
            print(f"- {disease}: {prob:.2f}%")