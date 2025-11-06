import streamlit as st
from datetime import datetime
import db_manager
from utils.prediction_logic import get_prediction
import pandas as pd
import sqlite3
import os

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="HidroDia AI - Predictor de Síntomas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar la base de datos y la conexión
CONN = db_manager.initialize_database()

# Definir el nombre del proyecto
PROJECT_NAME = "HidroDia AI"

# Cargar archivos auxiliares para descripciones y precauciones (Opcional, pero recomendado)
# Esto carga los CSV symptom_Description.csv y symptom_precaution.csv
def load_auxiliary_data():
    """Carga descripciones y precauciones para mostrar en la UI."""
    data_path = 'data/'
    
    desc_file = os.path.join(data_path, 'symptom_Description.csv')
    prec_file = os.path.join(data_path, 'symptom_precaution.csv')
    
    desc_df = pd.read_csv(desc_file, index_col='Disease')
    prec_df = pd.read_csv(prec_file, index_col='Disease')
    
    # Normalizar los índices (nombres de enfermedades) a minúsculas
    desc_df.index = desc_df.index.str.strip().str.lower()
    prec_df.index = prec_df.index.str.strip().str.lower()
    
    return desc_df, prec_df

# Cargar datos auxiliares una vez
try:
    DESC_DF, PREC_DF = load_auxiliary_data()
except FileNotFoundError:
    st.error("Error al cargar archivos auxiliares (symptom_Description.csv o symptom_precaution.csv). Asegúrate de que estén en la carpeta 'data/'.")
    DESC_DF = pd.DataFrame()
    PREC_DF = pd.DataFrame()


#  2. LAYOUT PRINCIPAL Y NAVEGACIÓN 

st.title(f"{PROJECT_NAME}")

# Usar pestañas (tabs) para los dos apartados principales: Predicción y Historial
tab1, tab2 = st.tabs(["Predicción y Alerta Inmediata", "Diario y Tendencias (Historial)"])


# APARTADO 1: PREDICCIÓN Y ALERTA INMEDIATA

with tab1:
    st.header("Diagnóstico Rápido de Síntomas")
    
    st.info("**Instrucciones:** Ingresar síntomas separados por comas, ej: vomiting, joint pain ( nombres de síntomas del dataset para mayor precisión)")

    # Área de entrada del usuario
    symptoms_input = st.text_area(
        "Ingresar síntomas:",
        key="symptoms_key",
        placeholder="Ej: high fever, chills, joint pain, headache"
    )

    if st.button("Analizar Síntomas y Predecir", type="primary"):
        if not symptoms_input:
            st.warning("Por favor, ingresa al menos un síntoma para el análisis.")
        else:
            with st.spinner("Analizando con el modelo de IA"):
                # Llamada a la lógica del módulo de utilidad
                top_predictions, hydro_alert, alert_level = get_prediction(symptoms_input)
            
            # RESULTADOS Y ALERTA ESPECIALIZADA
            
            st.subheader("Resultados del Análisis")

            # Módulo de Alerta de Hidrocefalia 
            if alert_level == "danger":
                st.error(f"**ALERTA ESPECIALIZADA:** {hydro_alert}")
            elif alert_level == "warning":
                st.warning(f"**ALERTA ESPECIALIZADA:** {hydro_alert}")
            else:
                st.success(f"**Alerta Especializada:** {hydro_alert}")
            
            st.markdown("---")
            
            # PREDICCIÓN DEL MODELO ML/DL
            
            if top_predictions:
                st.subheader("Clasificación de Posibles Enfermedades")
                
                # Crear columnas para mostrar el top 5 
                cols = st.columns(5)
                
                prediction_text_for_db = ""
                
                for i, (disease, probability) in enumerate(top_predictions):
                    # Formato para guardar en la BD
                    prediction_text_for_db += f"{disease}: {probability:.2f}%; "
                    
                    # Formato para la Interfaz
                    with cols[i]:
                        st.metric(label=f"Posibilidad #{i+1}", value=f"{disease}", delta=f"{probability:.1f}%")

                        # Mostrar descripción y precauciones (si hay)
                        if disease.lower() in DESC_DF.index:
                            st.write(f"**Descripción:** {DESC_DF.loc[disease.lower(), 'Description']}")
                            
                        if disease.lower() in PREC_DF.index:
                            precautions = PREC_DF.loc[disease.lower()].dropna().tolist()
                            if precautions:
                                st.markdown("**Precauciones Recomendadas:**")
                                st.markdown("".join([f"* {p}\n" for p in precautions]))
                            
                st.markdown("---")
                
                # GUARDAR EN EL DIARIO

                st.markdown("##### Registrar en el Diario de Síntomas")
                # El campo `sintomas_codificados` no lo usamos para mostrar, solo para la BD
                
                # Asumiendo que get_prediction devuelve la cadena de síntomas detectados
                # Para simplificar, se usa la entrada del usuario como síntomas_texto para el diario
                
                # entry = (fecha_registro, sintomas_texto, sintomas_codificados, prediccion_modelo, alerta_hidro)
                try:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Llamamos a get_prediction solo para obtener la lista de síntomas detectados si es necesario
                    # Para simplificar, se usa la entrada de texto completa en la BD.
                    
                    entry_data = (
                        current_time,
                        symptoms_input,
                        "Vector de Síntomas Omitido (para simplificar UI)", # Aquí iría el X_predict serializado si fuera necesario
                        prediction_text_for_db.strip(),
                        hydro_alert
                    )
                    
                    db_manager.add_entry(CONN, entry_data)
                    st.success("**Registro Guardado:** Tu entrada ha sido añadida al Diario de Síntomas")
                    
                    # Mensaje final requerido
                    st.warning("**Nota Importante:** Este es un sistema de apoyo. Siempre consulta a un profesional de la salud para un diagnóstico definitivo")
                    
                except sqlite3.Error as e:
                    st.error(f"Error al guardar en la base de datos: {e}")
            else:
                 st.info("No se pudo obtener una predicción válida. ¿Los síntomas ingresados son válidos?")


# APARTADO 2: DIARIO Y TENDENCIAS (HISTORIAL)

with tab2:
    st.header("Historial y Tendencias del Diario")
    
    if st.button("Actualizar Historial", key="refresh_diary"):
        st.cache_data.clear() # Limpiar cache para forzar la recarga
        
    st.subheader("Entradas Recientes del Diario")
    
    try:
        # Recuperar todas las entradas
        df_diary = db_manager.get_all_entries(CONN)
        
        if df_diary.empty:
            st.info("El diario está vacío. Realiza una predicción para agregar la primera entrada.")
        else:
            # Mostrar tabla de historial
            st.dataframe(df_diary, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Análisis de Tendencias (Frecuencia de Síntomas y Alertas)")
            
            #  1: Visualizar Alertas de Hidrocefalia
            alert_counts = df_diary['alerta_hidro'].value_counts()
            st.markdown("##### Frecuencia de Alertas Especializadas")
            st.bar_chart(alert_counts)
            
            #  2: Análisis de Frecuencia de Síntomas NLP Básico en el Historial
            # Esto es complejo, pero simplificamos contando palabras clave
            
            # Concatenar todos los textos de síntomas y convertirlos a minúsculas
            all_symptoms_text = ' '.join(df_diary['sintomas_texto'].astype(str).str.lower()).replace(',', ' ').split()
            
            # Contar la frecuencia de los 10 síntomas más comunes
            symptom_freq = pd.Series(all_symptoms_text).value_counts().head(10)
            
            st.markdown("##### 10 Síntomas Reportados Más Frecuentemente con Análisis Simple")
            st.bar_chart(symptom_freq)
            
    except sqlite3.Error:
        st.error("No se pudo conectar a la base de datos para recuperar el historial")


# Cerrar la conexión cuando la aplicación de Streamlit finaliza la sesión
if CONN:
    CONN.close()