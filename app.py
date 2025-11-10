import streamlit as st
from datetime import datetime
import db_manager
from utils.prediction_logic import get_prediction
import pandas as pd
import sqlite3
import os

# 1. CONFIGURACIÓN INICIAL
st.set_page_config(
    page_title="HidroDia AI - Prediccion de Sintomas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# diccionario de datos
ALERT_TRANSLATION = {
    "ALERTA URGENTE (Hidrocefalia, Sintomas Sospechosos)": "ALERTA URGENTE (Hidrocefalia, Sintomas Sospechosos)",
    "ALERTA MODERADA (Requiere Seguimiento)": "ALERTA MODERADA (Requiere Seguimiento)",
    "NORMAL": "NORMAL",
    "danger": "Peligro",
    "warning": "Advertencia",
    "success": "Éxito",
    "info": "Información",
}

COLUMN_TRANSLATION = {
    "fecha_registro": "Fecha de Registro",
    "sintomas_texto": "Sintomas Reportados (Inglés)",
    "prediccion_modelo": "Predicción del Modelo",
    "alerta_hidro": "Alerta Hidrocefalia"
}

# sintomas
TRADUCTION_MAP = {
    "itching": "Picazón",
    "skin_rash": "Erupción cutánea",
    "nodal_skin_eruptions": "Erupciones cutáneas nodulares",
    "continuous_sneezing": "Estornudos continuos",
    "shivering": "Escalofríos",
    "chills": "Tiritona",
    "joint_pain": "Dolor articular",
    "stomach_pain": "Dolor de estómago",
    "acidity": "Acidez",
    "ulcers_on_tongue": "Úlceras en la lengua",
    "muscle_wasting": "Atrofia muscular",
    "vomiting": "Vómitos",
    "burning_micturition": "Ardor al orinar (micción dolorosa)",
    "spotting_urination": "Manchas al orinar",
    "fatigue": "Fatiga",
    "weight_gain": "Aumento de peso",
    "anxiety": "Ansiedad",
    "cold_hands_and_feets": "Manos y pies fríos",
    "mood_swings": "Cambios de humor",
    "weight_loss": "Pérdida de peso",
    "restlessness": "Inquietud",
    "lethargy": "Letargo",
    "patches_in_throat": "Parches en la garganta",
    "irregular_sugar_level": "Nivel de azúcar irregular",
    "cough": "Tos",
    "high_fever": "Fiebre alta",
    "sunken_eyes": "Ojos hundidos",
    "breathlessness": "Falta de aliento",
    "sweating": "Sudoración",
    "dehydration": "Deshidratación",
    "indigestion": "Indigestión",
    "headache": "Dolor de cabeza",
    "yellowish_skin": "Piel amarillenta",
    "dark_urine": "Orina oscura",
    "nausea": "Náuseas",
    "loss_of_appetite": "Pérdida de apetito",
    "pain_behind_the_eyes": "Dolor detrás de los ojos",
    "back_pain": "Dolor de espalda",
    "constipation": "Estreñimiento",
    "abdominal_pain": "Dolor abdominal",
    "diarrhoea": "Diarrea",
    "mild_fever": "Fiebre leve",
    "yellow_urine": "Orina amarilla",
    "yellowing_of_eyes": "Coloración amarilla de los ojos",
    "acute_liver_failure": "Fallo hepático agudo",
    "fluid_overload": "Sobrecarga de líquidos",
    "swelling_of_stomach": "Hinchazón del estómago",
    "swelled_lymph_nodes": "Ganglios linfáticos inflamados",
    "malaise": "Malestar general",
    "blurred_and_distorted_vision": "Visión borrosa y distorsionada",
    "phlegm": "Flema",
    "throat_irritation": "Irritación de garganta",
    "redness_of_eyes": "Enrojecimiento de los ojos",
    "sinus_pressure": "Presión sinusal",
    "runny_nose": "Secreción nasal",
    "congestion": "Congestión",
    "chest_pain": "Dolor de pecho",
    "weakness_in_limbs": "Debilidad en las extremidades",
    "fast_heart_rate": "Ritmo cardíaco rápido",
    "pain_during_bowel_movements": "Dolor al evacuar",
    "pain_in_anal_region": "Dolor en la región anal",
    "bloody_stool": "Heces con sangre",
    "irritation_in_anus": "Irritación en el ano",
    "neck_pain": "Dolor de cuello",
    "dizziness": "Mareo",
    "cramps": "Calambres",
    "bruising": "Moretones",
    "obesity": "Obesidad",
    "swollen_legs": "Piernas hinchadas",
    "swollen_blood_vessels": "Vasos sanguíneos hinchados",
    "puffy_face_and_eyes": "Cara y ojos hinchados",
    "enlarged_thyroid": "Tiroides agrandada",
    "brittle_nails": "Uñas quebradizas",
    "swollen_extremeties": "Extremidades hinchadas",
    "excessive_hunger": "Hambre excesiva",
    "extra_marital_contacts": "Contactos extramaritales (asociado a ITS)",
    "drying_and_tingling_lips": "Labios secos y con hormigueo",
    "slurred_speech": "Dificultad para hablar (habla arrastrada)",
    "knee_pain": "Dolor de rodilla",
    "hip_joint_pain": "Dolor en la articulación de la cadera",
    "muscle_weakness": "Debilidad muscular",
    "stiff_neck": "Rigidez en el cuello",
    "swelling_joints": "Hinchazón de articulaciones",
    "movement_stiffness": "Rigidez al moverse",
    "spinning_movements": "Movimientos de giro (vértigo)",
    "loss_of_balance": "Pérdida de equilibrio",
    "unsteadiness": "Inestabilidad",
    "weakness_of_one_body_side": "Debilidad de un lado del cuerpo",
    "loss_of_smell": "Pérdida del olfato",
    "bladder_discomfort": "Molestia en la vejiga",
    "foul_smell_ofurine": "Mal olor de la orina",
    "continuous_feel_of_urine": "Sensación continua de orinar",
    "passage_of_gases": "Paso de gases",
    "internal_itching": "Picazón interna",
    "toxic_look_(typhos)": "Aspecto tóxico (similar al tifus)",
    "depression": "Depresión",
    "irritability": "Irritabilidad",
    "muscle_pain": "Dolor muscular",
    "altered_sensorium": "Sensorio alterado (confusión)",
    "red_spots_over_body": "Manchas rojas en el cuerpo",
    "belly_pain": "Dolor de vientre",
    "abnormal_menstruation": "Menstruación anormal",
    "dischromic_patches": "Parches discrómicos (decoloración de la piel)",
    "watering_from_eyes": "Lagrimeo de los ojos",
    "increased_appetite": "Aumento del apetito",
    "polyuria": "Poliuria (micción excesiva)",
    "family_history": "Antecedentes familiares",
    "mucoid_sputum": "Esputo mucoide",
    "rusty_sputum": "Esputo herrumbroso (con sangre vieja)",
    "lack_of_concentration": "Falta de concentración",
    "visual_disturbances": "Alteraciones visuales",
    "receiving_blood_transfusion": "Recibir transfusión de sangre",
    "receiving_unsterile_injections": "Recibir inyecciones no estériles",
    "coma": "Coma",
    "stomach_bleeding": "Sangrado estomacal",
    "distention_of_abdomen": "Distensión del abdomen",
    "history_of_alcohol_consumption": "Historial de consumo de alcohol",
    "blood_in_sputum": "Sangre en el esputo",
    "prominent_veins_on_calf": "Venas prominentes en la pantorrilla",
    "palpitations": "Palpitaciones",
    "painful_walking": "Caminar doloroso",
    "pus_filled_pimples": "Granos llenos de pus",
    "blackheads": "Puntos negros",
    "scurring": "Cicatrización (marcas en la piel)",
    "skin_peeling": "Descamación de la piel",
    "silver_like_dusting": "Escamación plateada (como caspa)",
    "small_dents_in_nails": "Pequeñas abolladuras en las uñas",
    "inflammatory_nails": "Uñas inflamadas",
    "blister": "Ampolla",
    "red_sore_around_nose": "Llaga roja alrededor de la nariz",
    "yellow_crust_ooze": "Costra amarilla supurante",
    "prognosis": "Pronóstico (término médico)"
}

# inicializar la base de datos y la conexión
CONN = db_manager.initialize_database()

PROJECT_NAME = "HidroDia AI"

# Cargar archivos auxiliares para descripciones y precauciones
def load_auxiliary_data():
    """Carga descripciones y precauciones para mostrar en la UI."""
    data_path = 'data/'
    
    desc_file = os.path.join(data_path, 'symptom_Description.csv')
    prec_file = os.path.join(data_path, 'symptom_precaution.csv')
    
    desc_df = pd.read_csv(desc_file, index_col='Disease')
    prec_df = pd.read_csv(prec_file, index_col='Disease')
    
    # normalizar los índices (nombres de enfermedades) a minusculas
    desc_df.index = desc_df.index.str.strip().str.lower()
    prec_df.index = prec_df.index.str.strip().str.lower()
    
    return desc_df, prec_df

# cargar datos auxiliares una vez
try:
    DESC_DF, PREC_DF = load_auxiliary_data()
except FileNotFoundError:
    st.error("Error al cargar archivos auxiliares (symptom_Description.csv o symptom_precaution.csv) ver carpeta data")
    DESC_DF = pd.DataFrame()
    PREC_DF = pd.DataFrame()

st.title(f"{PROJECT_NAME}")

tab1, tab2 = st.tabs(["Predicciones y alertas", "Historial de sintomas"])


# APARTADO 1: Predicciones y alertas
with tab1:
    st.header("Diagnóstico Rápido de Síntomas")
    
    st.info("**Instrucciones:** Ingresar síntomas separados por comas, ej: vomiting, joint pain (para mayor precision revise los sintomas del diccionario)")
    
    st.sidebar.title("Diccionario")

    # DataFrame para la tabla TRADUCTION_MAP 
    translated_data = [
        {"Espaniol": es_sym, "Ingles": en_sym}
        for en_sym, es_sym in TRADUCTION_MAP.items()
    ]
    df_reference = pd.DataFrame(translated_data)

    st.sidebar.info("Para la prediccion, use los sintomas en **'Ingles'**.")
    st.sidebar.dataframe(df_reference, use_container_width=True, hide_index=True)

    st.sidebar.markdown("---")
    st.sidebar.header("HidroDia AI")
    st.sidebar.markdown(f"**{PROJECT_NAME}** sistema de apoyo al diagnostico usando Deep Learning y reglas simbólicas")

    # area de entrada del usuario
    symptoms_input = st.text_area(
        "Ingresar sintomas:",
        key="symptoms_key",
        placeholder="Ej: high fever, chills, joint pain, headache"
    )

    if st.button("Analizar sintomas y predecir", type="primary"):
        if not symptoms_input:
            st.warning("Por favor, ingresa al menos un sintoma para comenzar a analizar")
        else:
            with st.spinner("Analizando con el modelo de IA"):
                # llamada a la lógica del módulo de utilidad
                top_predictions, hydro_alert, alert_level = get_prediction(symptoms_input)
                        
            st.subheader("Resultados del Análisis")

            # modulo alerta hidrocefalia
            if alert_level == "danger":
                st.error(f"**ALERTA ESPECIALIZADA:** {hydro_alert}")
            elif alert_level == "warning":
                st.warning(f"**ALERTA ESPECIALIZADA:** {hydro_alert}")
            else:
                st.success(f"**Alerta Especializada:** {hydro_alert}")
            
            st.markdown("---")
            
            # PREDICCIÓN DEL MODELO ML/DL
            
            if top_predictions:
                st.subheader("Clasificacion de posibles esnfermedades")
                
                # columnas de top 5 
                cols = st.columns(5)
                
                prediction_text_for_db = ""
                
                for i, (disease, probability) in enumerate(top_predictions):
                    # formato para guardar en la BD
                    prediction_text_for_db += f"{disease}: {probability:.2f}%; "
                    
                    # formato para la interfaz
                    with cols[i]:
                        st.metric(label=f"Posibilidad #{i+1}", value=f"{disease}", delta=f"{probability:.1f}%")

                        # mostrar descripción y precauciones si hay  
                        if disease.lower() in DESC_DF.index:
                            st.write(f"**Descripcion:** {DESC_DF.loc[disease.lower(), 'Description']}")
                            
                        if disease.lower() in PREC_DF.index:
                            precautions = PREC_DF.loc[disease.lower()].dropna().tolist()
                            if precautions:
                                st.markdown("**Precauciones recomendadas:**")
                                st.markdown("".join([f"* {p}\n" for p in precautions]))
                            
                st.markdown("---")
                
                # GUARDAR EN EL HISTORIAL
                st.markdown("##### Registrar en el historial")

                try:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # llamamos a get_prediction para obtener la lista de síntomas detectados si lo requiere
                    
                    entry_data = (
                        current_time,
                        symptoms_input,
                        "Vector de SIntomas Omitido (para simplificar UI)", # aqui el X_predict serializado si fuera necesario
                        prediction_text_for_db.strip(),
                        hydro_alert
                    )
                    
                    db_manager.add_entry(CONN, entry_data)
                    st.success("**Registro Guardado:** Consulta agregada al historial")
                    
                    st.warning("**IMPORTANTE:** Este es un sistema de apoyo, consulte a un profesional para un diagnóstico seguro")
                    
                except sqlite3.Error as e:
                    st.error(f"Error al guardar en la base de datos: {e}")
            else:
                 st.info("No se pudo obtener una predicción valida, revise los sintomas ingresados")


# APARTADO 2: HISTORIAL

with tab2:
    st.header("Historial y tendencias")
    
    if st.button("Actualizar Historial", key="refresh_diary"):
        st.cache_data.clear() # Limpiar cache Y forzar la recarga
        
    st.subheader("Entradas Recientes al historial")
    
    try:
        # recuperar todas las entradas
        df_diary = db_manager.get_all_entries(CONN)
        
        if df_diary.empty:
            st.info("Diario vacio - realiza una predicción para agregar la primera entrada")
        else:
            # Mostrar tabla de historial
            # st.dataframe(df_diary, use_container_width=True)
            # 1. Renombrar las columnas a espaniol ayuda auto traducir encabezados
            df_display = df_diary.rename(columns={
                'fecha_registro': 'Fecha de Registro',
                'sintomas_texto': 'Sintomas reportados',
                'prediccion_modelo': 'Prediccion del modelo',
                'alerta_hidro': 'Alerta especializada'
            })
            
            # 2. Convertir el DataFrame a un formato de visualización más simple (st.table)
            # st.table() usa menos optimización de JS que st.dataframe(), haciendo el texto más accesible al navegador
            st.table(df_display)
            
            st.markdown("---")
            st.subheader("Frecuencia de sintomas y alertas")
            
            st.markdown("---")
            st.subheader("Análisis de Tendencias (Frecuencia de Síntomas y Alertas)")
            
            #  1: Visualizar Alertas de Hidrocefalia
            alert_counts = df_diary['alerta_hidro'].value_counts()
            st.markdown("##### Frecuencia de alertas especializadas")
            st.bar_chart(alert_counts)
            
            #  2: analisis de frecuencia de síntomas NLP basico en el historial
            # contando palabras clave
            
            # concatenar todos los textos de sintomas y convertirlos a minuscu;as
            all_symptoms_text = ' '.join(df_diary['sintomas_texto'].astype(str).str.lower()).replace(',', ' ').split()
            
            # contar frecuencia de los 10 sintomas más comunes
            symptom_freq = pd.Series(all_symptoms_text).value_counts().head(10)
            
            st.markdown("##### Sintomas frecuentes reportados")
            st.bar_chart(symptom_freq)
            
    except sqlite3.Error:
        st.error("No se pudo conectar a la base de datos para recuperar el historial")


# cerrar la conexión cuando la app de streamlit finaliza la sesión
if CONN:
    CONN.close()