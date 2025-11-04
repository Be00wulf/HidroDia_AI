import sqlite3
import pandas as pd
import os

# nombre del archivo de la base de datos
DB_NAME = 'hidrodia_diary.db'

def create_connection():
    """Crea una conexión a la base de datos SQLite."""
    conn = None
    try:
        # la BD se creará automáticamente si no existe
        conn = sqlite3.connect(DB_NAME)
        return conn
    except sqlite3.Error as e:
        print(f"error al conectar con la base de datos: {e}")
        return None

def create_table(conn):
    """Crea la tabla de registro de síntomas (el Diario)."""
    sql_create_diary_table = """ CREATE TABLE IF NOT EXISTS diary (
                                id integer PRIMARY KEY,
                                fecha_registro text NOT NULL,
                                sintomas_texto text NOT NULL,
                                sintomas_codificados text,
                                prediccion_modelo text,
                                alerta_hidro text
                            ); """
    try:
        c = conn.cursor()
        c.execute(sql_create_diary_table)
        print(f"Tabla 'diary' creada o ya existente en {DB_NAME}.")
    except sqlite3.Error as e:
        print(f"error al crear la tabla: {e}")

def add_entry(conn, entry):
    """Añade una nueva entrada al diario."""
    sql_insert = """ INSERT INTO diary(fecha_registro, sintomas_texto, sintomas_codificados, prediccion_modelo, alerta_hidro)
                     VALUES(?,?,?,?,?) """
    try:
        c = conn.cursor()
        c.execute(sql_insert, entry)
        conn.commit()
        return c.lastrowid
    except sqlite3.Error as e:
        print(f"error al insertar entrada: {e}")
        return None

def get_all_entries(conn):
    """Recupera todas las entradas del diario como un DataFrame de Pandas."""
    try:
        query = "SELECT fecha_registro, sintomas_texto, prediccion_modelo, alerta_hidro FROM diary ORDER BY fecha_registro DESC"
        df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        print(f"error al obtener entradas: {e}")
        return pd.DataFrame()

# inicialización de la BD 
def initialize_database():
    """Función para conectar y asegurar que la tabla exista."""
    conn = create_connection()
    if conn:
        create_table(conn)
    return conn

if __name__ == "__main__":
    # script de prueba para db_manager
    conn = initialize_database()
    if conn:
        print("\nBase de datos inicializada")
        # ceierra la conexión
        conn.close()