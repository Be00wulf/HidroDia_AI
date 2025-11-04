import numpy as np
import os
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_prep import prepare_data  

# rutas
MODEL_DIR = 'models/'
MODEL_FILE = os.path.join(MODEL_DIR, 'disease_predictor_model.keras')

def build_model(input_shape, num_classes):
    """Define y compila el modelo de Red Neuronal Densa (DNN)."""
    
    # usando el modelo Sequential de Keras
    model = Sequential([
        # Primera Capa Densa: Alto número de neuronas para aprender patrones complejos
        Dense(128, activation='relu', input_shape=(input_shape,)),
        # Dropout: Técnica de regularización para prevenir el overfitting (sobreajuste)
        Dropout(0.3),
        
        # Segunda Capa Densa
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Capa de Salida:
        # 'num_classes' es el número total de enfermedades.
        # 'softmax' es la función de activación obligatoria para la clasificación multiclase,
        # produce probabilidades para cada clase que suman 1.
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilación del modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # Pérdida estándar para One-Hot Encoded
        metrics=['accuracy']
    )
    
    return model

def train_and_save_model():
    """Carga los datos, entrena el modelo y lo guarda."""
    
    print("\nIniciando entrenamiento del modelo Keras...")
    
    # 1. CARGAMOS datos preprocesados
    X_train, X_test, Y_train, Y_test, le, severity_map = prepare_data()
    
    if X_train is None:
        print("\nError: no se cargaron datos hay que revisar data_prep.py")
        return
    
    input_dim = X_train.shape[1]
    num_classes = Y_train.shape[1]
    
    # 2. se construye el modelo
    model = build_model(input_dim, num_classes)
    model.summary() # muestra la arquitectura del modelo
    
    # 3. se definen callbacks para mejorar el entrenamiento
    # EarlyStopping: Detiene el entrenamiento si el modelo deja de mejorar en el conjunto de prueba (val_loss)
    # ModelCheckpoint: Guarda el mejor modelo que se haya visto hasta el momento
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_FILE, monitor='val_loss', save_best_only=True)
    ]
    
    # 4. se entrena el modelo
    history = model.fit(
        X_train, Y_train,
        epochs=100, # un número alto, EarlyStopping lo detendrá
        batch_size=32,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. evaluación final y guardado del modelo - el checkpoint ya lo guarda - aqui se evalua
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\nEntrenamiento Finalizado")
    print(f"Precisión en datos de prueba: {accuracy*100:.2f}%")
    
    print(f"Modelo guardado en: {MODEL_FILE}")

if __name__ == "__main__":
    train_and_save_model()