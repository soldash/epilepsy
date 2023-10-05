from joblib import load
import pandas as pd
import numpy as np
import json

def model_fn(model_dir):
    """Carga el modelo del directorio."""
    model = load("{}/decision_tree_model.pkl".format(model_dir))
    
    # Capa de compatibilidad para el problema de versiones con sklearn
    if not hasattr(model, 'n_features_') and hasattr(model, 'n_features_in_'):
        model.n_features_ = model.n_features_in_
    
    return model

def predict_fn(input_data, model):
    """Realiza una predicción."""
    
    # Verifica si input_data es bytes, lo que significa que podría ser una cadena JSON
    if isinstance(input_data, bytes):
        data_dict = json.loads(input_data.decode('utf-8'))  # Decodificar bytes a str antes de deserializar
        data_df = pd.DataFrame([data_dict])
    elif isinstance(input_data, np.ndarray):
        data_df = pd.DataFrame(input_data)
    else:
        raise ValueError("Tipo de entrada no soportado: {}".format(type(input_data)))
    
    print(model)  # Imprimir el objeto del modelo para diagnosticar
    prediction = model.predict(data_df)
    
    return prediction.tolist()
