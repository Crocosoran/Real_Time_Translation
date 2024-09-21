'''
Imports
'''

import sys
import os
import streamlit as st
import tensorflow as tf

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import custom_stadardization_fn
from predict import show_predict_page


# Initiate Model as soon as the App is started + Cache Model

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    src_dir = os.path.join(project_root, "src")
    source_path = os.path.join(src_dir, "source_vectorisation")
    target_path = os.path.join(src_dir, "target_vectorisation")
    model_path = os.path.join(project_root, "saved_models/quantized_model/quantized_transformer.tflite")

    source_vectorisation = tf.keras.models.load_model(source_path)
    source_vectorisation = source_vectorisation.layers[0]

    target_vectorisation = tf.keras.models.load_model(target_path, custom_objects={
        "custom_stadardization_fn": custom_stadardization_fn
    })
    target_vectorisation = target_vectorisation.layers[0]

    predict_fn = tf.lite.Interpreter(
        model_path=model_path)

    return source_vectorisation, target_vectorisation, predict_fn


source_vectorisation, target_vectorisation, predict_fn = load_model()

if __name__ == "__main__":
    show_predict_page(source_vectorisation, target_vectorisation, predict_fn)
