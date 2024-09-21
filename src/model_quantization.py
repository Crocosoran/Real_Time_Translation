'''
Imports
'''

import tensorflow as tf
import os

'''
Implement TensorFlow Lite
'''
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
saved_models_path = os.path.join(project_root, "saved_models")
quantized_save_path = os.path.join(project_root, "saved_models/quantized_model/quantized_transformer.tflite")

reloaded = tf.saved_model.load(saved_models_path)

predict_fn = reloaded.signatures['serving_default']

converter = tf.lite.TFLiteConverter.from_concrete_functions([predict_fn])

# Enable post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops = True

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
]

# Convert the model to TFLite format
tflite_model = converter.convert()

with open(quantized_save_path, 'wb') as f:
    f.write(tflite_model)
