'''
Imports
'''

import tensorflow as tf

'''
Implement TensorFlow Lite
'''

reloaded = tf.saved_model.load('/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation'
                               '/saved_models')

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

with open('/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation/saved_models/quantized_model/quantized_transformer.tflite', 'wb') as f:
    f.write(tflite_model)
