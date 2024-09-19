'''
Imports
'''

from flask import Flask, request, render_template  # looks into the templates folder and feteches the index
import tensorflow as tf
from src.data_processing import custom_stadardization_fn
import os

'''
App Development
'''

app = Flask(__name__,template_folder='/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation/Flask/templates')

filepath = "/src"
source_path = os.path.join(filepath, "source_vectorisation")
target_path = os.path.join(filepath, "target_vectorisation")

source_vectorisation = tf.keras.models.load_model(source_path)
source_vectorisation = source_vectorisation.layers[0]

target_vectorisation = tf.keras.models.load_model(target_path, custom_objects={
    "custom_stadardization_fn": custom_stadardization_fn
})
target_vectorisation = target_vectorisation.layers[0]

reloaded = tf.saved_model.load('/saved_models')
predict_fn = reloaded.signatures['serving_default']

# Define the route
# The decorator below links the relative route of the URL to the function it is decorating
# Running the app sends us to the index.html
# Note that render_template looks for the file in the templates folder
# use the route() decorator to tell Flask what URL should trigger our function!
@app.route('/')
def home():
    return render_template('index.html')

# You can use the methods argument of the route() decorator to handle different HTML requests.
# GET: A GET message is send, and the server returns data.
# POST: Used to send HTML form data to the server. User enters something on the webpage it posts it to the server.
# Add Post method to the decorator to allow for form submission.
# Redirect to / predict page with the output
@app.route('/predict', methods=['POST'])
def Translator():
    encoder_input = source_vectorisation([f'{request.form["sentence_in_english"]}']) # fetches the values from the
    # browser -> gets what the POST request is/ what the USER has provided

    sentence_max_length = encoder_input.shape[1]
    decoder_output = target_vectorisation(['start'])[:, :-1]
    end_token = target_vectorisation(['end'])[:, 0]

    for i in tf.range(sentence_max_length - 1):
        predictions = predict_fn(source_inputs=encoder_input, target_inputs=decoder_output)
        predictions = predictions['outputs']
        predicted_id = tf.argmax(predictions[:, i, :], axis=-1)
        if predicted_id == end_token:
            break
        decoder_output = tf.tensor_scatter_nd_update(decoder_output, [[0, i + 1]], predicted_id)

    output_tokens = decoder_output.numpy()[0]
    ch_vocab = target_vectorisation.get_vocabulary()
    ch_index_lookup = dict(zip(range(len(ch_vocab)), ch_vocab))
    translated_sentence = ''.join(
        [ch_index_lookup[output_tokens[i]] for i in range(1, len(output_tokens)) if output_tokens[i] != 0])

    # sends the output (translated_sentence) back to the index.html and displays the result with prediction_text
    return render_template('index.html', prediction_text=f"Chinese Translation: {translated_sentence}")

# When the Python interpreter reads a source file, it first defines a few special variables
# For now, we care about the __name__ variable.
# If we execute our code in the main program, like in out case,
# it assigns __main__ as the name (__name__).
# So if we want to run out code here, we can check if __name__ == "__main__"
# If we import this file (module) to another file, then __name__ == app
if __name__ == "__main__":
    app.run(debug=True)
