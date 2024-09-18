'''
Imports
'''

import tensorflow as tf
import os
from src.data_processing import custom_stadardization_fn,head_cross_attention_scores
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Heiti TC']


'''
Web App Development
'''
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
source_path = os.path.join(src_dir, "source_vectorisation")
target_path = os.path.join(src_dir, "target_vectorisation")

# filepath = "/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation/src"
# source_path = os.path.join(filepath, "source_vectorisation")
# target_path = os.path.join(filepath, "target_vectorisation")

source_vectorisation = tf.keras.models.load_model(source_path)
source_vectorisation = source_vectorisation.layers[0]

target_vectorisation = tf.keras.models.load_model(target_path, custom_objects={
    "custom_stadardization_fn": custom_stadardization_fn
})
target_vectorisation = target_vectorisation.layers[0]

model_weights = os.path.join(project_root, "saved_models")
# reloaded = tf.saved_model.load('/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation/saved_models')
reloaded = tf.saved_model.load(model_weights)
predict_fn = reloaded.signatures['serving_default']

def translate_text(input_text):
    encoder_input = source_vectorisation([f'{input_text}'])
    sentence_max_length = encoder_input.shape[1]
    decoder_output = target_vectorisation(['start'])[:, :-1]
    end_token = target_vectorisation(['end'])[:, 0]
    attention_scores_list = []
    translated_sentence_list = []
    for i in tf.range(sentence_max_length - 1):
        predictions = predict_fn(source_inputs=encoder_input, target_inputs=decoder_output)
        attention_scores_list.append(predictions['last_attention_score'])
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
        translated_sentence_list.append(translated_sentence)

    return translated_sentence_list, attention_scores_list

'''
Model Inference: Translation
'''

def show_predict_page():
    st.title('English to Chinese Translator')
    # write text
    # markdown syntax: ### will be h3 (font size)
    input_text = st.text_input('English text', '', placeholder='I like to eat cake')
    button = st.button('Translate!')  # if click = True, if not click = False

    # Initialize session state for translated_sentence and attention_scores
    if 'translated_sentence_list' not in st.session_state:
        st.session_state.translated_sentence_list = []
        st.session_state.attention_scores_list = []

    if button:
        # Perform translation and store in session state
        st.session_state.translated_sentence_list, st.session_state.attention_scores_list = translate_text(input_text)

    # Display translation result
    if st.session_state.translated_sentence_list:
        st.subheader(f'Translated sentence: {st.session_state.translated_sentence_list[-1]}')

        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.subheader("Attention Scores for Heads")
        st.write('')
        # Slider to explore attention scores
        tokens_translated = st.slider("Attention Scores per Translated Token", 1,
                                      len(st.session_state.translated_sentence_list[-1]), 1)
        attention_score = st.session_state.attention_scores_list[tokens_translated-1]
        translated_sentence = st.session_state.translated_sentence_list[tokens_translated - 1]
        # Show attention plot
        fig = plt.figure(figsize=(16, 8))
        for h in range(attention_score.shape[1]):
            ax = fig.add_subplot(2, 4, h + 1)
            head = attention_score.numpy()[0, h, :, :]
            head_cross_attention_scores(input_text, translated_sentence, head)
            ax.set_xlabel(f'Head {h + 1}')

        st.pyplot(fig)





