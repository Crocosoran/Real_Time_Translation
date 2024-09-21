'''
Imports
'''

import tensorflow as tf
from src.data_processing import head_cross_attention_scores
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.family'] = ['Heiti TC']

'''
Web App Development
'''
def translate_text_tf_lite(input_text,source_vectorisation, target_vectorisation, interpreter):
    '''
    Arguments:
        input_text: Input text provided by the user in English!
        source_vectorisation: source language vocabulary
        target_vectorisation: target language vocabulary
        interpreter: loaded, trained transformer model

    Description:
        Perform inference (translation) one token at a time. When a token is translated it is added to the
        translated_sentence_list, alongside it's attention scores added to the attention_scores_list (used later
        for visualisation)
    '''
    # Interpreter set up
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Encoder/ Decoder data set up
    encoder_input = source_vectorisation([f'{input_text}'])
    sentence_max_length = encoder_input.shape[1]
    decoder_output = target_vectorisation(['start'])[:, :-1]
    end_token = target_vectorisation(['end'])[:, 0]
    attention_scores_list = []
    translated_sentence_list = []

    for i in tf.range(sentence_max_length - 1):
        interpreter.set_tensor(input_details[0]['index'], encoder_input)
        interpreter.set_tensor(input_details[1]['index'], decoder_output)
        interpreter.invoke()
        attention_scores_list.append(interpreter.get_tensor(output_details[0]['index']))
        predictions = interpreter.get_tensor(output_details[1]['index'])
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
def show_predict_page(source_vectorisation, target_vectorisation, predict_fn):
    '''
    Arguments:
        source_vectorisation: source language vocabulary
        target_vectorisation: target language vocabulary
        predict_fn: loaded, trained transformer model

    Description:
        Renders the web page with all of its functionalities, as well as the attention score plots.
    '''
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
        st.session_state.translated_sentence_list, st.session_state.attention_scores_list = translate_text_tf_lite(input_text,source_vectorisation, target_vectorisation, predict_fn)

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
        fig.subplots_adjust(hspace=0.4)
        for h in range(attention_score.shape[1]):
            ax = fig.add_subplot(2, 4, h + 1)
            head = attention_score[0, h, :, :]
            im = head_cross_attention_scores(input_text, translated_sentence, head)
            ax.set_xlabel(f'Head {h + 1}')

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        st.pyplot(fig)
