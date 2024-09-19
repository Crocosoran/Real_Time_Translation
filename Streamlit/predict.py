'''
Imports
'''

import tensorflow as tf
from src.data_processing import head_cross_attention_scores
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
# import plotly.figure_factory as ff
# from plotly.subplots import make_subplots
# import plotly.graph_objs as go




matplotlib.rcParams['font.family'] = ['Heiti TC']

'''
Web App Development
'''
def translate_text(input_text,source_vectorisation, target_vectorisation, predict_fn):
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

def translate_text_tf_lite(input_text,source_vectorisation, target_vectorisation, interpreter):
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
            head = attention_score[0, h, :, :] # need .numpy() when not using translate_text_tf_lite
            im = head_cross_attention_scores(input_text, translated_sentence, head)
            ax.set_xlabel(f'Head {h + 1}')

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        st.pyplot(fig)

'''
Using Plotly: Under Development
'''
# def show_predict_page_2():
#     st.title('English to Chinese Translator')
#     # write text
#     # markdown syntax: ### will be h3 (font size)
#     input_text = st.text_input('English text', '', placeholder='I like to eat cake')
#     button = st.button('Translate!')  # if click = True, if not click = False
#
#     # Initialize session state for translated_sentence and attention_scores
#     if 'translated_sentence_list' not in st.session_state:
#         st.session_state.translated_sentence_list = []
#         st.session_state.attention_scores_list = []
#
#     if button:
#         # Perform translation and store in session state
#         st.session_state.translated_sentence_list, st.session_state.attention_scores_list = translate_text(input_text)
#
#     # Display translation result
#     if st.session_state.translated_sentence_list:
#         st.subheader(f'Translated sentence: {st.session_state.translated_sentence_list[-1]}')
#
#         st.write('')
#         st.write('')
#         st.write('')
#         st.write('')
#         st.subheader("Attention Scores for Heads")
#         st.write('')
#         # Slider to explore attention scores
#         tokens_translated = st.slider("Attention Scores per Translated Token", 1,
#                                       len(st.session_state.translated_sentence_list[-1]), 1)
#         attention_score = st.session_state.attention_scores_list[tokens_translated - 1]
#         translated_sentence = st.session_state.translated_sentence_list[tokens_translated - 1]
#         # Show attention plot
#
#         subplot_titles = [f'Head {h + 1}' for h in range(attention_score.shape[1])]
#         fig = make_subplots(
#             rows=2,
#             cols=4,
#             subplot_titles=subplot_titles,
#             shared_xaxes=False,
#             shared_yaxes=False,
#             vertical_spacing=0.1,
#             horizontal_spacing=0.05
#         )
#         input_labels = input_text.split()
#         translated_labels = [translated_token for translated_token in translated_sentence]
#
#         fig.update_layout(
#             height=1000,
#             width=1000,
#             annotations= [],
#             showlegend = False
#         )
#
#         annotations = [
#             dict(x=0.105, y= 0.515, xref='paper', yref='paper', showarrow=False, text="Title 1"),
#             dict(x=0.375, y=0.515, xref='paper', yref='paper', showarrow=False, text="Title 2"), # 170
#             dict(x=0.640, y=0.515, xref='paper', yref='paper', showarrow=False, text="Title 3"),
#             dict(x=0.875, y=0.515, xref='paper', yref='paper', showarrow=False, text="Title 4"),
#             dict(x=0.105, y=-0.035, xref='paper', yref='paper', showarrow=False, text="Title 5"),
#             dict(x=0.375, y=-0.035, xref='paper', yref='paper', showarrow=False, text="Title 6"),
#             dict(x=0.625, y=-0.035, xref='paper', yref='paper', showarrow=False, text="Title 7"),
#             dict(x=0.875, y=-0.035, xref='paper', yref='paper', showarrow=False, text="Title 8")
#         ]
#
#         fig.update_layout(annotations= annotations)
#         for h in range(attention_score.shape[1]):
#             head = attention_score.numpy()[0, h, :len(translated_sentence), :len(input_text.split())]
#             heatmap = go.Heatmap(
#                 z=head,
#                 colorscale='Viridis',
#                 x = input_labels,
#                 y = translated_labels,
#                 colorbar={
#                     "ticks":'outside',
#                     "ticktext":["0","0.25","0.5","0.75","1"],
#                     "len":0.5
#
#                 }
#             )
#             fig.add_trace(
#                 heatmap,
#                 row=h // 4 + 1,
#                 col=h % 4 + 1
#             )
#             fig.update_xaxes(side='top',row=h // 4 + 1, col=h % 4 + 1)
#             fig.update_yaxes(autorange='reversed', row=h // 4 + 1, col=h % 4 + 1)
#
#         st.plotly_chart(fig)
