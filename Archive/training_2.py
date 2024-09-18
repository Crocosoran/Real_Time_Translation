'''
Imports

'''
import tensorflow as tf
from tensorflow.keras import layers
import keras
import numpy as np
import tensorflow as tf
from model import build_transformer_model
from data_processing import data_load, split_data, vectorise_data
from model_2 import PositionalEmbedding_sinusoids,BaseAttention, CausalSelfAttention,\
    GlobalSelfAttention, CrossAttention, Encoder, Decoder, Transformer,CustomSchedule, masked_accuracy,masked_loss
import random
import string
import jieba
import tensorflow as tf
from typing import List
import re
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = ['Heiti TC']

'''
Load and preprocess data
'''
text_pairs = data_load("/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation/data/cmn-eng/cmn.txt")
train_pairs, val_pairs, test_pairs = split_data(text_pairs)
source_vectorisation, target_vectorisation, en, cn = vectorise_data(train_pairs)
_, _, en_val, cn_val = vectorise_data(val_pairs)
_, _, en_test, cn_test = vectorise_data(test_pairs)

def create_datasets(source_language: str, target_language: str, batch_size: int = 64):
    def format_dataset(eng, ch):
        eng = source_vectorisation(eng)
        ch = target_vectorisation(ch)
        return (eng, ch[:, :-1]), ch[:, 1:]
        # return ({
        #             "english": eng,
        #             "chinese": ch[:, :-1]}, ch[:, 1:])  # (eng, ch [same as eng length]), ch [one token ahead]

    eng_texts = list(source_language)
    ch_texts = list(target_language)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, ch_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()  # Use in-memory caching to speed up preprocessing.


train_ds = create_datasets(source_language=en, target_language=cn)
val_ds = create_datasets(source_language=en_val, target_language=cn_val)
test_ds = create_datasets(source_language=en_test, target_language=cn_test)

for (en,cn), cn_label in train_ds.take(1):
    break

transformer = Transformer(
    source_vocab_size=len(source_vectorisation.get_vocabulary()),
    target_vocab_size=len(target_vectorisation.get_vocabulary()),
    seq_len=30,
    embed_dim=128,
    dense_output_dim=512,
    num_heads=8,
    num_layers=4,
    dropout_rate=0.1
)

learning_rate = CustomSchedule(embed_dim=128)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# transformer.compile(
#     loss="sparse_categorical_crossentropy",
#     optimizer='adam',
#     metrics=['accuracy']
# )
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)
train_out = transformer.fit(
    train_ds,
    epochs=1,
    validation_data=val_ds
)
stop
tf.saved_model.save(transformer, export_dir='../saved_models')
reloaded = tf.saved_model.load('../saved_models')
inference_func = reloaded.signatures['serving_default']



loaded_model_keras = tf.keras.models.load_model('../src/saved_models/keras_transformer', custom_objects={'CustomSchedule':CustomSchedule, 'masked_loss':masked_loss, 'masked_accuracy':masked_accuracy})


loaded_model_keras.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)
loaded_model_keras.fit(
    train_ds,
    epochs=1,
    validation_data=val_ds
)

en_sentence = 'I want to eat cake'
encoder_input = source_vectorisation([f'{en_sentence}'])
sentence_max_length = encoder_input.shape[1]
decoder_output = target_vectorisation(['start'])[:, :-1]
end_token = target_vectorisation(['end'])[:, 0]

for i in tf.range(sentence_max_length - 1):
    predictions = reloaded([encoder_input, decoder_output], training=False)
    predicted_id = tf.argmax(predictions[:, i, :], axis=-1)
    if predicted_id == end_token:
        break
    decoder_output = tf.tensor_scatter_nd_add(decoder_output, [[0, i + 1]], predicted_id)

output_tokens = decoder_output.numpy()[0]
ch_vocab = target_vectorisation.get_vocabulary()
ch_index_lookup = dict(zip(range(len(ch_vocab)), ch_vocab))
translated_sentence = ''.join(
    [ch_index_lookup[output_tokens[i]] for i in range(1, len(output_tokens)) if output_tokens[i] != 0])


sample_encoder = Encoder(num_layers=4,
                         embed_dim=512,
                         num_heads=8,
                         dense_output_dim=2048,
                         vocab_size=20000,
                         sequence_length=30
                         )


sample_encoder_output = sample_encoder(iter(train_ds).next()[0][0], training=False)

print(sample_encoder_output.shape)

sample_decoder = Decoder(seq_len=30,
                         vocab_size=20000,
                         embed_dim=512,
                         dense_output_dim=0,
                         num_heads=8,
                         num_layers=4)

output = sample_decoder(x=iter(train_ds).next()[0][1], encoder_output=sample_encoder_output)

output.shape