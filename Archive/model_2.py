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
# text_pairs = data_load("/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation/data/cmn-eng/cmn.txt")
# train_pairs, val_pairs, test_pairs = split_data(text_pairs)
# source_vectorisation, target_vectorisation, en, cn = vectorise_data(train_pairs)
# _, _, en_val, cn_val = vectorise_data(val_pairs)
# _, _, en_test, cn_test = vectorise_data(test_pairs)

'''
Build dataset pipeline
'''

# def create_datasets(source_language: str, target_language: str, batch_size: int = 64):
#     def format_dataset(eng, ch):
#         eng = source_vectorisation(eng)
#         ch = target_vectorisation(ch)
#         return (eng, ch[:, :-1]), ch[:, 1:]
#         # return ({
#         #             "english": eng,
#         #             "chinese": ch[:, :-1]}, ch[:, 1:])  # (eng, ch [same as eng length]), ch [one token ahead]
#
#     eng_texts = list(source_language)
#     ch_texts = list(target_language)
#     dataset = tf.data.Dataset.from_tensor_slices((eng_texts, ch_texts))
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.map(format_dataset, num_parallel_calls=4)
#     return dataset.shuffle(2048).prefetch(16).cache()  # Use in-memory caching to speed up preprocessing.


# train_ds = create_datasets(source_language=en, target_language=cn)
# val_ds = create_datasets(source_language=en_val, target_language=cn_val)
# test_ds = create_datasets(source_language=en_test, target_language=cn_test)
#
# for (en, cn), cn_l in train_ds.take(1):
#     break
# for (en, cn), cn_l in val_ds.take(1):
#     break


'''
Positional Encoding
'''


class PositionalEmbedding_sinusoids(layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.word_embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        position_embedding_matrix = self.get_position_encoding(seq_len=seq_len, embed_dim=embed_dim)

        self.position_embedding_layer = layers.Embedding(
            input_dim=seq_len, output_dim=embed_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    # The sinusoidal encoding ensures that each position in the sequence is mapped to a unique combination of
    # sinusoidal values, which the Transformer model can use to determine the relative positions of tokens.
    # The smooth variation across embedding dimensions helps the model capture both local and
    # global positional relationships within the sequence.

    # As i increases, the frequency of the sinusoid decreases, meaning that lower dimensions correspond to
    # high-frequency variations (which can capture fine-grained positional differences), while higher dimensions
    # correspond to low-frequency variations (which capture broader positional information).
    def get_position_encoding(self, seq_len, embed_dim,
                              n=10000):  # positional embeddings are applied to the embed_dim !
        P = np.zeros((seq_len, embed_dim))  # length of sequence, embedding dimension e.g., (20, 512)
        for k in range(seq_len):
            for i in np.arange(int(embed_dim / 2)):
                denominator = n ** (2 * i / embed_dim)
                P[k, 2 * i] = np.sin(k / denominator)  # 0, 2, 4 -> sine for even
                P[k, 2 * i + 1] = np.cos(k / denominator)  # 1, 3, 5 -> cosine for odd
        return P

    def call(self, inputs):  # shape = (64, 20)
        length = tf.shape(inputs)[-1]  # 20
        positions = tf.range(start=0, limit=length, delta=1)  # [0, 1, 2 ..., 20]
        embedded_tokens = self.word_embedding_layer(inputs)  # shape  = (64, 20, 256)
        embedded_positions = self.position_embedding_layer(positions)  # shape = (20, 256)
        # add both embedding vectors together via broadcasting
        return embedded_tokens + embedded_positions  # shape = (64, 20, 256)

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    # Serialisation to save the model
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size
        })
        return config


# positional_embedding = PositionalEmbedding_sinusoids(
#             seq_len = 30,
#             vocab_size= 20000,
#             embed_dim= 512
#         )
#
# test = iter(train_ds).next()[0]['chinese']  # shape (batch_size = 64, sequence_length = 30)
# test_en = iter(train_ds).next()[0]['english']
#
# res = positional_embedding(test, training = False)
# res.shape


'''
Transformer Components
'''


class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.attention = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attention_output, attention_scores = self.attention(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )

        # Cache the attention scores for plotting
        self.last_attention_scores = attention_scores

        x = self.add([x, attention_output])  # ensures masks are propagated, because + does not
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attention_output = self.attention(
            query=x,
            key=x,
            value=x
        )
        x = self.add([x, attention_output])  # Residual Connection
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attention_output = self.attention(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True
        )
        x = self.add([x, attention_output])
        x = self.layernorm(x)
        return x


class FeedForward(layers.Layer):
    def __init__(self, embed_dim, dense_output_dim, dropout_rate=0.1):
        super().__init__()
        self.sequence = tf.keras.Sequential(
            [  # why do we put the dense_output_dim first? What is the syntax of Sequential?
                layers.Dense(dense_output_dim, activation='relu'),
                layers.Dense(embed_dim),
                layers.Dropout(dropout_rate)
            ])
        self.add = tf.keras.layers.Add()
        self.layernorm = layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.sequence(x)])
        x = self.layernorm(x)
        return x  # return (batch_size, sequence_size, embed_dim)


class EncoderLayer(layers.Layer):
    def __init__(self, *, embed_dim, num_heads, dense_output_dim, dropout_rate=0.1):
        super().__init__()

        # all of these are for the MultiHeadAttention, as it takes **kwargs
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(embed_dim=embed_dim,
                               dense_output_dim=dense_output_dim)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


'''
Transformer Encoder
'''


class Encoder(layers.Layer):
    def __init__(self, *, sequence_length, num_layers, embed_dim, num_heads, dense_output_dim, vocab_size,
                 dropout_rate=0.1):
        # set up autograd
        super().__init__()

        # set up arguments
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # set up word embeddings and positional embeddings

        self.positional_embedding = PositionalEmbedding_sinusoids(
            seq_len=sequence_length,
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )

        # set up encoder layer with attention and ffn
        self.encoder_layer = [
            EncoderLayer(
                embed_dim=embed_dim,
                dense_output_dim=dense_output_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        # x: (batch, seq_len)
        x = self.positional_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.encoder_layer[i](x)

        return x  # shape (batch_size, seq_len, embed_dim)


# sample_encoder = Encoder(num_layers=4,
#                          embed_dim=512,
#                          num_heads=8,
#                          dense_output_dim=2048,
#                          vocab_size=20000,
#                          sequence_length=30
#                          )
#
# sample_encoder_output = sample_encoder(test, training=False)
# sample_encoder_output = sample_encoder(iter(train_ds).next()[0], training=False)
#
# print(sample_encoder_output.shape)
# pos_embed = PositionalEmbedding_sinusoids(
#     seq_len=30,
#     embed_dim=512,
#     vocab_size=20000
# )
#
# print(pos_embed(test, training=False).shape)
'''
Transformer Decoder
'''


class DecoderLayer(layers.Layer):
    def __init__(self, embed_dim, dense_output_dim, num_heads, dropout_rate=0.1):
        super().__init__()

        self.causal_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(
            embed_dim=embed_dim,
            dense_output_dim=dense_output_dim
        )

    def call(self, x, encoder_output):
        x = self.causal_attention(x)
        x = self.cross_attention(x=x, context=encoder_output)

        self.last_attention_score = self.cross_attention.last_attention_scores
        x = self.ffn(x)  # (batch_size, seq_size, embed_dim)

        return x


class Decoder(layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim, dense_output_dim, num_heads, num_layers,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.dense_output_dim = dense_output_dim

        self.positional_embeddings = PositionalEmbedding_sinusoids(seq_len=seq_len,
                                                                   vocab_size=vocab_size,
                                                                   embed_dim=embed_dim)
        self.dropout = layers.Dropout(dropout_rate)

        self.decoder_layer = [DecoderLayer(embed_dim=embed_dim,
                                           dense_output_dim=dense_output_dim,
                                           num_heads=num_heads)
                              for _ in range(num_layers)]

        self.last_attention_scores = None

    def call(self, x, encoder_output):
        x = self.positional_embeddings(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.decoder_layer[i](x, encoder_output)

        self.last_attention_scores = self.decoder_layer[-1].last_attention_score

        return x  # (batch_size, seq_size, emebd_dim)


# sample_decoder = Decoder(seq_len=30,
#                          vocab_size=20000,
#                          embed_dim=512,
#                          dense_output_dim=0,
#                          num_heads=8,
#                          num_layers=4)
#
# output = sample_decoder(x=test_en, encoder_output=sample_encoder_output)
#
# output.shape

'''
Transformer
'''


class Transformer(tf.keras.Model):
    def __init__(self, source_vocab_size, target_vocab_size, seq_len, embed_dim, dense_output_dim, num_heads,
                 num_layers, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(sequence_length=seq_len,
                               num_layers=num_layers,
                               embed_dim=embed_dim,
                               num_heads=num_heads,
                               dense_output_dim=dense_output_dim,
                               vocab_size=source_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(seq_len=seq_len,
                               vocab_size=target_vocab_size,
                               embed_dim=embed_dim,
                               dense_output_dim=dense_output_dim,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)#, activation='softmax')

    def call(self, inputs):

        source_inputs, target_inputs = inputs

        encoder_output = self.encoder(source_inputs)
        decoder_output = self.decoder(target_inputs, encoder_output)

        logits = self.final_layer(decoder_output)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


# transformer = Transformer(
#     source_vocab_size=len(source_vectorisation.get_vocabulary()),
#     target_vocab_size=len(target_vectorisation.get_vocabulary()),
#     seq_len=30,
#     embed_dim=128,
#     dense_output_dim=512,
#     num_heads=8,
#     num_layers=4,
#     dropout_rate=0.1
# )
#
# res = transformer((iter(train_ds).next()[0][0], iter(train_ds).next()[0][1]))
# len(iter(train_ds).next())
#
# iter(train_ds).next()[0]
# print(res.shape)
# print(iter(train_ds).next()[0]['english'].shape)
# print(iter(train_ds).next()[0]['chinese'].shape)

# attention_scores = transformer.decoder.last_attention_scores
# print(attention_scores.shape)  # (batch_size, num_heads, target_seq_len, source_seq_len)
#
# transformer.summary()

'''
Training
'''


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embed_dim, warmup_steps=4000):
        super().__init__()

        self.embed_dim = embed_dim
        self.embed_dim = tf.cast(self.embed_dim, dtype=tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': int(self.embed_dim),
            'warmup_steps': int(self.warmup_steps)
        })
        return config

    # def get_config(self):
    #     return {
    #         'embed_dim': int(self.embed_dim),
    #         'warmup_steps': int(self.warmup_steps)
    #     }

# learning_rate = CustomSchedule(embed_dim=128)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#
# import matplotlib.pyplot as plt
#
# plt.plot(learning_rate(step=tf.range(40000, dtype=tf.float32)))
# plt.ylabel('Learning Rate')
# plt.xlabel("Train Step")

'''
Loss and Metrics
'''


def masked_loss(label, pred):
    mask = label != 0

    # Expects logits NOT probabilities.
    # Internally applies sfotmax to convert logits to probabilities
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


# transformer.compile(
#     loss=masked_loss,
#     optimizer=optimizer,
#     metrics=[masked_accuracy]
# )
#
# train_out = transformer.fit(
#     train_ds,
#     epochs=1,
#     validation_data=val_ds
# )

'''
Inference
'''


class Translator(tf.Module):
    def __init__(self, transformer, source_vectorisation, target_vectorisation):
        self.transformer = transformer
        self.source_vectorisation = source_vectorisation
        self.target_vectorisation = target_vectorisation

    def __call__(self, en_sentence):

        encoder_input = self.source_vectorisation([f'{en_sentence}'])
        sentence_max_length = encoder_input.shape[1]
        decoder_output = self.target_vectorisation(['start'])[:, :-1]
        end_token = self.target_vectorisation(['end'])[:, 0]

        for i in tf.range(sentence_max_length - 1):
            predictions = self.transformer([encoder_input, decoder_output], training=False)
            predicted_id = tf.argmax(predictions[:, i, :], axis=-1)
            if predicted_id == end_token:
                break
            decoder_output = tf.tensor_scatter_nd_add(decoder_output, [[0, i + 1]], predicted_id)

        output_tokens = decoder_output.numpy()[0]
        ch_vocab = self.target_vectorisation.get_vocabulary()
        ch_index_lookup = dict(zip(range(len(ch_vocab)), ch_vocab))
        translated_sentence = ''.join(
            [ch_index_lookup[output_tokens[i]] for i in range(1, len(output_tokens)) if output_tokens[i] != 0])
        attention_weights = self.transformer.decoder.last_attention_scores

        return translated_sentence, output_tokens, attention_weights


# translator = Translator(transformer=transformer, source_vectorisation=source_vectorisation,
#                         target_vectorisation=target_vectorisation)
# en_sentence = "I like eating cake"
# translated_sentence, output_tokens, attention_weights = translator(en_sentence)


'''
Attention Plots
'''


def plot_attention_scores(source_language_sentence, target_langauge_sentence, attention_weights, number_of_heads):
    fig = plt.figure(figsize=(16, 8))
    source_language_sentence = source_language_sentence.split()
    target_langauge_sentence = [i for i in target_langauge_sentence]
    source_language_sentence_length = len(source_language_sentence)
    target_langauge_sentence_length = len(target_langauge_sentence)
    number_of_heads = attention_weights.shape[1]

    for h in range(number_of_heads):
        ax = fig.add_subplot(2, 4, h + 1)
        ax.imshow(attention_weights[0, h, 0:target_langauge_sentence_length, 0:source_language_sentence_length])
        ax.set_xticks(np.arange(source_language_sentence_length), source_language_sentence, rotation=90)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(target_langauge_sentence_length), target_langauge_sentence)
        ax.set_xlabel(f'Head {h + 1}')
        for c in range(target_langauge_sentence_length):
            for e in range(source_language_sentence_length):
                ax.text(e, c, round(attention_weights[0, h, c, e].numpy(), 3), ha='center', va='center', color='w')
    plt.tight_layout()
    plt.show()


# plot_attention_scores(source_language_sentence=en_sentence, target_langauge_sentence=translated_sentence,
#                       attention_weights=attention_weights, number_of_heads=8)

'''
####################
'''


# class ExportTranslator(tf.Module):
#     def __init__(self, translator):
#         self.translator = translator
#
#     @tf.function(input_signature=[
#         tf.TensorSpec(shape=[], dtype=tf.string)])  # converts a python function into a tensor flow graph
#     def __call__(self, sentence):
#         (result,
#          tokens,
#          attention_weights) = self.translator(sentence)
#
#         return result


#
# translator_wrapped = ExportTranslator(translator)
#
# transformer.save('../saved_models/keras_transformer',save_format='tf')
# tf.saved_model.save(transformer, export_dir='../saved_models/keras_transformer',save_format='h5')
#
# transformer_loaded = tf.saved_model.load('../saved_models/keras_transformer')
# en_sentence = "I like eating cake"
#
# transformer_loaded.signatures["serving_default"]
#
# encoder_input = tf.reshape(encoder_input, (-1, 30))
# decoder_output = tf.reshape(decoder_output, (-1, 30))
#
# inference_func = transformer_loaded.signatures['serving_default']
#
# results = inference_func(input_1=encoder_input, input_2=decoder_output)
#
# print(results)
#
# encoder_input = source_vectorisation([f'{en_sentence}'])
# sentence_max_length = encoder_input.shape[1]
# decoder_output_2 = target_vectorisation(['start'])[:, :-1]
# end_token = target_vectorisation(['end'])[:, 0]
#
# for i in tf.range(sentence_max_length - 1):
#     predictions = inference_func(input_1=encoder_input, input_2=decoder_output_2)['output_1']
#     predicted_id = tf.argmax(predictions[:, i, :], axis=-1)
#     if predicted_id == end_token:
#         break
#     decoder_output_2 = tf.tensor_scatter_nd_add(decoder_output_2, [[0, i + 1]], predicted_id)
# decoder_output_2 == decoder_output
#
# encoder_input = source_vectorisation([f'{en_sentence}'])
# sentence_max_length = encoder_input.shape[1]
# decoder_output = target_vectorisation(['start'])[:, :-1]
# end_token = target_vectorisation(['end'])[:, 0]
#
# for i in tf.range(sentence_max_length - 1):
#     predictions = transformer([encoder_input, decoder_output], training=False)
#     predicted_id = tf.argmax(predictions[:, i, :], axis=-1)
#     if predicted_id == end_token:
#         break
#     decoder_output = tf.tensor_scatter_nd_add(decoder_output, [[0, i + 1]], predicted_id)
'''
####################
'''

'''
KERAS IMPLEMENTATION WITH CUSTOM OBJECTS
'''

source_vocab_size=len(source_vectorisation.get_vocabulary())
target_vocab_size=len(target_vectorisation.get_vocabulary())
seq_len=30
embed_dim=128
dense_output_dim=512
num_heads=8
num_layers=4
dropout_rate=0.1

inputs = (iter(train_ds).next()[0][0], iter(train_ds).next()[0][1])
source_inputs, target_inputs = inputs

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
encoder = Encoder(sequence_length=seq_len,
                               num_layers=num_layers,
                               embed_dim=embed_dim,
                               num_heads=num_heads,
                               dense_output_dim=dense_output_dim,
                               vocab_size=source_vocab_size)(encoder_inputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="chinese")
decoder = Decoder(seq_len=seq_len,
                               vocab_size=target_vocab_size,
                               embed_dim=embed_dim,
                               dense_output_dim=dense_output_dim,
                               num_heads=num_heads,
                               num_layers=num_layers)(decoder_inputs,encoder)

dropout = layers.Dropout(dropout_rate)(decoder)

final_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')(dropout)

transformer = keras.Model([encoder_inputs, decoder_inputs], final_layer)

learning_rate = CustomSchedule(embed_dim=128)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformer.compile(
    optimizer=optimizer, #optimizer
    loss=masked_loss, #masked_loss
    metrics=[masked_accuracy])

transformer.fit(
    train_ds,
    epochs=1,
    validation_data=val_ds
)

transformer.save('../saved_models/keras_transformer')
loaded_model_keras = tf.keras.models.load_model('../saved_models/keras_transformer', custom_objects={'CustomSchedule':CustomSchedule, 'masked_loss':masked_loss, 'masked_accuracy':masked_accuracy})
