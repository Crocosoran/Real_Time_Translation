'''
Imports
'''

import tensorflow as tf
from tensorflow.keras import layers
import keras
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = ['Heiti TC']

'''
Model Development
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

    # def compute_mask(self, inputs, mask=None):
    #     return tf.math.not_equal(inputs, 0)

    # Serialisation to save the model
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size
        })
        return config


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


@keras.saving.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, source_vocab_size, target_vocab_size, seq_len, embed_dim, dense_output_dim, num_heads,
                 num_layers, dropout_rate=0.1, **kwargs):
        # if dtype is not None:
        #     if isinstance(dtype, dict) and 'class_name' in dtype and dtype['class_name'] == 'DTypePolicy':
        #         dtype = Policy(dtype['config']['name'])
        #     elif isinstance(dtype, str):
        #         dtype = Policy(dtype)
        super(Transformer, self).__init__(**kwargs)

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.dense_output_dim = dense_output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

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

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)  # , activation='softmax')

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

    def get_config(self):
        # Include all parameters used in __init__
        config = super(Transformer, self).get_config()
        config.update({
            'source_vocab_size': self.source_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'seq_len': self.seq_len,
            'embed_dim': self.embed_dim,
            'dense_output_dim': self.dense_output_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Return a new instance from the config
        return cls(**config)


@keras.saving.register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embed_dim, warmup_steps=4000, **kwargs):
        # if dtype is not None:
        #     if isinstance(dtype, dict) and 'class_name' in dtype and dtype['class_name'] == 'DTypePolicy':
        #         dtype = Policy(dtype['config']['name'])
        #     elif isinstance(dtype, str):
        #         dtype = Policy(dtype)
        super(CustomSchedule, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.embed_dim = tf.cast(self.embed_dim, dtype=tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'embed_dim': int(self.embed_dim),
            'warmup_steps': int(self.warmup_steps)
        }


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
