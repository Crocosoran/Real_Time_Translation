"""
Imports
"""

import tensorflow as tf
from tensorflow.keras import layers
import keras
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = ['Heiti TC']

"""
Model Development
"""


class PositionalEmbedding_sinusoids(layers.Layer):
    """
    Arguments:
        seq_len = length of the input sequence measured in tokens
        vocab_size = size of the input vocabulary
        embed_dim = size of the embedding dimension to be added to the input sequence dimensions

    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        This effectively adds positional information to each token in the sequence.

        The PositionalEmbedding layer operates on a single sequence of token indices, not on batches directly.
        When you apply this layer to the entire dataset, it is applied independently to each sample within the batch.
        Therefore, the batch dimension is preserved.

        The call method (defines the forward pass) is also invoked automatically upon initialisation of the
        PositionalEmbedding class.

        The compute_mask method will also be called automatically by the framework.


        The sinusoidal encoding ensures that each position in the sequence is mapped to a unique combination of
        sinusoidal values, which the Transformer model can use to determine the relative positions of tokens.
        The smooth variation across embedding dimensions helps the model capture both local and
        global positional relationships within the sequence.

        As i increases, the frequency of the sinusoid decreases, meaning that lower dimensions correspond to
        high-frequency variations (which can capture fine-grained positional differences), while higher dimensions
        correspond to low-frequency variations (which capture broader positional information).
    """

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

    def get_position_encoding(self, seq_len, embed_dim,
                              n=10000):
        P = np.zeros((seq_len, embed_dim))
        for k in range(seq_len):
            for i in np.arange(int(embed_dim / 2)):
                denominator = n ** (2 * i / embed_dim)
                P[k, 2 * i] = np.sin(k / denominator)  # 0, 2, 4 -> sine for even
                P[k, 2 * i + 1] = np.cos(k / denominator)  # 1, 3, 5 -> cosine for odd
        return P

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.word_embedding_layer(inputs)
        embedded_positions = self.position_embedding_layer(positions)
        return embedded_tokens + embedded_positions

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
    """
    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Perform the multi-head attention operation coupled with normalisation
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.attention = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()


class CrossAttention(BaseAttention):
    """
    Input_1 & Input_2:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Perform the multi-head attention operation specifically in the case where the query = Encoder Output
        (token sequence in source langauge/ language to translated from, the key & value = Decoder Input (token sequence
        in target language/ language to translate to).
    """

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
    """
    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Perform the multi-head attention operation specifically in the case where the query & key & value =
        Source Sequence.
    """

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
    """
    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Perform the multi-head attention operation specifically in the case where the query & key & value =
        Target Sequence. A mask is also applied in order to prevent the sequence to infer future tokens during inference.
    """

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
    """
    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Passes the Output of an attention layer through Dense, Dropout, and Normalisation layers.
    """

    def __init__(self, embed_dim, dense_output_dim, dropout_rate=0.1):
        super().__init__()
        self.sequence = tf.keras.Sequential(
            [
                layers.Dense(dense_output_dim, activation='relu'),
                layers.Dense(embed_dim),
                layers.Dropout(dropout_rate)
            ])
        self.add = tf.keras.layers.Add()
        self.layernorm = layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.sequence(x)])
        x = self.layernorm(x)
        return x


class EncoderLayer(layers.Layer):
    """
    Arguments:
        embed_dim = size of the embedding dimension to be added to the input sequence dimensions
        num_heads = number of heads in the multi-head attention mechanism
        dense_output_dim = desired output size from the Dense layers
        dropout_rate = a regularisation parameter that assigns a probability for each neuron in a layer to be cast as 0

    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Applies the attention operation for the source sequence, followed by a feed forward neural network.
    """

    def __init__(self, *, embed_dim, num_heads, dense_output_dim, dropout_rate=0.1):
        super().__init__()

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


class Encoder(layers.Layer):
    """
    Arguments:
        sequence_length = length of the input sequence measured in tokens
        num_layers = number of encoder blocks applied sequentially
        embed_dim = size of the embedding dimension to be added to the input sequence dimensions
        num_heads = number of heads in the multi-head attention mechanism
        dense_output_dim = desired output size from the Dense layers
        vocab_size = size of the input vocabulary
        dropout_rate = a regularisation parameter that assigns a probability for each neuron in a layer to be cast as 0

    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Calls all components that make-up the encoder, sequentially.

        Transforms source input vectors in English to vectors that have context-awareness due to self-attention via MultiHeadAttention.

        Residual connection are added with the goal of re-injecting past valuable residual information, which would otherwise be lost.

        Normalisation layer helps gradients flow better during back-propagation. Each sequence is normalised independently from other sequence in the batch.

        Two Dense layers separated by a ReLu activation function are added to introduce non-linearity into the model,
        allowing it to learn more complex relationships and patterns in the data resulting in a richer overall output.

        Multi-head requires both num_heads and embed_dim for the following reason:
        - The word "water" would be linearly projected to a query vector Q2: [0.7, 0.4, 0.8, 0.2].
        - The embedding space of each token is split based on num_head along embed_dim axis. For a single token in a sequence,
        assuming num_head = 2 and embed_dim = 4-> Q2_head1: [0.7, 0.4] & Q2_head2: [0.8, 0.2].
    """
    def __init__(self, *, sequence_length, num_layers, embed_dim, num_heads, dense_output_dim, vocab_size,
                 dropout_rate=0.1):
        # set up autograd
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        self.positional_embedding = PositionalEmbedding_sinusoids(
            seq_len=sequence_length,
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )

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
        x = self.positional_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.encoder_layer[i](x)

        return x

class DecoderLayer(layers.Layer):
    """
    Arguments:
        embed_dim = size of the embedding dimension to be added to the input sequence dimensions
        dense_output_dim = desired output size from the Dense layers
        num_heads = number of heads in the multi-head attention mechanism
        dropout_rate = a regularisation parameter that assigns a probability for each neuron in a layer to be cast as 0

    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Applies the causal attention operation for the source sequence, followed by the cross attention operation
        that integrates both the encoder output and the causal attention output. The final output passes through
        a feed forward neural network.
    """
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
        x = self.ffn(x)

        return x


class Decoder(layers.Layer):
    """
    Arguments:
        seq_len = length of the input sequence measured in tokens
        vocab_size = size of the input vocabulary
        embed_dim = size of the embedding dimension to be added to the input sequence dimensions
        dense_output_dim = desired output size from the Dense layers
        num_heads = number of heads in the multi-head attention mechanism
        num_layers = number of decoder blocks applied sequentially
        dropout_rate = a regularisation parameter that assigns a probability for each neuron in a layer to be cast as 0

    Input:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Output:
        Type: tf.tensor()
        Shape: [batch_size, sequence_size, embed_dim]

    Description:
        Calls all components that make-up the decoder, sequentially.
        The Decoder has similar architecture to the Encoder, but with the addition of
        an additional attention block (sandwiched between two LayerNormalization layers)

        The first attention block (Attention 1) would take as query, key and value the target embedded Chinese
        sequence and perform self-attention.

        Attention 2 on the other hand would take query as the output from attention 1; the key and value as the
        output of the English source TransformerEncoder.

        Causal mask is quintessential for the correct training of the sequence-to-sequence Transformer. As the
        Decoder is order-agnostic, during training, it would use the entire input sequence and simply
        learn to copy input step N + 1 to location N in the output. It will have perfect training accuracy but the
        representation it would have learnt would be useless as it is able to "peak" into the future word tokens of
        the sequence, instead of generating them one step at a time. What the causal mask does to resolve this issue
        is that it essentially masks the pairwise attention matrix.

        The causal mask would have values  of 1 and 0:

        [[1 0 0 0 0 0 0 0 0 0]
         [1 1 0 0 0 0 0 0 0 0]
         [1 1 1 0 0 0 0 0 0 0]
         [1 1 1 1 0 0 0 0 0 0]
         [1 1 1 1 1 0 0 0 0 0]
         [1 1 1 1 1 1 0 0 0 0]
         [1 1 1 1 1 1 1 0 0 0]
         [1 1 1 1 1 1 1 1 0 0]
         [1 1 1 1 1 1 1 1 1 0]
         [1 1 1 1 1 1 1 1 1 1]]

         Hence, you can imagine placing this mask on-top of the actual attention matrix (same shape = (
         sequence_length, sequence_length)). In this way the 0 values would not allow Transformer to "see" values
         beyond N (it will not simply look them up and add them), but rather it would have to predict and generate
         them.
    """
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

        return x


@keras.saving.register_keras_serializable()
class Transformer(tf.keras.Model):
    """
    Arguments:
        source_vocab_size = size of the source language vocabulary
        target_vocab_size = size of the target language vocabulary
        seq_len = length of the input sequence measured in tokens
        embed_dim = size of the embedding dimension to be added to the input sequence dimensions
        dense_output_dim = desired output size from the Dense
        layers num_heads = number of heads in the multi-head attention mechanism
        num_layers = number of encoder & decoder blocks applied sequentially
        dropout_rate = a regularisation parameter that assigns a probability for each neuron in a layer to be cast as 0

        Input:
            Type: tf.tensor()
            Shape: [batch_size, sequence_size, embed_dim]

        Output:
            Type: tf.tensor()
            Shape: [batch_size, sequence_size, embed_dim]

        Description: Calls all components that make-up the transformer, sequentially. Applies the encoder blocks,
        followed by decoder blocks. The final output passes through a dense layer and logits from the transformer are
        converted into probabilities via softmax activation function.
    """
    def __init__(self, source_vocab_size, target_vocab_size, seq_len, embed_dim, dense_output_dim, num_heads,
                 num_layers, dropout_rate=0.1, **kwargs):
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
    """
    Description:
        Uses Adam as an optimiser alongside a custom learning rate secularised according to the formulation in the
        paper "Attention Is All You Need"
    """
    def __init__(self, embed_dim, warmup_steps=4000, **kwargs):
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
    """
    Description:
        Implements a Custom Sparse Categorical Cross Entropy as the loss function
        that accounts for the padding mask.
        Internally applies a softmax to convert logits to probabilities
    """
    mask = label != 0

    # Expects logits NOT probabilities.
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
    """
    Description:
        Applies a custom accuracy function that takes into consideration sequence masking, as the evaluation metric
    """
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)
