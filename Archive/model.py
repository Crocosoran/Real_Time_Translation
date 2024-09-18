'''
Imports

'''
import tensorflow as tf
from tensorflow.keras import layers
import keras
import numpy as np

'''
Positional Encoding

The broadcasting operation results in each sequence in the batch, the corresponding row from embedded_positions (representing positional embeddings) is
added to all the token embeddings in that sequence.
This effectively adds positional information to each token in the sequence.

The PositionalEmbedding layer operates on a single sequence of token indices, not on batches directly.
When you apply this layer to the entire dataset (int_train_ds) (shape = (325, 64, 20)), it is applied independently to each sample within the batch (takes only (64, 20) at a time 325 times).
Therefore, the batch dimension is preserved.

The call method (defines the forward pass) is also invoked automatically upon initialisation of the PositionalEmbedding class.

The compute_mask method will also be called automatically by the framework.

Final output has shape = (number_of_batches, batch_size, sequence_length, embed_dim) = (325, 64, 20, 256)
'''


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, embed_dim, **kwargs):
        super().__init__(**kwargs)
        # Prepare an Embedding layer for the token indices
        # Mask used to ignore the padding 0s in the inputs.
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=embed_dim)  # (20000, 256)
        # Add another one for the token positions
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim)  # (20, 256)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.embed_dim = embed_dim

    def call(self, inputs):  # shape = (64, 20)
        length = tf.shape(inputs)[-1]  # 20
        positions = tf.range(start=0, limit=length, delta=1)  # [0, 1, 2 ..., 20]
        embedded_tokens = self.token_embeddings(inputs)  # shape  = (64, 20, 256)
        embedded_positions = self.position_embeddings(positions)  # shape = (20, 256)
        # add both embedding vectors together via broadcasting
        return embedded_tokens + embedded_positions  # shape = (64, 20, 256)

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0) # generates a tensor with True or Flase values (True if value is not 0 and
    # False if values is 0)

    # Serialisation to save the model
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

class PositionalEmbedding_sinusoids(layers.Layer):
    def __init__(self, sequence_length, input_dim, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.word_embedding_layer = layers.Embedding(input_dim = input_dim, output_dim = embed_dim)
        position_embedding_matrix = self.get_position_encoding(sequence_length, embed_dim)

        self.position_embedding_layer = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )

    # The sinusoidal encoding ensures that each position in the sequence is mapped to a unique combination of
    # sinusoidal values, which the Transformer model can use to determine the relative positions of tokens.
    # The smooth variation across embedding dimensions helps the model capture both local and
    # global positional relationships within the sequence.

    # As i increases, the frequency of the sinusoid decreases, meaning that lower dimensions correspond to
    # high-frequency variations (which can capture fine-grained positional differences), while higher dimensions
    # correspond to low-frequency variations (which capture broader positional information).
    def get_position_encoding(self, seq_len, d, n=10000): # positional embeddings are applied to the embed_dim !
        P = np.zeros((seq_len, d)) # length of sequence, embedding dimension e.g., (20, 512)
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = n**(2*i/d)
                P[k, 2 * i] = np.sin(k / denominator) # 0, 2, 4 -> sine for even
                P[k, 2 * i + 1] = np.cos(k / denominator) # 1, 3, 5 -> cosine for odd
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
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


'''
Transformer Encoder

Transforms in source input vectors in English to vectors that have context-awareness due to self-attention via MultiHeadAttention.

Residual connection are added with the goal of re-injecting past valuable residual information, which would otherwise be lost.

Normalisation layer helps gradients flow better during back-propagation. Each sequence is normalised independently from other sequence in the batch.

Two Dense layers separated by a ReLu activation function are added to introduce non-linearity into the model,
allowing it to learn more complex relationships and patterns in the data resulting in a richer overall output.

Multi-head requires both num_heads and embed_dim for the following reason:
- The word "water" would be linearly projected to a query vector Q2: [0.7, 0.4, 0.8, 0.2].
- The embedding space of each token is split based on num_head along embed_dim axis. For a single token in a sequence,
assuming num_head = 2 and embed_dim = 4-> Q2_head1: [0.7, 0.4] & Q2_head2: [0.8, 0.2].

Final output shape = (number_of_batches, batch_size, sequence_length, embed_dim) = (325, 64, 20, 256)
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
            query = x,
            key = context,
            value = context,
            return_attention_scores = True
        )

    # Cache the attention scores for plotting
        self.last_attention_scores = attention_scores

        x = self.add([x, attention_output]) # ensures masks are propagated, because + does not
        x = self.layernorm(x)

        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attention_output = self.attention(
            query=x,
            key=x,
            value=x
        )
        x = self.add([x, attention_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attention_output = self.attention(
            query=x,
            key=x,
            value=x,
            use_causal_mask = True
        )
        x = self.add([x, attention_output])
        x = self.layernorm(x)
        return x



class TransformerEncoder(layers.Layer):
    def __init__(self,num_layers, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # 256
        self.dense_dim = dense_dim  # 2048
        self.num_head = num_heads  # 8
        self.num_layers = num_layers # 4 how many times should we repeat the encoder block
        self.attention = layers.MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=embed_dim)  # need both embed_dim and num_heads as num_heads
        # cannot be larger than embed_dim
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation='relu'),
             layers.Dense(embed_dim), ]  # default: activation = 'linear'
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    # Computation goes in call
    # Mask is created automatically from the embedding layer
    def call(self, inputs, mask=None):
        # the mask that will be generated by the embedding layer will be 2D
        # but the attention layer expects it to be 3D or 4D
        # so we expand its rank
        if mask is not None:
            mask = mask[:, tf.newaxis, :]  # shape = (64, 1, 20)
        # Input and Output = (input_seq_number, embed_dim) (64, 20)
        # Self attention of English sentences + Mask only!
        attention_output = self.attention(inputs, inputs,inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)  # Residual Connection
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)  # Residual Connection

    # Serialisation to save the model
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

class TransformerEncoder_2(layers.Layer):
    def __init__(self, num_layers, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers  # Number of times to repeat the encoder block

        # Create multiple encoder layers (blocks)
        self.encoder_layers = [
            EncoderBlock(embed_dim, dense_dim, num_heads) for _ in range(num_layers)
        ]

    def call(self, inputs, mask=None):
        x = inputs
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, mask)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "embed_dim": self.encoder_layers[0].embed_dim,  # Assuming all layers have the same dimensions
            "dense_dim": self.encoder_layers[0].dense_dim,
            "num_heads": self.encoder_layers[0].num_heads,
        })
        return config

class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)  # Residual connection
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config
'''
Transformer Decoder
'''


# Generate matrix of shape (sequence_length, sequence_length) (20, 20) with 1s
# in one half and 0s in the other. (Mimic the Attention matrix during
# self-attention.

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        # Ensures that the layer will propagate its input mask to its outputs.
        # Masking in Keras is secplicitly opt-in.
        # If we pass a mask to a layer that doesn't implement compute_mask() and
        # that doesn't expose this supports_masking attribute, that's an error.
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, p):
        input_shape = tf.shape(p)  # (64, 20, 256)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]  # (20, 1)
        j = tf.range(sequence_length)  # (20)
        mask = tf.cast(i >= j, dtype="int32")  # (20, 20) (compare each i with each j)
        # Replicate it along the batch axis to get a matrix of shape
        # (batch_size, sequence_length, sequence_length)
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))  # (1, 20, 20)
        # same as mask[tf.newaxis, :, :]
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),  # convert scalar [64] -> [64,] 1D vector same as x.reshape(batch_size,
             # (1,))
             tf.constant([1, 1], dtype=tf.int32)], axis=0)  # tensor = [1,1], shape (2,)
        # mult shape = (3,) = [64, 1, 1]
        return tf.tile(mask, mult)  # [64, 1, 1] and mask.shape = (1, 20, 20)

    def call(self, inputs, encoder_outputs, mask=None):  # inputs = chinese
        # Retrieve the causal mask
        causal_mask = self.get_causal_attention_mask(inputs)
        # Prepare input mask (describes the padding locations in the target sequence)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            # Merge the two masks together
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)  # Pass casual mask to the first attention
        # layer, which performs self-attention over the target sequence.
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,  # chinese attention-aware vectors
            value=encoder_outputs,  # english attention-aware encoder output vectors
            key=encoder_outputs,
            attention_mask=padding_mask)  # Pass the combined mask to the second
        # attention layer, which relates the source seqeucne to the target sequence
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)  # residual connection
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)  # residual connection


    

'''
Build Transformer Model
'''

# sequence_length=30
# vocab_size=15000
# embed_dim=256
# dense_dim = 2048
# num_heads=8
# encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
# encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
#
# decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="chinese")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
# # Encode the target sentence and combine it with encoded source sentence
# x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)

def build_transformer_model(sequence_length: int, vocab_size: int, embed_dim: int, dense_dim: int, num_heads: int,
                            dropout: int = 0.5):
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
    x = PositionalEmbedding_sinusoids(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="chinese")
    x = PositionalEmbedding_sinusoids(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    # Encode the target sentence and combine it with encoded source sentence
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)

    x = layers.Dropout(dropout)(x)
    # Predict a word for each output position
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
    transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return transformer

# transformer = build_transformer_model(sequence_length=30, vocab_size=15000, embed_dim=256, dense_dim = 2048, num_heads=8)