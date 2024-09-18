'''
Imports
'''
import random
import tensorflow as tf
import string
import re
from tensorflow.keras import layers
from tensorflow import keras

'''
Data extraction
'''
text_file = "../data/cmn-eng/cmn.txt"
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]

text_pairs = []

# iterate over the lines in the file
for line in lines:
    english, chinese, _ = line.split("\t")  # each line has English, Chinese translation
    # tab separated
    chinese = "[start]" + chinese + "[end]"  # prepend "start" and "end" to match
    # template
    text_pairs.append((english, chinese))

# Number of phrases in the data
len(text_pairs)  # 29,668 phrases

'''
Train, Validation, Test Split
'''
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]  # 20,768
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]  # 4450
test_pairs = text_pairs[num_train_samples + num_val_samples:]  # 4450

'''
Text Vectorisation of English and Chinese

Prepare a custom string standardization function for the Chinese 
TextVectorization layer: it preserves [ and ] but strips ¿ 
(as well as all other characters from strings.punctuation).
'''
#
# strip_chars = string.punctuation + "¿"  # add a new character to be removed
# strip_chars = strip_chars.replace("[", "")  # remove character that were going to be removed
# strip_chars = strip_chars.replace("]", "")  # remove character that were going to be removed
#
# # No need for lowercasing the chinese words, as in Chinese there is no notion of upper or lower characters
# # def custom_standardisation(input_string):
# #     lowercase = tf.strings.lowercase(input_string)
# #     return tf.string.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")
#
#
# # Look at only the top 15000 most frequent words
# # Restrict each sentence to a maximum 20 tokens
# vocab_size = 15000
# sequence_length = 20
#
# # English layer
# source_vectorisation = tf.keras.layers.TextVectorization(
#     max_tokens=vocab_size,
#     output_mode='int',
#     output_sequence_length=sequence_length
# )
#
# # Chinese layer
# target_vectorisation = tf.keras.layers.TextVectorization(
#     max_tokens=vocab_size,
#     output_mode="int",
#     output_sequence_length=sequence_length + 1  # One extra token since we will
#     # need to offset the sentecnce by one step during training
# )
#
# # Learn Vocabulary/ Create Vocab
# train_english_texts = [pair[0] for pair in train_pairs]
# train_chinese_texts = [pair[1] for pair in train_pairs]
# source_vectorisation.adapt(train_english_texts)  # 15,000
# target_vectorisation.adapt(train_chinese_texts)  # 15,000

import jieba


strip_chars = string.punctuation + "¿"  # add a new character to be removed
strip_chars = strip_chars.replace("[", "")  # remove character that were going to be removed
strip_chars = strip_chars.replace("]", "")  # remove character that were going to be removed

# No need for lowercasing the chinese words, as in Chinese there is no notion of upper or lower characters
# def custom_standardisation(input_string):
#     lowercase = tf.strings.lowercase(input_string)
#     return tf.string.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


# Look at only the top 15000 most frequent words
# Restrict each sentence to a maximum 20 tokens
vocab_size = 15000
sequence_length = 20

# English layer
source_vectorisation = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Chinese layer


def chinese_tokenizer(text):
  # Tokenise using Jieba
  words = jieba.lcut(text)
  return words


# Chinese layer with Jieba tokenization
target_vectorisation = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1
)

# Learn Vocabulary/ Create Vocab
train_english_texts = [pair[0] for pair in train_pairs]
train_chinese_texts_tokenized = [" ".join(chinese_tokenizer(pair[1])) for pair in train_pairs] # Apply Chinese tokenization to the training data
source_vectorisation.adapt(train_english_texts)  # 15,000
target_vectorisation.adapt(train_chinese_texts_tokenized)  # 15,000



'''
Tokenisaation: Preparing dataset for translation task (via tf.data pipeline)
'''

batch_size = 64


# The input Chinese sentence doesn’t include the last token
# to keep inputs and targets at the same length.

# The target Chinese sentence is one step ahead.
# Both are still the same length (20 words).
def format_dataset(eng, ch):
    eng = source_vectorisation(eng)
    ch = target_vectorisation(ch)
    return ({
                "english": eng,
                "chinese": ch[:, :-1]}, ch[:, 1:])  # (eng, ch [same as eng length]), ch [one token ahead]


def make_dataset(pairs):
    eng_texts, ch_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    ch_texts = list(ch_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, ch_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()  # Use in-memory caching to speed up preprocessing.


for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['chinese'].shape: {inputs['chinese'].shape}")
    print(f"targets.shape: {targets.shape}")
# train_ds shape = (number_of_batches, batch_size) =(325, 64) -> (20,800 samples) (source samples were 20,768 so 32 are empty)
# the shape of every sample within a batch = (batch_size, sequence_length) = (64, 20)
# Hence we can look at train_ds shape to be (325, 64, 20) = (number_of_batches, batch_size, sequence_length)

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
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        # Prepare an Embedding layer for the token indices
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)  # (20000, 256)
        # Add another one for the token positions
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)  # (20, 256)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):  # shape = (64, 20)
        length = tf.shape(inputs)[-1]  # 20
        positions = tf.range(start=0, limit=length, delta=1)  # [0, 1, 2 ..., 20]
        embedded_tokens = self.token_embeddings(inputs)  # shape  = (64, 20, 256)
        embedded_positions = self.position_embeddings(positions)  # shape = (20, 256)
        # add both embedding vectors together via broadcasting
        return embedded_tokens + embedded_positions  # shape = (64, 20, 256)

        # Mask used to ignore the padding 0s in the inputs.

    # The mask will be propagated to the next layer.
    # Returns same shape vectors where values are True/ False if not 0 or 0, respectively.
    def compute_mask(self, inputs, mask=None):  # shape = (64, 20)
        return tf.math.not_equal(inputs, 0)  # similar shape to inputs = (64, 20)

    # Serialisation to save the model
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
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

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # 256
        self.dense_dim = dense_dim  # 2048
        self.num_head = num_heads  # 8
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
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
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


'''
TransformerDecoder
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

    # new_shape = (64*1, 1*20, 1*20)
    # mask is replciated 64 times -> shape (64, 20, 20)

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
Training the model
'''

embed_dim = 256
dense_dim = 2048
num_heads = 8

# For a
# sequence_length = 20
# vocab_size = 15000
# embed_dim = 256

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="chinese")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
# Encode the target sentence and combine it with encoded source sentence
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)

x = layers.Dropout(0.5)(x)
# Predict a word for each output position
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

transformer.fit(train_ds, epochs=1, validation_data=val_ds)

# Saving the model
transformer.save('my_transformer_model')


########
########
########


ch_vocab = target_vectorisation.get_vocabulary()
ch_index_lookup = dict(zip(range(len(ch_vocab)), ch_vocab))
max_decoded_sentence_length = 20

import numpy as np


def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorisation([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorisation(
            [decoded_sentence])[:, :-1]
        # Sample the next token.
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        # Convert the next token prediction to a string,
        # and append it to the generated sentence.
        sampled_token = ch_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        # Exit condition
        if sampled_token == "[end]":
            break
    return decoded_sentence


test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
