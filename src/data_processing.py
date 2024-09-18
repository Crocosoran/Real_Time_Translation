'''
Data extraction
Splitting
Vectorisation

'''

'''
Imports
'''
import random
import string
import jieba
import tensorflow as tf
from typing import List
import re
import matplotlib.pyplot as plt
import numpy as np
'''
Data extraction
Input: text_file = "../data/cmn-eng/cmn.txt"
'''


def data_load(text_file: str) -> List:
    with open(text_file) as f:
        lines = f.read().split('\n')[:-1]

    text_pairs = []
    for line in lines:
        english, chinese, _ = [splits for splits in line.split('\t') if
                               splits]  # may not need _ as if statement remvoes spaces
        # chinese = '[start]' + chinese + '[end]'
        text_pairs.append((english, chinese))
    return text_pairs


'''
Train, Validation, Test Split
'''


def split_data(text_pairs: List, val_split: float = 0.15) -> List:
    random.seed(0)  # !!! may not work
    random.shuffle(text_pairs)
    num_val_samples = int(val_split * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]  # 20,768
    val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]  # 4450
    test_pairs = text_pairs[num_train_samples + num_val_samples:]  # 4450
    return train_pairs, val_pairs, test_pairs


'''
Text Vectorisation of English and Chinese
'''


def custom_stadardization_fn(string_tensor):
    additional_characters = "。？，！"
    strip_char = string.punctuation + additional_characters
    return tf.strings.regex_replace(string_tensor, f"[{re.escape(strip_char)}]", "")


def vectorise_data(text_pairs: List, vocab_size: int = 15000, sequence_length: int = 30):  # return vocabularies

    # No need for lowercasing the chinese words, as in Chinese there is no notion of upper or lower characters

    # English layer
    source_vectorisation = tf.keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation',
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # Chinese layer
    def chinese_tokenizer(text):
        # Tokenise using Jieba
        words = jieba.lcut(text)
        return words

    def chinese_tokenizer_single_characters(text):
        words = " ".join([word for word in text])
        return words

    # padding token = index 0 and ''
    # OOV token = index 1 and '[UNK]'

    # Chinese layer with Jieba tokenization

    target_vectorisation = tf.keras.layers.TextVectorization(
        standardize=custom_stadardization_fn,
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1
    )

    # Learn Vocabulary/ Create Vocab via adapt
    english_texts_tokenized = [pair[0] for pair in text_pairs]
    # may need to split the chinese ones the same way we split the english ones
    # train_chinese_texts_tokenized = ["[start] " + " ".join(chinese_tokenizer(pair[1])) + " [end]" for pair in train_pairs] # Apply Chinese tokenization to the training data
    chinese_texts_tokenized = ["[start] " + chinese_tokenizer_single_characters(pair[1]) + " [end]" for pair in
                               text_pairs]  # Apply Chinese tokenization to the training data
    source_vectorisation.adapt(english_texts_tokenized)  # 15,000
    target_vectorisation.adapt(chinese_texts_tokenized)  # 15,000
    # get vocabulary: target_vectorisation.get_vocabulary()
    return source_vectorisation, target_vectorisation, english_texts_tokenized, chinese_texts_tokenized



def head_cross_attention_scores(source_sentence_tokens, translated_tokens, attention_score):
    ax = plt.gca()
    ax.matshow(attention_score[:len(translated_tokens),:len(source_sentence_tokens.split())])
    ax.set_xticks(range(len(source_sentence_tokens.split())))
    ax.set_yticks(range(len(translated_tokens)))


    x_labels = source_sentence_tokens.split()
    ax.set_xticklabels(x_labels, rotation = 90)

    y_labels = [token for token in translated_tokens]
    ax.set_yticklabels(y_labels)

    for t in range(len(translated_tokens)):
        for s in range(len(source_sentence_tokens.split())):
            text = ax.text(s,t,np.round(attention_score[t,s],2), ha = 'center', va='center', color='w')


def plot_attention_scores(source_sentence_tokens, translated_tokens, attention_score):
    fig = plt.figure(figsize=(16,8))

    for h in range(attention_score.shape[1]):
        ax = fig.add_subplot(2,4,h+1)
        head = attention_score.numpy()[0,h,:,:]
        head_cross_attention_scores(source_sentence_tokens, translated_tokens, head)
        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout
    plt.show()

