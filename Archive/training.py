'''
Imports

'''

import tensorflow as tf
from model import build_transformer_model
from data_processing import data_load, split_data, vectorise_data
import random
import string
import jieba
import tensorflow as tf
from typing import List
import re

'''
Load and preprocess data
'''
text_pairs = data_load("../data/cmn-eng/cmn.txt")
train_pairs, val_pairs, test_pairs = split_data(text_pairs)
source_vectorisation, target_vectorisation, en, cn = vectorise_data(train_pairs)

'''
Build dataset pipeline
'''
def create_datasets(source_language: str, target_language: str, batch_size: int = 64):

    def format_dataset(eng, ch):
        eng = source_vectorisation(eng)
        ch = target_vectorisation(ch)
        return ({
                    "english": eng,
                    "chinese": ch[:, :-1]}, ch[:, 1:])  # (eng, ch [same as eng length]), ch [one token ahead]

    eng_texts = list(source_language)
    ch_texts = list(target_language)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, ch_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()  # Use in-memory caching to speed up preprocessing.


train_ds = create_datasets(source_language = en, target_language = cn)
val_ds = create_datasets(source_language = en, target_language = cn)