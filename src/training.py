'''
Imports
'''

from data_processing import data_load, split_data, vectorise_data
from model import Transformer, CustomSchedule, masked_accuracy, masked_loss
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.family'] = ['Heiti TC']

'''
Load and preprocess data
'''
text_pairs = data_load("/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation/data/cmn-eng/cmn.txt")

train_pairs, val_pairs, test_pairs = split_data(text_pairs)
source_vectorisation, target_vectorisation, en, cn = vectorise_data(train_pairs)
_, _, en_val, cn_val = vectorise_data(val_pairs)
_, _, en_test, cn_test = vectorise_data(test_pairs)

# Save source and target vocabularies
model_sv = tf.keras.models.Sequential()
model_sv.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model_sv.add(source_vectorisation)
filepath = "source_vectorisation"
model_sv.save(filepath, save_format="tf")

model_tv = tf.keras.models.Sequential()
model_tv.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model_tv.add(target_vectorisation)
filepath = "target_vectorisation"
model_tv.save(filepath, save_format="tf")

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

'''
Train model locally
'''

transformer = Transformer(
    source_vocab_size=len(source_vectorisation.get_vocabulary()),
    target_vocab_size=len(target_vectorisation.get_vocabulary()),
    seq_len=30,
    embed_dim=128,
    dense_output_dim=512,
    num_heads=8,
    num_layers=4,
    dropout_rate=0.5
)

learning_rate = CustomSchedule(embed_dim=128)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

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
transformer.save('/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation/saved_models/keras_transformer.keras')





