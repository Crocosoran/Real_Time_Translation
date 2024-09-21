'''
Imports
'''

from data_processing import data_load, split_data, vectorise_data
from model import CustomSchedule, masked_accuracy, masked_loss
import tensorflow as tf
import matplotlib
import os
matplotlib.rcParams['font.family'] = ['Heiti TC']

'''
Load, preprocess, and create train, validation, and test sets
'''
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
saved_models_path = os.path.join(project_root, "saved_models")
data_path = os.path.join(project_root, "data/cmn-eng/cmn.txt")
text_pairs = data_load(data_path)

train_pairs, val_pairs, test_pairs = split_data(text_pairs)
source_vectorisation, target_vectorisation, en, cn = vectorise_data(train_pairs)
_, _, en_val, cn_val = vectorise_data(val_pairs)
_, _, en_test, cn_test = vectorise_data(test_pairs)

def create_datasets(source_language: str, target_language: str, batch_size: int = 64):
    def format_dataset(eng, ch):
        eng = source_vectorisation(eng)
        ch = target_vectorisation(ch)
        return (eng, ch[:, :-1]), ch[:, 1:]

    eng_texts = list(source_language)
    ch_texts = list(target_language)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, ch_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()  # Use in-memory caching to speed up preprocessing.


train_ds = create_datasets(source_language=en[0:10], target_language=cn[0:10])
val_ds = create_datasets(source_language=en_val[0:10], target_language=cn_val[0:10])
test_ds = create_datasets(source_language=en_test, target_language=cn_test)

'''
Inference
'''

reloaded = tf.saved_model.load(saved_models_path)

predict_fn = reloaded.signatures['serving_default']

en_sentence = 'I want to eat cake'
encoder_input = source_vectorisation([f'{en_sentence}'])
sentence_max_length = encoder_input.shape[1]
decoder_output = target_vectorisation(['start'])[:, :-1]
end_token = target_vectorisation(['end'])[:, 0]

for i in tf.range(sentence_max_length - 1):
    predictions = predict_fn(source_inputs=encoder_input, target_inputs=decoder_output)
    attention_score = predictions['last_attention_score']
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

'''
Fine-Tuning: Wrap Model in Keras Wrapper
'''

class KerasWrapper(tf.keras.Model):
    '''
    Description:
        Wraps the loaded model in a Keras Wrapped (functional api) allowing for the model fine-tuning (training
        on additional data).
    '''
    def __init__(self, tf_model):
        super(KerasWrapper, self).__init__()
        self.tf_model = tf_model

    def call(self, inputs):
        source_inputs, target_inputs = inputs
        # Call the model's 'serving_default' function
        outputs = self.tf_model.signatures['serving_default'](
            source_inputs=source_inputs,
            target_inputs=target_inputs
        )['outputs']
        return outputs


keras_model = KerasWrapper(reloaded)

learning_rate = CustomSchedule(embed_dim=128)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

keras_model.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)

train_out = keras_model.fit(
    train_ds,
    epochs=1,
    validation_data=val_ds
)

'''
Wrapped Keras Model Inference: Translation
'''

en_sentence = 'I want to eat cake'
encoder_input = source_vectorisation([f'{en_sentence}'])
sentence_max_length = encoder_input.shape[1]
decoder_output = target_vectorisation(['start'])[:, :-1]
end_token = target_vectorisation(['end'])[:, 0]

for i in tf.range(sentence_max_length - 1):
    predictions = keras_model([encoder_input, decoder_output], training=False)
    predicted_id = tf.argmax(predictions[:, i, :], axis=-1)
    if predicted_id == end_token:
        break
    decoder_output = tf.tensor_scatter_nd_add(decoder_output, [[0, i + 1]], predicted_id)

output_tokens = decoder_output.numpy()[0]
ch_vocab = target_vectorisation.get_vocabulary()
ch_index_lookup = dict(zip(range(len(ch_vocab)), ch_vocab))
translated_sentence = ''.join(
    [ch_index_lookup[output_tokens[i]] for i in range(1, len(output_tokens)) if output_tokens[i] != 0])

'''
Export Fine-Tuned Wrapped Keras Model
'''

# Define a method to export the wrapped model
def export_model(model, export_dir):
    '''
    Description:
        A custom function that exports the Keras Wrapped Model, by creating an input_signature using the @tf.function
        decorator
    '''
    # Create a signature definition for saving
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 30], dtype=tf.int64, name='source_inputs'),
        tf.TensorSpec(shape=[None, 30], dtype=tf.int64, name='target_inputs')
    ])
    def serving_fn(source_inputs, target_inputs):
        outputs = model([source_inputs, target_inputs])
        return {'outputs': outputs}

    # Save the model
    tf.saved_model.save(
        model,
        export_dir,
        signatures={'serving_default': serving_fn}
    )


# Save the wrapped model
export_model(keras_model, '/Users/daniel/Desktop/PycharmProjects/Real_Time_Translation'
                          '/saved_models/cream')

# Load it the same way it was originally loaded -> tf.saved_model.load(saved_models_path)!
