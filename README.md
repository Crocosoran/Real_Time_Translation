# Real Time Translation: English to Chinese

The Real Time Translation model created in this project is based on the popular sequence-to-sequence Transformer architecture
that was originally proposed in the paper ["Attention is all you need" by Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).

In short, the Transformer is a neural network that leverages the self-attention mechanism to easily compute the relationship
between different parts of the input sequence. Moreover, the Transformer does not use a single attention mechanism, but 
multiple attention "heads", i.e., Multi Head Attention. It allows the model to attend to different parts of the 
sequence simultaneously, capturing different aspects of the relationships between tokens.

The Real Time Translation model follows closely the architecture from the original paper. Hence, it is composed of a 
4-layer Transformer (4-layer Encoder and 4-layer Decoder).

The font-end and back-end of the web service for Real Time Translation application are managed by Streamlit. 
To containerise the application and ensure consistent performance across different environments, Docker was utilised.

The containerised application is deployed on Google Cloud Run. [Demo here](https://crocosoran-app-1-2-32712861319.us-central1.run.app)

The application allows for the transaltion of simple sentences from English to Chinese, and subsequently 
visualises the attention heads for each translated token, providing a visual understanding how the Multi Head
Attention mechanism captures different aspects/ relationships between tokens.

## Data
The data used for the English and Chinese vocabulary construction, and subsequent model training and inference, 
consisted of 29,668 unique sentence in the following format ((English, Chinese)).

The latest version of the data can be found here: http://www.manythings.org/anki/
```
git clone https://github.com/Crocosoran/Real_Time_Translation.git
```

## Model Training
The data was tokenised into the individual logographic characters in Chinese (i.e., Hanzi), due to the fact that
typically each character represents a syllable and has its own meaning, and many characters can be combined to form words.
This also simplifies the data, by avoiding explicitly injecting multi-character relationships, allowing the model
to infer those on its own.

The data was split into: training (70% = 20,768), validation (15% = 4450), and testing (15% = 4450).
To train the model, the data was organized into batches with a batch size of 64, and the training process spanned 30 epochs.

***Note***: More epochs and wider, more diverse vocabulary would improve the model's 
generalisation capabilities significantly


Used the Adam optimiser to update the model weights during backpropagation and adaptively adjust the learning rates for each parameter.

As the target sequences (Chinese) are padded to avoid "looking" into future token during inference, 
the loss function employed was a custom cross-entropy function that applied a padding mask when calculating the loss.

## Setup
### Start the App
1. Clone the repository:
```
git clone https://github.com/Crocosoran/Real_Time_Translation.git
```
2. Make sure you cd into the project directory


3. Run the Streamlit app
```
streamlit run Streamlit/app.py
```

### Fine-tuning the model on more data
To fine tune the model, you need to load the trained model from:
```
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
saved_models_path = os.path.join(project_root, "saved_models")

reloaded = tf.saved_model.load(saved_models_path)
```

Afterwards, you can follow the steps in docstrings ***"Fine-Tuning: Wrap Model in Keras Wrapper"***
```src/test.py```, which involves wrapping the loaded model with a Keras wrapper function.

### Perform inference on source code level
To perform inference on the model outside the app you would need to:
1. Tokenise and vectorise your input sentence.
2. Load source and target vocabularies.
3. Create a training loop to perform inference on each token at a time.
4. Concatenate results to see the complete transalted sentence.

***Note:*** You can follow the steps in ```src/test.py``` for detailed guide. This applies to both the base
model and a fine-tuned version.
### Saving the model 
It is recommended to save the model in the TensorFlow SavedModel format, as it allows the model to be loaded 
in environments that have different tensorflow and/or keras versions.

## Things to keep in mind!
The model was trained in Google Collab on A-100. The entire model implementation works both locally and
on Google Colab. However, when saving the model on Google Collab in ***.keras*** format and loading the same
keras model locally there would be a version incompatibility, as locally the model was developed on
***tensorflow 2.13.1*** and Google Collab has ***tensorflow 2.17.0***, at the time of development. To circumvent 
this version issue, it's adviced to save the model in the TensorFlow SavedModel format, as allows the model
to be loaded across different version of TensorFlow.

