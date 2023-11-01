
# Line-by-line Explanation of Sentiment Analysis with RNN

## Import Statements

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
```
- **numpy**: Used for numerical operations.
- **tensorflow**: The deep learning framework used for model creation, training, and evaluation.
- **Tokenizer**: A class from Keras that helps in tokenizing text data.
- **pad_sequences**: A function to ensure all sequences have the same length.
- **imdb**: The IMDb dataset provided by Keras.
- **matplotlib**: Used for plotting graphs.
- **precision_score, recall_score, f1_score**: Metrics from scikit-learn to evaluate the model's performance.
- **EarlyStopping**: A callback to stop training when a monitored metric has stopped improving.

## Data Loading and Conversion to Text

```python
(x_train_raw, y_train), (x_test_raw, y_test) = imdb.load_data()
```
- Loads the IMDb dataset. Reviews are represented as sequences of integers. 

```python
word_to_id = imdb.get_word_index()
```
- Retrieves a dictionary where keys are words and values are their corresponding integer representations.

```python
word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
```
- Adjusts the dictionary to account for special tokens: padding, start, and unknown.

```python
id_to_word = {value: key for key, value in word_to_id.items()}
```
- Creates an inverse dictionary to map integer representations back to words.

```python
x_train_text = [' '.join([id_to_word.get(i, '?') for i in seq]) for seq in x_train_raw]
x_test_text = [' '.join([id_to_word.get(i, '?') for i in seq]) for seq in x_test_raw]
```
- Converts the integer sequences back to text to simulate raw text data.

## Tokenization

```python
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train_text)
x_train = tokenizer.texts_to_sequences(x_train_text)
x_test = tokenizer.texts_to_sequences(x_test_text)
```
- Initializes a tokenizer, fits it on the training text data, and then converts the text data into sequences of integers.

```python
x_train = pad_sequences(x_train, maxlen=300, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=300, padding='post', truncating='post')
```
- Ensures all sequences have a consistent length of 300 by padding or truncating them.

## Model Architecture

```python
model = tf.keras.Sequential()
```
- Initializes a sequential model, which is a linear stack of layers.

```python
model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=300))
```
- Adds an embedding layer that converts integer sequences into dense vectors of fixed size.

```python
model.add(tf.keras.layers.GRU(64, dropout=0.2, recurrent_dropout=0.2))
```
- Adds a GRU layer, a variant of RNN, with dropout for regularization.

```python
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```
- Adds two dense layers: the first with ReLU activation and the second with sigmoid activation for binary classification.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- Compiles the model with the Adam optimizer, binary cross-entropy loss, and tracks accuracy as a metric.

## Training

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
```
- Initializes an early stopping callback to monitor validation loss and stop training if it doesn't improve for 2 consecutive epochs.

```python
history = model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_test, y_test), callbacks=[early_stopping])
```
- Trains the model on the training data and validates on the test data.

## Plotting Training History

- Uses `matplotlib` to plot training and validation accuracy and loss over epochs.

## Model Evaluation

```python
loss, accuracy = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```
- Evaluates the model on the test data and calculates accuracy, precision, recall, and F1-score.

## Sample Predictions

- Displays two random reviews from the test set and their predicted sentiment.
