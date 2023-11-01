
# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

# Load raw IMDb dataset
(x_train_raw, y_train), (x_test_raw, y_test) = imdb.load_data()

# Convert integer sequences back to text
word_to_id = imdb.get_word_index()
word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value: key for key, value in word_to_id.items()}
x_train_text = [' '.join([id_to_word.get(i, '?') for i in seq]) for seq in x_train_raw]
x_test_text = [' '.join([id_to_word.get(i, '?') for i in seq]) for seq in x_test_raw]

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train_text)
x_train = tokenizer.texts_to_sequences(x_train_text)
x_test = tokenizer.texts_to_sequences(x_test_text)

# Padding
x_train = pad_sequences(x_train, maxlen=300, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=300, padding='post', truncating='post')

# Data Preprocessing
# ------------------

# Load IMDb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pad the sequences
x_train = pad_sequences(x_train, maxlen=300)
x_test = pad_sequences(x_test, maxlen=300)

# Model Architecture
# ------------------

# Initialize the model
model = tf.keras.Sequential()

# Embedding layer
model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=300))

# GRU layer
model.add(tf.keras.layers.GRU(64, dropout=0.2, recurrent_dropout=0.2))

# Fully connected layer
model.add(tf.keras.layers.Dense(64, activation='relu'))

# Output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
# --------

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Train the model
history = model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Plotting
# --------

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()

# Evaluation
# ----------

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Calculate precision, recall, and F1-score
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")

# Display sample reviews with predicted sentiment
# -----------------------------------------------

# Function to decode the reviews
def decode_review(encoded_review):
    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value: key for key, value in word_to_id.items()}
    return ' '.join([id_to_word.get(i, '?') for i in encoded_review])

# Randomly select 2 reviews
indices = np.random.choice(len(x_test), 2, replace=False)

for index in indices:
    print("-" * 50)
    print("Review:")
    print(decode_review(x_test[index]))
    sentiment = "Positive" if y_pred[index] == 1 else "Negative"
    print(f"Predicted Sentiment: {sentiment}")
