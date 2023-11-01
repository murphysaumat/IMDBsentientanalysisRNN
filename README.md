# IMDBsentientanalysisRNN
Sentiment Analysis using RNN on IMDb Reviews

Objective

The objective of this project is to implement a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units to perform sentiment analysis on IMDb movie reviews. The goal is to classify reviews as either 'positive' or 'negative'.

Approach

Data Preprocessing

The IMDb dataset provided by TensorFlow is loaded and tokenized.
The reviews are padded to have a uniform length of 300.

Model Architecture

For word representation, an embedding layer with an input dimension of 10,000, an output dimension of 128, and an input length of 300 is used.
An LSTM layer with 128 units and dropout to reduce overfitting.
A Dense layer with 128 units and ReLU activation.
A Dense layer with a single unit and Sigmoid activation for binary classification.

Training

Adam optimizer is used with a learning rate of 0.001.
Binary Cross-Entropy is used as it's a binary classification problem.
Accuracy is tracked during training.
The model is trained using mini-batch gradient descent with a batch size of 32.

Evaluation

The model is evaluated on a separate test set.
Besides accuracy, precision, recall, and F1-score are also calculated for evaluation.

Results

Results will vary depending on the training run and the hyperparameters used. 
You can expect an accuracy around 85% (subject to change) with the current setup.

How to Run

Prerequisites

- Python 3.x
- TensorFlow
- Matplotlib
- NumPy
- scikit-learn

Installation

Install the required packages using pip:

pip install numpy tensorflow matplotlib scikit-learn

Execution

1. Save the Python code in a file, for example, `sentiment_analysis_rnn.py`.
2. Run the script from the terminal:

python sentiment_analysis_rnn.py

This will train the model, plot the training and validation accuracy and loss, and print the evaluation metrics on the console.

Further Work

Hyperparameter tuning can be done for better results. You can experiment with:

1. Different learning rates
2. Different batch sizes
3. Different sequence lengths
4. Adding more LSTM or GRU layers
