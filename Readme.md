
# Sentient Analysis with RNN
Partial Requirement for MEM 531
USTP CdO
This code provides an example of sentiment analysis using a Recurrent Neural Network (RNN) on the IMDb dataset. The goal is to predict whether a movie review is positive or negative based on its text.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- scikit-learn

## Overview

1. **Data Preprocessing**: 
   - The IMDb dataset is loaded, and each movie review is represented as a sequence of integers.
   - Custom tokenization is applied to convert raw movie reviews into sequences of integers.
   - Sequences are padded to have a consistent length.
   
2. **Model Architecture**: 
   - An embedding layer that converts integer representations into dense vectors.
   - A GRU layer (a variant of RNN) for sequence processing.
   - A dense layer with ReLU activation.
   - An output layer with a sigmoid activation to predict the probability of the review being positive.

3. **Training**:
   - The model is trained using the Adam optimizer and binary cross-entropy loss.
   - Early stopping is implemented to halt training if the validation loss doesn't improve for consecutive epochs.

4. **Evaluation**:
   - The model's performance is evaluated on a test set.
   - Metrics like accuracy, precision, recall, and F1-score are calculated.

5. **Sample Predictions**:
   - Two random reviews from the test set are selected, and their predicted sentiment is displayed.

## Usage

1. Ensure you have the required libraries installed:

```bash
pip install tensorflow matplotlib scikit-learn
```

2. Run the Python script:

```bash
python sentiment_analysis_rnn_tokenized.py
```

3. Observe the training process, the plotted training/validation accuracy and loss curves, and the evaluation metrics.
4. Check the sample reviews and their predicted sentiments at the end of the output.
