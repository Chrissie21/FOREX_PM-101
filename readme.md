# Forex Prediction Machine Learning Model

This project builds a deep learning model to predict the closing prices of the USD/JPY currency pair using historical data. The model is built using a Bidirectional LSTM (Long Short-Term Memory) network and Gated Recurrent Unit(GRU) with feature engineering and data preprocessing techniques to improve accuracy.

# Objective 

The objective of this project is to forecast the closing price of USD/JPY using past data. The model leverages LSTM and GRU networks, which are effective in learning from sequential data. The network is Bidirectional, allowing it to learn from both past and future trends in the time series data.

This model uses various input features, including price data, time-related features (hour, day, week), and engineered features like momentum and average price, to enhance prediction accuracy.

Data Preprocessing
The model uses a dataset containing historical forex data for USD/JPY, where the features are as follows:

timestamp: Date of the observation
open: Opening price
high: Highest price
low: Lowest price
close: Closing price
volume: Volume of trades
vol: Unknown metric, included as additional data
spread: Difference between the bid and ask price
Additional engineered features include:

hour: Hour of the day
day: Day of the week
week: Week number of the year
momentum: Volume * (Open - Close)
avg_price: Average of low and high prices
ohlc_price: Average of Open, High, Low, Close
oc_diff: Difference between Open and Close prices
Data is cleaned by handling missing values, and the numeric values are scaled using MinMaxScaler to fit the model's requirements.

# Model Architectures
The models use the following architecture:

LSTM Model:
Input Layer: Takes the scaled time series data.
Bidirectional LSTM Layer: A Bidirectional LSTM layer with 64 units that learns from both past and future trends.
LSTM Layer: A second LSTM layer with 32 units for further learning.
Dropout Layers: Dropout layers are added to prevent overfitting (0.3 and 0.2).
Dense Layers: Dense layers for final regression mapping, with ReLU activation for intermediate layers and linear activation for the output layer.
The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss.

GRU Model:
GRU Layers (50, 50, 10, 4 units):
These are the "memory" layers that analyze sequences of data (like price movements over time). The GRU layers help the model learn patterns in time-series data, with each successive layer focusing on different aspects of the sequence, starting from more general patterns to more specific ones.
Dropout Layer (0.2):
This layer randomly ignores 20% of neurons during training to prevent overfitting, helping the model generalize better and not memorize the training data.
Dense Layers (4 and 1 unit):
These fully connected layers take the learned features from the GRU layers and combine them into a final output. The first dense layer prepares the data, and the final dense layer gives the model's prediction (such as a price forecast).
In short, the GRU layers process time-based data, Dropout helps avoid overfitting, and Dense layers produce the final prediction.

Training and Evaluation
The models are trained in two phases:

Training the LSTM Model process:
Initial Training: The model is first trained for 200 epochs using a batch size of 1024, with a validation split of 10%. Model checkpoints are saved during training to ensure the best weights are kept.
Fine-tuning: After initial training, the model is fine-tuned for up to 20 epochs, with a learning rate decay applied every 2 epochs.
After training, the model is evaluated using test data and metrics such as:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
Mean Percentage Error
Accuracy (calculated as 100 - mean_percentage_error)
Results
After training, the model's predictions are compared against actual values. Several visualizations are generated, including:

Actual vs Predicted Closing Prices: A line plot showing the comparison.
Prediction Error Distribution: A histogram to visualize prediction errors.
Error vs Predicted Prices: A KDE plot to visualize the error distribution relative to predicted prices.
Accuracy: The model's accuracy is calculated based on the mean percentage error.
The results are printed with the following metrics:

MSE: Mean Squared Error
MAE: Mean Absolute Error
Mean Percentage Error: The average percentage error between predicted and actual closing prices.
Model Accuracy: The accuracy, defined as 100% - mean percentage error.

Training the GRU Model process:
Model Checkpointing: The model saves its best weights (based on validation loss) during training.
Training: The model is trained for 50 epochs, using trainX and trainY for training and testX and testY for validation. It updates weights after every 32 samples.
Loading Best Weights: After training, the model loads the best weights saved during training.
Training History: The final training and validation losses are printed to show the model's performance.


LSTM Model output results:
MSE :  1.48338998745506
MAE :  0.9690638950892857
Mean Percentage Error: 0.63%
Model Accuracy: 99.37%

GRU Model output results:
MSE :  1.3912346332688772
MAE :  0.9548252650669643
Mean Percentage Error: 0.62%
Model Accuracy: 99.38%
