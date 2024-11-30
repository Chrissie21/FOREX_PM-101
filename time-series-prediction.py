import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
# Remove: `from keras.layers.wrappers import *` Deprecated

from config import *

# Load the CSV file, assuming tab-separated values
df = pd.read_csv("USDJPY.csv", sep='\t')

# Print the column names and first few rows of the data for inspection
print("Columns in the CSV file:", df.columns)
print("First few rows of the data:\n", df.head())

# Rename the columns to meaningful names
df.rename(columns={'<DATE>': 'timestamp', 
                   '<OPEN>': 'open', 
                   '<HIGH>': 'high', 
                   '<LOW>': 'low', 
                   '<CLOSE>': 'close', 
                   '<TICKVOL>': 'volume', 
                   '<VOL>': 'vol', 
                   '<SPREAD>': 'spread'}, inplace=True)

# Print out columns to verify the renaming was successful
print("Columns after renaming:", df.columns)

# Determine batch size
num_samples = df.shape[0]
print(f"Total number of samples in the dataset: {num_samples}")

# Convert the 'timestamp' column to datetime and set it as the index
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y.%m.%d')
df.set_index('timestamp', inplace=True)

# Convert all other columns to numeric (useful for columns like volume, spread, etc.)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values (in case conversion to numeric introduced NaNs)
df.dropna(inplace=True)

# Add additional features based on the time and price data
df['hour'] = df.index.hour
df['day'] = df.index.weekday
df['week'] = df.index.isocalendar().week
df['momentum'] = df['volume'] * (df['open'] - df['close'])
df['avg_price'] = (df['low'] + df['high']) / 2
df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
df['oc_diff'] = df['open'] - df['close']

# Print the first few rows to see the additional features
print("Data after feature engineering:\n", df.head())

# Function to create dataset for model input (Creating X & Y)
def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# Scale and create datasets
target_index = df.columns.tolist().index('close')
high_index = df.columns.tolist().index('high')
low_index = df.columns.tolist().index('low')
dataset = df.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
y_scaler = MinMaxScaler(feature_range=(0, 1))
t_y = df['close'].values.astype('float32')
t_y = np.reshape(t_y, (-1, 1))
y_scaler = y_scaler.fit(t_y)

X, y = create_dataset(dataset, look_back=30)
y = y[:, target_index]

train_size = int(len(X) * 0.99)
trainX = X[:train_size]
trainY = y[:train_size]
testX = X[train_size:]
testY = y[train_size:]

# Create an improved LSTM network
model = Sequential()

# Add an explicit Input layer
model.add(Input(shape=(X.shape[1], X.shape[2])))

# First Bidirectional LSTM layer
model.add(Bidirectional(
    LSTM(64, return_sequences=True),
    merge_mode='sum'
))

# Second LSTM layer with reduced units
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.3))  # Increased dropout to reduce overfitting

# Third LSTM layer
model.add(LSTM(16, return_sequences=False))

# Dense layers for output mapping
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))  # Dropout for regularization
model.add(Dense(1, activation='linear'))  # Use 'linear' activation for regression

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])

# Print the model summary
print(model.summary())


def train():
    # Save the best weights during training
    checkpoint = ModelCheckpoint("bi_rnn_weights.keras",
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min')

    callbacks_list = [checkpoint]
    
    # Train the model
    history = model.fit(trainX, trainY, epochs=200, batch_size=1024, verbose=1,
                        callbacks=callbacks_list, validation_split=0.1)
    
    # Plot training and validation metrics
    for k in list(history.history.keys()):
        plt.figure(figsize=(40, 10))
        plt.plot(history.history[k], label=f'Training {k}')
        if f'val_{k}' in history.history:
            plt.plot(history.history[f'val_{k}'], label=f'Validation {k}')
        plt.title(k)
        plt.ylabel(k)
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()

    # Check if the weights file exists before loading it
    if os.path.exists("bi_rnn_weights.keras"):
        model.load_weights("bi_rnn_weights.keras")
        print("Loaded best weights from initial training phase.")
    else:
        print("No pre-trained weights found. Proceeding without loading weights.")

    # Learning rate scheduler
    def scheduler(epoch, lr):
        if epoch % 2 == 0 and epoch != 0:
            new_lr = lr * 0.9  # Decay the learning rate
            print(f"Learning rate changed to: {new_lr}")
            return new_lr  # Return the updated learning rate as a float
        return lr  # If no change, return the current learning rate

    lr_decay = LearningRateScheduler(scheduler)

    # Checkpoint for saving the best weights during fine-tuning
    fine_tune_checkpoint = ModelCheckpoint("bi_rnn_weights_finetune.keras",
                                            monitor='val_mean_squared_error',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='min')

    # Fine-tuning epochs
    fine_tune_epochs = int(min(len(history.history['loss']) / 3, 20))  # Adjusted epochs

    # Fine-tune the model
    history = model.fit(
        trainX,
        trainY,
        epochs=fine_tune_epochs,
        batch_size=500,
        verbose=1,
        callbacks=[fine_tune_checkpoint, lr_decay],
        validation_split=0.1
    )

    # Plot training and validation metrics for fine-tuning phase
    for k in list(history.history.keys()):
        plt.figure(figsize=(40, 10))
        plt.plot(history.history[k], label=f'Training {k}')
        if f'val_{k}' in history.history:
            plt.plot(history.history[f'val_{k}'], label=f'Validation {k}')
        plt.title(k)
        plt.ylabel(k)
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()



def benchmark():
    # Benchmark
    model.load_weights("bi_rnn_weights.keras")

    # Make predictions
    pred = model.predict(testX)
    pred = y_scaler.inverse_transform(pred)  # Inverse scale predictions
    close = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))  # Inverse scale actual values

    # Prepare predictions DataFrame
    predictions = pd.DataFrame()
    predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
    predictions['close'] = pd.Series(np.reshape(close, (close.shape[0])))

    # Slice 'p' from df and ensure it has the correct datetime index
    p = df[-pred.shape[0]:].copy()  # Slice the last rows to match the prediction shape
    print(p.index)  # Verify it has the correct datetime index

    # Assign 'p's index to predictions
    predictions.index = p.index

    # Ensure predictions contains the necessary columns
    predictions = predictions.astype(float)
    predictions = predictions.merge(p[['low', 'high']], right_index=True, left_index=True)

    # Plotting
    # Plot actual vs. predicted values
    plt.figure(figsize=(16, 8))
    plt.plot(predictions.index, predictions['close'], label='Actual Close', color='red', linewidth=2)
    plt.plot(predictions.index, predictions['predicted'], label='Predicted Close', color='blue', linestyle='--', linewidth=2)
    plt.fill_between(predictions.index, predictions['low'], predictions['high'], color='lightblue', alpha=0.3, label='Low-High Range')

    plt.title('Actual vs Predicted Close Prices', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.show()

    # Other visualizations
    # Add a new column for the difference between predicted and actual values
    predictions['diff'] = predictions['predicted'] - predictions['close']

    # Plot the distribution of differences
    plt.figure(figsize=(10, 10))
    sns.histplot(predictions['diff'], kde=True)
    plt.title('Distribution of Differences Between Actual and Prediction')
    plt.show()

    # Plot a KDE distribution of error vs predicted prices
    sns.jointplot(x="diff", y="predicted", data=predictions, kind="kde", space=0)
    plt.suptitle('Distribution of Error and Price', y=1.02)
    plt.show()

    predictions['correct'] = (predictions['predicted'] <= predictions['high']) & (predictions['predicted'] >= predictions['low'])
    sns.catplot(data=predictions, x='correct', kind='count')

    print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['close'].values))
    print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['close'].values))
    predictions['diff'].describe()

    # Calculate percentage error
    predictions['percentage_error'] = (abs(predictions['predicted'] - predictions['close']) / predictions['close']) * 100

    # Calculate mean percentage error
    mean_percentage_error = predictions['percentage_error'].mean()

    # Calculate accuracy (as 100% - mean percentage error)
    accuracy = 100 - mean_percentage_error

    # Print percentage error and accuracy
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    print(f"Model Accuracy: {accuracy:.2f}%")

    # Optional: Plot the distribution of percentage errors
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions['percentage_error'], kde=True, bins=20)
    plt.title('Distribution of Percentage Error')
    plt.show()


train()
benchmark()
