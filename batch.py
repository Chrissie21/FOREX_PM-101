import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional
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