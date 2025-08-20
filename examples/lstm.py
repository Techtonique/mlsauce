import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import mlsauce as ms  # Assuming your class is in a separate module

# Load the same dataset
url = "https://raw.githubusercontent.com/Techtonique/"
url += "datasets/main/time_series/multivariate/"
url += "ice_cream_vs_heater.csv"
df_temp = pd.read_csv(url)
df_temp.index = pd.DatetimeIndex(df_temp.date)
df_icecream = df_temp.drop(columns=['date']).diff().dropna()

# Create and fit the LSTM model
lstm_regressor = ms.LSTMRegressor(
    sequence_length=20,  # Matching the lags=20 in the example
    lstm_units=50,
    dropout_rate=0.2,
    learning_rate=0.001,
    epochs=100,
    batch_size=32,
    verbose=1,
    random_state=42
)

# Fit the model (note we're using just the 'heater' column as target)
target_col = 'heater'
lstm_regressor.fit(df_icecream[target_col].values.reshape(-1, 1))

# Make predictions
h = 30  # Forecast horizon
predictions = lstm_regressor.predict(h=h)

# Create a plot similar to nnetsauce.MTS
def plot_predictions(actual, predictions, target_col, h):
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(actual.index, actual[target_col], label='Actual', color='blue')
    
    # Create future dates for predictions
    last_date = actual.index[-1]
    freq = pd.infer_freq(actual.index)
    future_dates = pd.date_range(
        start=last_date, 
        periods=h+1,  # +1 because we include the last actual point
        freq=freq
    )[1:]  # Exclude the last actual point
    
    # Plot predictions
    plt.plot(future_dates, predictions, label='LSTM Forecast', color='red', linestyle='--')
    
    plt.title(f'LSTM Forecast for {target_col}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_predictions(df_icecream, predictions, target_col, h)

# If you want to test multiple targets like in the original example:
for target_col in ['heater', 'icecream']:
    print(f"\nRunning LSTM for {target_col}")
    
    # Reinitialize model for each target
    lstm_regressor = ms.LSTMRegressor(
        sequence_length=20,
        lstm_units=50,
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        verbose=0,
        random_state=42
    )
    
    try:
        lstm_regressor.fit(df_icecream[target_col].values.reshape(-1, 1))
        predictions = lstm_regressor.predict(h=30)
        plot_predictions(df_icecream, predictions, target_col, 30)
    except Exception as e:
        print(f"Error with {target_col}: {e}")
        continue