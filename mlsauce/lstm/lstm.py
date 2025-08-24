# import numpy as np
# import pandas as pd
# try:
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import LSTM, Dense, Dropout
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.callbacks import EarlyStopping
# except ImportError:
#     raise ImportError("TensorFlow is required for LSTMRegressor. Please install it manually using 'pip install tensorflow'.")
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.base import BaseEstimator, RegressorMixin
# import warnings

# warnings.filterwarnings('ignore')

# class LSTMRegressor(BaseEstimator, RegressorMixin):
#     """
#     LSTM Regressor for time series forecasting.

#     This regressor learns to predict the next value in a sequence based on historical values.

#     Parameters:
#     -----------
#     sequence_length : int, default=10
#         Number of time steps to look back
#     lstm_units : int, default=50
#         Number of LSTM units
#     dropout_rate : float, default=0.2
#         Dropout rate
#     learning_rate : float, default=0.001
#         Learning rate
#     epochs : int, default=50
#         Training epochs
#     batch_size : int, default=32
#         Batch size
#     verbose : int, default=0
#         Verbosity
#     random_state : int, default=42
#         Random seed
#     """

#     def __init__(self, sequence_length=10, lstm_units=50, dropout_rate=0.2,
#                  learning_rate=0.001, epochs=50, batch_size=32, verbose=0, random_state=42):
#         self.sequence_length = sequence_length
#         self.lstm_units = lstm_units
#         self.dropout_rate = dropout_rate
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.verbose = verbose
#         self.random_state = random_state

#         # Internal state
#         self.model = None
#         self.scaler = None
#         self.xreg_scaler = None  # For scaling external regressors
#         self.last_sequence = None
#         self.is_fitted = False
#         self.training_data = None

#         # Set random seeds
#         np.random.seed(random_state)
#         tf.random.set_seed(random_state)

#     def _create_sequences(self, data):
#         """Create sequences for LSTM training"""
#         sequences_X = []
#         sequences_y = []

#         for i in range(len(data) - self.sequence_length):
#             sequences_X.append(data[i:(i + self.sequence_length)])
#             sequences_y.append(data[i + self.sequence_length])

#         return np.array(sequences_X), np.array(sequences_y)

#     def fit(self, X, xreg=None, **kwargs):
#         """
#         Fit MTS model to training data X, with optional regressors xreg

#         X: {array-like}, shape = [n_samples, n_features]
#             Training time series.

#         xreg: {array-like}, shape = [n_samples, n_features_xreg]
#             Optional external regressors.
#         """
#         # Convert and handle input data
#         X = np.array(X, dtype=np.float32)

#         if X.ndim == 1:
#             X = X.reshape(-1, 1)

#         # Handle external regressors if provided
#         if xreg is not None:
#             xreg = np.array(xreg, dtype=np.float32)
#             if xreg.ndim == 1:
#                 xreg = xreg.reshape(-1, 1)
#             self.xreg_scaler = MinMaxScaler()
#             xreg = self.xreg_scaler.fit_transform(xreg)  # Scale the external regressors
#             self.training_data = np.column_stack([X, xreg])
#         else:
#             self.training_data = X.copy()

#         # Check minimum data requirements
#         if len(self.training_data) <= self.sequence_length + 1:
#             # Fallback for very small datasets
#             self.mean_target = np.mean(X[:, -1]) if X.shape[1] > 1 else np.mean(X)
#             self.last_sequence = self.training_data[-self.sequence_length:] if len(self.training_data) >= self.sequence_length else self.training_data
#             self.is_fitted = True
#             return self

#         # Scale the data
#         self.scaler = MinMaxScaler()
#         scaled_data = self.scaler.fit_transform(self.training_data)

#         # Create sequences for training
#         X_seq, y_seq = self._create_sequences(scaled_data)

#         if len(X_seq) == 0:
#             self.mean_target = np.mean(X[:, -1]) if X.shape[1] > 1 else np.mean(X)
#             self.last_sequence = scaled_data[-self.sequence_length:]
#             self.is_fitted = True
#             return self

#         # Store the last sequence for future predictions
#         self.last_sequence = scaled_data[-self.sequence_length:]

#         # Build LSTM model
#         n_features = scaled_data.shape[1]
#         self.model = Sequential([
#             LSTM(self.lstm_units, input_shape=(self.sequence_length, n_features), return_sequences=True),
#             Dropout(self.dropout_rate),
#             LSTM(self.lstm_units),
#             Dropout(self.dropout_rate),
#             Dense(1)
#         ])

#         self.model.compile(
#             optimizer=Adam(learning_rate=self.learning_rate),
#             loss='mse'
#         )

#         # Reshape sequences for LSTM input
#         X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], n_features)

#         # Train model
#         callbacks = [EarlyStopping(patience=10, restore_best_weights=True, min_delta=1e-6)]

#         self.model.fit(
#             X_seq, y_seq,
#             epochs=self.epochs,
#             batch_size=min(self.batch_size, len(X_seq)),
#             validation_split=0.2 if len(X_seq) > 10 else 0,
#             callbacks=callbacks if len(X_seq) > 10 else [],
#             verbose=self.verbose
#         )

#         self.is_fitted = True
#         return self

#     def predict(self, h=5, **kwargs):
#         """
#         Forecast all the time series, h steps ahead
#         """
#         if not self.is_fitted:
#             raise ValueError("Model must be fitted first")

#         # Handle fallback case
#         if not hasattr(self, 'model') or self.model is None:
#             return np.full(h, getattr(self, 'mean_target', 0.0))

#         predictions = []
#         current_sequence = self.last_sequence.copy()

#         # Generate predictions step by step
#         for _ in range(h):
#             input_seq = current_sequence.reshape(1, len(current_sequence), current_sequence.shape[1])
#             pred_scaled = self.model.predict(input_seq, verbose=0)
#             predictions.append(pred_scaled[0])

#             # Update sequence for next prediction
#             current_sequence = np.roll(current_sequence, -1, axis=0)
#             current_sequence[-1] = pred_scaled[0]

#         # Convert predictions to numpy array
#         predictions = np.array(predictions)

#         # Inverse transform predictions
#         if self.scaler is not None:
#             predictions_reshaped = predictions.reshape(-1, 1)
#             predictions = self.scaler.inverse_transform(predictions_reshaped).ravel()

#         return predictions

#     def score(self, X, y):
#         """
#         Calculate R² score using walk-forward validation
#         """
#         if not self.is_fitted:
#             return 0.0

#         # Use the last part of training for scoring
#         n_test = min(len(y), 20)  # Test on last 20 points
#         predictions = self.predict(h=n_test)

#         # Compare with actual values
#         if len(predictions) >= len(y):
#             predictions = predictions[:len(y)]

#         from sklearn.metrics import r2_score
#         try:
#             return r2_score(y[-len(predictions):], predictions)
#         except Exception as e:
#             print(f"Error in scoring: {e}")
#             return 0.0

#     def save_model(self, path):
#         """
#         Save the trained model to a file.
#         """
#         if self.model is not None:
#             self.model.save(path)
#             print(f"Model saved to {path}")
#         else:
#             print("Model is not trained yet.")

#     def load_model(self, path):
#         """
#         Load a pre-trained model from a file.
#         """
#         from tensorflow.keras.models import load_model
#         self.model = load_model(path)
#         print(f"Model loaded from {path}")
