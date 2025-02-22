import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = data['timestamp'].astype(int) // 10**9  # Convert to UNIX timestamp
    return data

# Function to create time sequences for training
def create_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])  # Last 'sequence_length' values
        y.append(data[i + sequence_length, -1])  # Corresponding latitude value
    return np.array(X), np.array(y)

# Function to train XGBoost with early stopping
def train_xgboost(X_train, X_val, y_train, y_val, model_path, features_path, feature_names):
    # Convert training and validation data to DataFrame with appropriate feature names
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_val = pd.DataFrame(X_val, columns=feature_names)

    # Define and train the XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=2000, learning_rate=0.05, random_state=42, early_stopping_rounds=10)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=True, 
    )

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)

    # Save the feature names used for training
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)

    return model

# Path to dataset (modify as needed)
file_path = 'Data/Filtered_Migration_Data.csv'

# Load dataset
data = load_data(file_path)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['timestamp', 'longitude', 'latitude']])

# Split dataset for XGBoost WITHOUT sequences (only timestamp and longitude)
X_no_seq = scaled_data[:, :-1]  # Only timestamp and longitude
y_no_seq = scaled_data[:, -1]  # Latitude

# Split into train (70%), validation (20%), and test (10%)
X_train_no_seq, X_temp_no_seq, y_train_no_seq, y_temp_no_seq = train_test_split(X_no_seq, y_no_seq, test_size=0.3, random_state=42)
X_val_no_seq, X_test_no_seq, y_val_no_seq, y_test_no_seq = train_test_split(X_temp_no_seq, y_temp_no_seq, test_size=0.33, random_state=42)

# Split dataset for XGBoost WITH time sequences (10 previous time steps as input)
sequence_length = 10
X_seq, y_seq = create_sequences(scaled_data)

# Split into train (70%), validation (20%), and test (10%)
X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(X_temp_seq, y_temp_seq, test_size=0.33, random_state=42)

# Reshape X_train_seq for XGBoost (convert from 3D to 2D)
X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], X_train_seq.shape[1] * X_train_seq.shape[2])
X_val_seq = X_val_seq.reshape(X_val_seq.shape[0], X_val_seq.shape[1] * X_val_seq.shape[2])
X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1] * X_test_seq.shape[2])

# Create feature names for XGBoost with sequences
feature_names_seq = [f"time_lag_{i}_timestamp" for i in range(sequence_length)] + \
                    [f"time_lag_{i}_longitude" for i in range(sequence_length)]

# Train the models
model_no_seq = train_xgboost(
    X_train_no_seq, X_val_no_seq, y_train_no_seq, y_val_no_seq,
    model_path="Models/XGB/xgboost_no_sequence.json",
    features_path="Models/XGB/features_no_sequence.pkl",
    feature_names=['timestamp', 'longitude']
)

model_seq = train_xgboost(
    X_train_seq, X_val_seq, y_train_seq, y_val_seq,
    model_path="Models/XGB/xgboost_with_sequence.json",
    features_path="Models/XGB/features_with_sequence.pkl",
    feature_names=feature_names_seq
)

# Evaluate models on test set
y_pred_no_seq = model_no_seq.predict(X_test_no_seq)
y_pred_seq = model_seq.predict(X_test_seq)

# Compute evaluation metrics
mse_no_seq = mean_squared_error(y_test_no_seq, y_pred_no_seq)
mae_no_seq = mean_absolute_error(y_test_no_seq, y_pred_no_seq)

mse_seq = mean_squared_error(y_test_seq, y_pred_seq)
mae_seq = mean_absolute_error(y_test_seq, y_pred_seq)

# Print test results
print("\nTest Results:")
print(f"XGBoost WITHOUT Sequence:  MSE = {mse_no_seq:.5f}, MAE = {mae_no_seq:.5f}")
print(f"XGBoost WITH Sequence:  MSE = {mse_seq:.5f}, MAE = {mae_seq:.5f}")
