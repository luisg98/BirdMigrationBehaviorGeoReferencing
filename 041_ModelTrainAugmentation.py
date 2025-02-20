import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os



# Load data from CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = data['timestamp'].astype('int64') // 10**9  # Convert to seconds
    return data

# Preprocess data: normalize features and remove NaN values
def preprocess_data(data):
    data = data.dropna(subset=['timestamp', 'latitude', 'longitude'])
    scaler_input = MinMaxScaler(feature_range=(0, 1))
    scaler_output = MinMaxScaler(feature_range=(0, 1))
    
    scaled_inputs = scaler_input.fit_transform(data[['timestamp', 'longitude']])
    scaled_output = scaler_output.fit_transform(data[['latitude']])
    
    return scaled_inputs, scaled_output, scaler_input, scaler_output

# Create sequences for LSTM training
def create_sequences(inputs, outputs, sequence_length=10):
    X, y = [], []
    for i in range(len(inputs) - sequence_length):
        X.append(inputs[i:i + sequence_length])
        y.append(outputs[i + sequence_length])
    return np.array(X), np.array(y)

# Build, train, and save the LSTM model
def train_lstm(X_train, X_val, y_train, y_val, model_path='Model/LSTM/lstm_model.h5'):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Implement Early Stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val), verbose=1, callbacks=[early_stopping])
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print("Model trained and saved successfully!")

# Main function to load data, preprocess, split, and train the model
def main():
    file_path = 'Data/Filtered_Migration_Data.csv'
    data = load_data(file_path)
    scaled_inputs, scaled_output, scaler_input, scaler_output = preprocess_data(data)
    X, y = create_sequences(scaled_inputs, scaled_output)
    
    # Split data into training (70%), validation (20%), and test (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)  # 10% test, 20% validation
    
    train_lstm(X_train, X_val, y_train, y_val)
    print("Dataset split: Training set: {} | Validation set: {} | Test set: {}".format(len(X_train), len(X_val), len(X_test)))

if __name__ == '__main__':
    main()
