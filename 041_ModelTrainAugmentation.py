import pandas as pd
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pickle

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = data['timestamp'].astype(int) // 10**9  # Convert to UNIX timestamp
    return data

# Preprocess data
def preprocess_data(data):
    data = data.dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['timestamp', 'longitude', 'latitude']])
    return scaled_data, scaler

# Create sequences for LSTM
def create_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])
        y.append(data[i + sequence_length, -1])
    return np.array(X), np.array(y)

# Train LSTM model
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model

# Train XGBoost
def train_xgboost(X_train, X_val, y_train, y_val, model_path='Model/XGB/xgboost_model.json'):
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    return model

# Train MLP model
def train_mlp(X_train, X_val, y_train, y_val, model_path='Model/MLP/mlp_model.pkl'):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.01, solver='adam', random_state=42, early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, verbose=True)
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, 'wb'))
    return model

# Define GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
# Train GNN model
def train_gnn(X_train, y_train, model_path='Model/GNN/gnn_model.pth'):
    num_nodes = X_train.shape[0]
    edge_index = torch.randint(0, num_nodes, (2, num_nodes), dtype=torch.long)
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    model = GNN(input_dim=X_train.shape[1], hidden_dim=16, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 5
    no_improve = 0
    loss_values = []
    
    for epoch in range(500):
        optimizer.zero_grad()
        output = model(X_train_torch, edge_index)
        loss = loss_fn(output, y_train_torch)
        loss.backward()
        optimizer.step()
        
        loss_values.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model

# Model evaluation function
def evaluate_model(model, X_test, y_test, model_type):
    if model_type == 'lstm':
        predictions = model.predict(X_test)
    elif model_type == 'gnn':
        X_test_torch = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)
        edge_index = torch.randint(0, X_test.shape[0], (2, X_test.shape[0]), dtype=torch.long)

        model.eval()
        with torch.no_grad():
            predictions = model(X_test_torch, edge_index).squeeze().numpy()
    else:
        predictions = model.predict(X_test.reshape(X_test.shape[0], -1))
    
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    return mse, mae

# Main function
def main():
    file_path = 'Data/Filtered_Migration_Data.csv'
    data = load_data(file_path)
    scaled_data, scaler = preprocess_data(data)

    # Split data
    X, y = create_sequences(scaled_data)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Train models
    models = {
        "LSTM": train_lstm(X_train, X_val, y_train, y_val),
        "XGBoost": train_xgboost(X_train.reshape(X_train.shape[0], -1), X_val.reshape(X_val.shape[0], -1), y_train, y_val),
        "MLP": train_mlp(X_train.reshape(X_train.shape[0], -1), X_val.reshape(X_val.shape[0], -1), y_train, y_val),
        "GNN": train_gnn(X_train.reshape(X_train.shape[0], -1), y_train),
    }

    # Evaluate models
    results = {name: evaluate_model(model, X_test, y_test, name.lower()) for name, model in models.items()}
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MSE', 'MAE'])
    print(results_df)

if __name__ == '__main__':
    main()
