import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import pickle

# Load expected feature names from features.pkl
with open('Model/XGB/features.pkl', 'rb') as file:
    expected_features = pickle.load(file)  

# Define sequence length (e.g., 10 timestamps)
SEQUENCE_LENGTH = len(expected_features) // 2  # Since each step has 2 features

# Function to impute missing latitude values using XGBoost
def impute_latitude(data):
    # Preserve original column names
    original_columns = data.columns.tolist()

    # Load the trained XGBoost model
    model = XGBRegressor()
    model.load_model('Model/XGB/xgboost_model.json')

    print(f"Using expected features: {expected_features}")

    # Ensure required columns exist
    required_columns = ['timestamp', 'location-long', 'location-lat']
    if not set(required_columns).issubset(data.columns):
        raise KeyError(f"Dataset is missing required columns: {required_columns}")

    # Store original timestamp format for later
    data['timestamp_original'] = data['timestamp']

    # Convert timestamp to UNIX time for model input
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp_unix'] = data['timestamp'].astype(int) // 10**9  

    # Identify rows with missing latitude
    missing_lat_index = data['location-lat'].isna()
    if missing_lat_index.sum() == 0:
        print(" No missing latitude values found. Nothing to predict.")
        return

    # Add 'synthetic' column (0 for existing values, 1 for generated values)
    if 'synthetic' not in data.columns:
        data['synthetic'] = 0  # Default to 0 (not synthetic)

    # Sort dataset by timestamp to ensure correct temporal ordering
    data = data.sort_values(by='timestamp_unix')

    # Select required features for sequence creation
    input_features = ['timestamp_unix', 'location-long']
    data_filtered = data[input_features].copy()

    # Normalize the input data
    scaler_input = MinMaxScaler(feature_range=(0, 1))
    scaler_lat = MinMaxScaler(feature_range=(0, 1))

    # Fit normalizers using known latitude values
    scaler_input.fit(data_filtered)
    scaler_lat.fit(data[['location-lat']].dropna())

    # Scale dataset
    data_filtered_scaled = pd.DataFrame(scaler_input.transform(data_filtered), columns=input_features)
    data['location-lat-scaled'] = np.nan  # Temporary column for scaled latitude
    data.loc[~missing_lat_index, 'location-lat-scaled'] = scaler_lat.transform(data[['location-lat']].dropna()).flatten()

    # Create sequences for missing latitude values
    missing_indices = data.index[missing_lat_index]
    predicted_latitudes_scaled = []

    for idx in missing_indices:
        if idx < SEQUENCE_LENGTH:
            predicted_latitudes_scaled.append(np.nan)  # Cannot predict if no full sequence exists
            continue

        sequence = data_filtered_scaled.iloc[idx - SEQUENCE_LENGTH:idx].values.flatten().reshape(1, -1)
        predicted_lat_scaled = model.predict(sequence)[0]
        predicted_latitudes_scaled.append(predicted_lat_scaled)

    # Convert predicted latitudes back to the original scale
    predicted_latitudes = scaler_lat.inverse_transform(np.array(predicted_latitudes_scaled).reshape(-1, 1)).flatten()

    # Fill in missing latitude values
    data.loc[missing_lat_index, 'location-lat'] = predicted_latitudes
    data.loc[missing_lat_index, 'synthetic'] = 1  # Mark predicted values as synthetic

    # Restore original timestamp format and remove temporary columns
    data['timestamp'] = data['timestamp_original']
    data.drop(columns=['timestamp_unix', 'location-lat-scaled', 'timestamp_original'], inplace=True, errors='ignore')

    # Save the final dataset (keeping the original timestamp format)
    data.to_csv('Data/AugmentedData.csv', index=False, encoding='utf-8')
    print(" Missing latitude values imputed, descaled, and dataset saved successfully!")

# Load dataset and run imputation
data = pd.read_csv('Data/DataWithUnknownLatitude.csv')
impute_latitude(data)
