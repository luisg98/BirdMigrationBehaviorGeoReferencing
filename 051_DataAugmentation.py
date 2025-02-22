import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Function to fill missing latitude values and restore other features
def impute_latitude(data):
    # Save the original column names to reintegrate later
    original_columns = data.columns

    # Load the pre-trained LSTM model
    model = load_model('Model/LSTM/lstm_model.h5')

    # Drop rows where 'timestamp' or 'location-long' are null
    data = data.dropna(subset=['timestamp', 'location-long']).copy()

    # Create a new DataFrame containing only relevant columns for prediction
    data_filtered = data[['timestamp', 'location-lat', 'location-long']].copy()

    # Add a column to mark synthetic (imputed) values
    data['synthetic'] = 0  
    data_filtered['synthetic'] = 0  

    # Convert timestamp to numeric format (seconds since epoch)
    data_filtered['timestamp'] = pd.to_datetime(data_filtered['timestamp'])
    data_filtered['timestamp'] = data_filtered['timestamp'].astype('int64') // 10**9  

    # Separate known and unknown latitude values
    known_lat = data_filtered.dropna(subset=['location-lat']).copy()
    unknown_lat = data_filtered[data_filtered['location-lat'].isna()].copy()

    # Create separate scalers for input features and latitude
    scaler_input = MinMaxScaler(feature_range=(0, 1))  # For timestamp and location-long
    scaler_lat = MinMaxScaler(feature_range=(0, 1))  # For location-lat

    # Fit the scalers using only known latitude values
    scaler_input.fit(known_lat[['timestamp', 'location-long']])
    scaler_lat.fit(known_lat[['location-lat']])  

    # Normalize input data (timestamp and longitude)
    unknown_lat_scaled = scaler_input.transform(unknown_lat[['timestamp', 'location-long']])

    # Predict and replace missing latitude values
    for i, row in unknown_lat.iterrows():
        X_single = unknown_lat_scaled[unknown_lat.index.get_loc(i)].reshape(1, 1, -1)

        print(f"Predicting latitude for row {i} with normalized data: {X_single}")

        # Predict normalized latitude
        predicted_latitude = model.predict(X_single)
        predicted_value_normalized = predicted_latitude[0, 0]

        # Denormalize the prediction
        predicted_value = scaler_lat.inverse_transform([[predicted_value_normalized]])[0, 0]

        # Validate the prediction
        if not pd.isna(predicted_value) and np.isfinite(predicted_value):
            # Replace only null values with the predicted latitude
            data_filtered.at[i, 'location-lat'] = predicted_value
            data_filtered.at[i, 'synthetic'] = 1
            print(f"Predicted latitude for row {i} (denormalized): {predicted_value}")

    # Restore the updated latitude values into the original dataset
    data.update(data_filtered[['location-lat', 'synthetic']])

    # Save the final dataset
    data.to_csv('Data/AugmentedData.csv', index=False, encoding='utf-8')
    print("Latitude values filled and dataset saved successfully!")

# Load the dataset and run the imputation
data = pd.read_csv('Data/DataWithUnkownLatitude.csv')
impute_latitude(data)
