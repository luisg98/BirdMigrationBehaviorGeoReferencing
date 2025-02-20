import pandas as pd
import requests
import time
from datetime import datetime
from tqdm import tqdm
import os

def fetch_weather_data(lat, lon, timestamp):
    """Fetches historical weather data from Open-Meteo API for the given timestamp."""
    date_str = timestamp.strftime('%Y-%m-%d')
    hour_str = timestamp.strftime('%H')
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m,precipitation&timezone=auto&start_date={date_str}&end_date={date_str}"
    weather_response = requests.get(weather_url, timeout=100)
    weather_data = weather_response.json()
    
    if weather_response.status_code == 200 and 'hourly' in weather_data:
        times = weather_data['hourly'].get('time', [])
        if f"{date_str}T{hour_str}:00" in times:
            index = times.index(f"{date_str}T{hour_str}:00")
            return {
                "temperature": weather_data['hourly'].get('temperature_2m', [None])[index],
                "wind_speed": weather_data['hourly'].get('wind_speed_10m', [None])[index],
                "precipitation": weather_data['hourly'].get('precipitation', [None])[index]
            }
    
    print("\nWeather API request failed" + weather_response.text)
    return None

def fetch_solar_radiation(lat, lon, timestamp):
    """Fetches solar radiation data from NASA POWER API with hourly parameter."""
    date_str = timestamp.strftime('%Y%m%d')
    hour_str = timestamp.strftime('%H')
    solar_url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN&latitude={lat}&longitude={lon}&start={date_str}&end={date_str}&format=JSON&community=RE"
    solar_response = requests.get(solar_url, timeout=100)
    
    if solar_response.status_code == 200:
        solar_data = solar_response.json()
        if 'properties' in solar_data and 'parameter' in solar_data['properties']:
            return {"solar_radiation": solar_data['properties']['parameter'].get('ALLSKY_SFC_SW_DWN', {}).get(f'{date_str}{hour_str}', None)}
    else:
        print("\nSolar Radiation API request failed" + solar_response.text)
    return None

def enrich_migration_data(file_path, output_file, fetch_weather=True, fetch_solar=True):
    """
    Enriches migration data by obtaining real environmental factors that may influence bird migration.
    Resumes processing from the first row with missing values if interrupted.
    """
    # Load data
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, parse_dates=['timestamp'])
    else:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Ensure necessary columns exist
    columns = ["temperature", "wind_speed", "precipitation", "solar_radiation"]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    
    # Find the first row with missing values to resume processing
    start_index = df[columns].isna().any(axis=1).idxmax()
    
    # Iterate through rows and enrich data
    for index, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df) - start_index, desc="Processing rows"):
        updated = False
        
        try:
            if fetch_weather and pd.isna(row['temperature']):
                weather_data = fetch_weather_data(row['latitude'], row['longitude'], row['timestamp'])
                if weather_data is None:
                    fetch_weather = False
                else: 
                    for key, value in weather_data.items():
                        df.at[index, key] = value
                    updated = True
            
            if fetch_solar and pd.isna(row['solar_radiation']):
                solar_data = fetch_solar_radiation(row['latitude'], row['longitude'], row['timestamp'])
                if solar_data is None:
                    fetch_solar = False
                else:
                    for key, value in solar_data.items():
                        df.at[index, key] = value
                    updated = True
        
        except Exception as e:
            print(f"Error: {e}. Stopping execution.")
            return
        
        # Save dataset after processing each row
        if updated and len(df) == 51150:
            df.to_csv(output_file, index=False)
                    
    # Avoid excessive API requests
     #time.sleep(1)
    
    print(f"Enriched dataset saved as {output_file}")

# Example usage
input_csv = "Data/Filtered_Migration_Data.csv"
output_csv = "Data/Enriched_Migration_Data.csv"
enrich_migration_data(input_csv, output_csv, fetch_weather=True, fetch_solar=True)