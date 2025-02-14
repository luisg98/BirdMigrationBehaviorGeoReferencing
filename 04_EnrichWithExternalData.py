import pandas as pd
import requests
import time
from datetime import datetime
from tqdm import tqdm

def fetch_weather_data(lat, lon):
    """Fetches weather data from Open-Meteo API."""
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m,precipitation&timezone=auto"
    weather_response = requests.get(weather_url, timeout=100)
    weather_data = weather_response.json()
    
    if 'hourly' in weather_data:
        return {
            "temperature": weather_data['hourly'].get('temperature_2m', [None])[0],
            "wind_speed": weather_data['hourly'].get('wind_speed_10m', [None])[0],
            "precipitation": weather_data['hourly'].get('precipitation', [None])[0]
        }
    return {}

def fetch_solar_radiation(lat, lon, timestamp):
    """Fetches solar radiation data from NASA POWER API."""
    date_str = timestamp.strftime('%Y%m%d')
    solar_url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN&latitude={lat}&longitude={lon}&start={date_str}&end={date_str}&format=JSON"
    solar_response = requests.get(solar_url, timeout=100)
    solar_data = solar_response.json()
    if 'properties' in solar_data and 'parameter' in solar_data['properties']:
        return {"solar_radiation": solar_data['properties']['parameter'].get('ALLSKY_SFC_SW_DWN', {}).get(f'{date_str}00', None)}
    return {}

def fetch_proximity_to_parks(lat, lon):
    """Fetches proximity to natural parks from OpenStreetMap Overpass API."""
    parks_url = f"https://overpass-api.de/api/interpreter?data=[out:json];node[natural=park](around:5000,{lat},{lon});out;"
    parks_response = requests.get(parks_url, timeout=100)
    if parks_response.status_code == 200:
        parks_data = parks_response.json()
        return {"proximity_to_natural_parks": len(parks_data.get('elements', []))}
    return {}

def enrich_migration_data(file_path, output_file, fetch_weather=True, fetch_solar=True, fetch_parks=True):
    """
    Enriches migration data by obtaining real environmental factors that may influence bird migration.
    Allows selecting which APIs to use and fetches only missing values.
    Saves the dataset after processing each row to prevent data loss in case of interruption.
    
    :param file_path: Path to the cleaned dataset CSV file
    :param output_file: Path to save the enriched dataset
    :param fetch_weather: Whether to fetch weather data
    :param fetch_solar: Whether to fetch solar radiation data
    :param fetch_parks: Whether to fetch proximity to parks data
    """
    # Load dataset
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Initialize new columns if not present
    columns = ["temperature", "wind_speed", "precipitation", "solar_radiation", "proximity_to_natural_parks"]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    
    # Iterate through rows and enrich data with progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        updated = False
        
        if fetch_weather and pd.isna(row['temperature']):
            weather_data = fetch_weather_data(row['latitude'], row['longitude'])
            for key, value in weather_data.items():
                df.at[index, key] = value
            updated = True
        
        if fetch_solar and pd.isna(row['solar_radiation']):
            solar_data = fetch_solar_radiation(row['latitude'], row['longitude'], row['timestamp'])
            for key, value in solar_data.items():
                df.at[index, key] = value
            updated = True
        
        if fetch_parks and pd.isna(row['proximity_to_natural_parks']):
            parks_data = fetch_proximity_to_parks(row['latitude'], row['longitude'])
            for key, value in parks_data.items():
                df.at[index, key] = value
            updated = True
        
        # Save dataset after processing each row
        if updated:
            df.to_csv(output_file, index=False)
        
        # Avoid excessive API requests
        time.sleep(1)
    
    print(f"Enriched dataset saved as {output_file}")

# Example usage
input_csv = "Data/Filtered_Migration_Data.csv"
output_csv = "Data/Enriched_Migration_Data.csv"
enrich_migration_data(input_csv, output_csv, fetch_weather=True, fetch_solar=False, fetch_parks=False)
