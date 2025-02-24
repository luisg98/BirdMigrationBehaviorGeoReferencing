import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from scipy.stats import gaussian_kde
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# Load the dataset
df = pd.read_csv("Data/Filtered_Migration_Data.csv")
print("Dataset loaded successfully.")

# Convert timestamp to datetime and extract year and month
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
print("Timestamp conversion completed.")

# Map month numbers to names
month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
df['month_name'] = df['month'].map(month_dict)

# Define dataset coordinate boundaries
lat_min, lat_max = int(df['latitude'].min()), int(df['latitude'].max())
lon_min, lon_max = int(df['longitude'].min()), int(df['longitude'].max())
print(f"Latitude range: {lat_min} to {lat_max}, Longitude range: {lon_min} to {lon_max}")

# Create a grid of integer coordinates within the limits
lat_range = np.arange(lat_min, lat_max + 1, 1)
lon_range = np.arange(lon_min, lon_max + 1, 1)
grid_coords = np.array(np.meshgrid(lat_range, lon_range)).T.reshape(-1, 2)
grid_df = pd.DataFrame(grid_coords, columns=['Latitude', 'Longitude'])
print("Grid of coordinates created.")

# Optimize Kernel Density Estimation bandwidth
bandwidths = np.logspace(-1, 1, 20)
kde_model = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)
print("KDE bandwidth optimization initialized.")

# Calculate bird density at each coordinate per month using optimized KDE
probabilities = []
saved_models = {}
for month, month_name in month_dict.items():
    print(f"Processing KDE for {month_name}...")
    month_data = df[df['month'] == month][['latitude', 'longitude']].values
    
    if len(month_data) > 1:  # Avoid errors with small samples
        kde_model.fit(month_data)
        best_bandwidth = kde_model.best_params_['bandwidth']
        kde = KernelDensity(bandwidth=best_bandwidth).fit(month_data)
        joblib.dump(kde, f"Model/KDE/kde_model_{month_name}.pkl")  # Save the model
        saved_models[month_name] = kde
        
        log_probs = kde.score_samples(grid_coords)
        probs = np.exp(log_probs)
        probs = (probs / probs.sum()) * 100  # Convert to a scale of 0 to 100
        print(f"Model saved for {month_name} with bandwidth {best_bandwidth}.")
    else:
        probs = np.zeros(len(grid_coords))
        print(f"Insufficient data for {month_name}, skipping KDE model training.")
    
    probabilities.append(probs)

# Create DataFrame with probabilities
prob_df = pd.DataFrame(np.array(probabilities).T, columns=[month_dict[m] for m in range(1, 13)])
final_df = pd.concat([grid_df, prob_df], axis=1)
print("Probability DataFrame created.")

# Transform data for Plotly visualization
final_long_df = final_df.melt(id_vars=['Latitude', 'Longitude'], var_name='Month', value_name='Probability')
print("Data transformed for visualization.")

# Create an interactive map with a dropdown for month selection
fig = px.density_mapbox(
    final_long_df,
    lat='Latitude',
    lon='Longitude',
    z='Probability',
    radius=9,
    animation_frame='Month',
    color_continuous_scale="YlOrRd",
    mapbox_style="carto-positron",
    title="Bird Migration Probability Heatmap"
)

# Adjust zoom to view the entire world map
fig.update_layout(
    mapbox=dict(
        center=dict(lat=0, lon=0),
        zoom=1  # Set for wide visualization
    ),
    height=900  # Increase map window size
)

# Display the map
fig.show()
fig.savefig("Results/Heatmap.png")
print("Heatmap visualization displayed.")

print("All KDE models saved in the 'Models/' folder.")
