import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Load the original CSV file
file_path = "Data/Filtered_Migration_Data.csv"
df = pd.read_csv(file_path)

# Define a one-year period
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2024-12-31")

# Generate timestamps twice a day
timestamps = pd.date_range(start=start_date, end=end_date, freq='24h')

# Create multiple synthetic bird IDs
bird_ids = [99999, 88888]
synthetic_data_all = []

# Model 1: Optimized Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=8, covariance_type='full', max_iter=500, random_state=42)
gmm.fit(df[['longitude', 'latitude']])
synthetic_coords_gmm, _ = gmm.sample(len(timestamps))
synthetic_data_all.append(pd.DataFrame({
    "timestamp": timestamps,
    "longitude": synthetic_coords_gmm[:, 0],
    "latitude": synthetic_coords_gmm[:, 1],
    "bird_id": bird_ids[0]
}))

# Model 2: Enhanced Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

vae = VAE(input_dim=2)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
criterion = nn.MSELoss()
data_tensor = torch.tensor(df[['longitude', 'latitude']].values, dtype=torch.float32)

# Enhanced VAE training
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    recon = vae(data_tensor)
    loss = criterion(recon, data_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

mu, logvar = vae.encode(data_tensor)
z = vae.reparameterize(mu, logvar)
generated_data = vae.decode(z).detach().numpy()

# Adjust the number of timestamps to match the generated data
num_samples = min(len(timestamps), len(generated_data))
synthetic_data_all.append(pd.DataFrame({
    "timestamp": timestamps[:num_samples],
    "longitude": generated_data[:num_samples, 0],
    "latitude": generated_data[:num_samples, 1],
    "bird_id": bird_ids[1]
}))

# Concatenate all synthetic data
synthetic_data = pd.concat(synthetic_data_all, ignore_index=True)

# Remove extreme values to improve robustness
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

synthetic_data = remove_outliers(synthetic_data, "longitude")
synthetic_data = remove_outliers(synthetic_data, "latitude")

# Save as CSV
synthetic_generative_file_path = "Data/SyntheticData_Generative_MultiModel.csv"
synthetic_data.to_csv(synthetic_generative_file_path, index=False)

print(f"Synthetic data generated and saved at {synthetic_generative_file_path}")
