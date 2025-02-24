import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Load dataset
file_path = "Data/Filtered_Migration_Data.csv"
df = pd.read_csv(file_path)

# Define a one-year period
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2024-12-31")
timestamps = pd.date_range(start=start_date, end=end_date, freq='24h')

# Create multiple synthetic bird IDs
bird_ids = [99999, 88888, 77777]
synthetic_data_all = []

### MODEL 1: Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=8, covariance_type='full', max_iter=500, random_state=42)
gmm.fit(df[['longitude', 'latitude']])
synthetic_coords_gmm, _ = gmm.sample(len(timestamps))
synthetic_data_all.append(pd.DataFrame({
    "timestamp": timestamps,
    "longitude": synthetic_coords_gmm[:, 0],
    "latitude": synthetic_coords_gmm[:, 1],
    "bird_id": bird_ids[0]
}))


### MODEL 2: Variational Autoencoder (VAE)
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

# Train VAE
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
generated_data_vae = vae.decode(z).detach().numpy()

# Ajustar número de amostras
num_samples = min(len(timestamps), len(generated_data_vae))
synthetic_data_all.append(pd.DataFrame({
    "timestamp": timestamps[:num_samples],
    "longitude": generated_data_vae[:num_samples, 0],
    "latitude": generated_data_vae[:num_samples, 1],
    "bird_id": bird_ids[1]
}))


### MODEL 3: Generative Adversarial Network (GAN)
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(input_dim=2, output_dim=2)
discriminator = Discriminator(input_dim=2)

# Optimizers and loss function
gen_optimizer = optim.Adam(generator.parameters(), lr=0.001)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Training GAN
num_epochs = 100
for epoch in range(num_epochs):
    # Sample real data
    real_data = torch.tensor(df[['longitude', 'latitude']].values, dtype=torch.float32)
    
    # Generate fake data
    noise = torch.randn(real_data.size(0), 2)
    fake_data = generator(noise)
    
    # Train Discriminator
    disc_optimizer.zero_grad()
    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data.detach())
    loss_disc = loss_fn(real_pred, torch.ones_like(real_pred)) + loss_fn(fake_pred, torch.zeros_like(fake_pred))
    loss_disc.backward()
    disc_optimizer.step()
    
    # Train Generator
    gen_optimizer.zero_grad()
    fake_pred = discriminator(fake_data)
    loss_gen = loss_fn(fake_pred, torch.ones_like(fake_pred))
    loss_gen.backward()
    gen_optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss Disc = {loss_disc.item()}, Loss Gen = {loss_gen.item()}")

# Generate synthetic data using GAN
noise = torch.randn(len(timestamps), 2)
generated_data_gan = generator(noise).detach().numpy()

synthetic_data_all.append(pd.DataFrame({
    "timestamp": timestamps,
    "longitude": generated_data_gan[:, 0],
    "latitude": generated_data_gan[:, 1],
    "bird_id": bird_ids[2]
}))

# Combine all synthetic data
synthetic_data = pd.concat(synthetic_data_all, ignore_index=True)

# Remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

synthetic_data = remove_outliers(synthetic_data, "longitude")
synthetic_data = remove_outliers(synthetic_data, "latitude")

# Save to CSV
synthetic_generative_file_path = "Data/SyntheticData_Ensemble.csv"
synthetic_data.to_csv(synthetic_generative_file_path, index=False)

print(f"Synthetic data generated and saved at {synthetic_generative_file_path}")
