# Red-backed Shrike Migration Forecasting

ML pipeline for predicting and synthesizing migratory trajectories of the red-backed shrike (*Lanius collurio*). GPS telemetry + environmental data from Movebank (four Pedersen/Tøttrup studies, 2017–2020).

Models: XGBoost (best performing), LSTM, MLP, GNN. Latitude imputation via XGBoost. Synthetic trajectory generation via GMM and VAE. Monthly heatmaps for seasonal pattern analysis.

## Stack

Python, PyTorch, XGBoost, scikit-learn, matplotlib.
