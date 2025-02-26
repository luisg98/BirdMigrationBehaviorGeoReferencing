# Bird Migration Behavior Georeferencing

## 📌 Overview
This repository contains the code and data used for analyzing and predicting the migratory trajectories of the **red-backed shrike** (*Lanius collurio*). The study employs various **machine learning** models to predict bird migration patterns and includes data preprocessing, feature engineering, model evaluation, and synthetic trajectory generation. 

## 📊 Key Features
- **Data Integration & Cleaning**: Merging and standardizing GPS telemetry and environmental datasets.
- **Machine Learning Models**:
  - XGBoost (Best performing model)
  - LSTM (Long Short-Term Memory networks)
  - Multi-Layer Perceptron (MLP)
  - Graph Neural Networks (GNN)
- **Latitude Imputation**: Handling missing latitude values using an XGBoost-based imputation method.
- **Synthetic Data Generation**:
  - **Gaussian Mixture Model (GMM)**: Used for realistic migration trajectory simulations.
  - **Variational Autoencoder (VAE)**: Learning latent spatial patterns for synthetic migration modeling.
- **Heatmap Visualization**: Generating monthly heatmaps to analyze seasonal migration trends.

## 📁 Dataset
The dataset consists of GPS tracking records of **red-backed shrikes**
The four CSV files merged into one dataset are:

1. **Migration of male and female red-backed shrikes from southern Scandinavia (Pedersen et al. 2019)**  
2. **Migration of red-backed shrike populations (Pedersen et al. 2020)**  
3. **Migration of red-backed shrikes from southern Scandinavia (Pedersen et al. 2018)**  
4. **Migration of red-backed shrikes from the Iberian Peninsula (Tttrup et al. 2017)**  

- Datasets downloaded from: [Movebank](https://www.movebank.org/cms/webapp?gwt_fragment=page%3Dstudies%2Cpath%3Dstudy225376313)
