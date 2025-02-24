import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import xgboost as xgb
import pickle
import torch

# 🚀 Carregar dataset
data = pd.read_csv("Data/Filtered_Migration_Data.csv")
X = data[['timestamp', 'longitude']]
y = data['latitude']

# 🚀 Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Carregar modelos previamente treinados
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("Model/XGB/xgboost_model.json")

with open("Model/MLP/mlp_model.pkl", "rb") as f:
    mlp_model = pickle.load(f)

lstm_model = load_model("Model/LSTM/lstm_model.h5")

# 🚀 Definir função para prever com LSTM
def predict_lstm(X):
    X_seq = np.reshape(X.values, (X.shape[0], 10, X.shape[1] // 10))
    return lstm_model.predict(X_seq).flatten()

# 🚀 Criar modelo de ensemble (Stacking)
estimators = [
    ('xgb', xgb_model),
    ('mlp', mlp_model),
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5
)

# 🚀 Treinar ensemble
stacking_model.fit(X_train, y_train)

# 🚀 Fazer previsões
y_pred_stacking = stacking_model.predict(X_test)

# 🚀 Avaliação do ensemble
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)

print(f"📊 Stacking Ensemble - MSE: {mse_stacking:.5f}, MAE: {mae_stacking:.5f}")

# 🚀 Salvar modelo treinado
with open("Model/Ensemble/stacking_model.pkl", "wb") as f:
    pickle.dump(stacking_model, f)
