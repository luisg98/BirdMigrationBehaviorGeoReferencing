import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Carregar os dados
df = pd.read_csv("Data/Filtered_Migration_Data.csv")


# Converter timestamp para datetime e extrair ano e mês
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month


# Agrupar os dados para obter a posição mais provável mensal por ano
#df_grouped = df.groupby(['year', 'month']).agg({'latitude': lambda x: x.mode()[0], 'longitude': lambda x: x.mode()[0]}).reset_index()

# Definir as features (ano e mês) e os alvos (latitude e longitude)
X = df[['month']].values
y = df[['latitude', 'longitude']].values

# Normalizar os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# # Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# Definir a arquitetura da rede neural
model = Sequential([
    Dense(16, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2)  # Saída com duas unidades para latitude e longitude
])

# Adicionar early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compilar o modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Avaliar o modelo
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Mean Absolute Error: {mae}')

# Prever os próximos 12 meses
future_months = np.arange(1, 13).reshape(-1, 1)  # Criar array de 12 meses
future_months_scaled = scaler_X.transform(future_months)

y_pred = model.predict(future_months_scaled)
y_pred_rescaled = scaler_y.inverse_transform(y_pred)

# Criar dataframe das previsões
predicted_df = pd.DataFrame(y_pred_rescaled, columns=['Latitude Prevista', 'Longitude Prevista'])
predicted_df['Mês'] = future_months.flatten()
predicted_df = predicted_df.sort_values(by=['Mês'])
predicted_df.head()

# Criar mapa com os pontos previstos
mapa = folium.Map(location=[predicted_df['Latitude Prevista'].mean(), predicted_df['Longitude Prevista'].mean()], zoom_start=4)

for _, row in predicted_df.iterrows():
    folium.Marker(
        location=[row['Latitude Prevista'], row['Longitude Prevista']],
        popup=f"Mês: {int(row['Mês'])}",
        icon=folium.Icon(color='blue')
    ).add_to(mapa)


mapa