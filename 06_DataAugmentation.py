import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Função para preencher os valores faltantes de latitude e restaurar as outras features
def impute_latitude(data):
    # Salvar as colunas originais para reintegrá-las depois
    original_columns = data.columns

    # Carregar o modelo treinado
    model = load_model('model/lstm_model.h5')

    # Remover linhas onde 'timestamp' ou 'location-long' são nulos
    data = data.dropna(subset=['timestamp', 'location-long']).copy()

    # Criar um DataFrame apenas com as colunas necessárias para a previsão
    data_filtered = data[['timestamp', 'location-lat', 'location-long']].copy()

    # Converter o timestamp para formato numérico
    data_filtered['timestamp'] = pd.to_datetime(data_filtered['timestamp'])
    data_filtered['timestamp'] = data_filtered['timestamp'].astype('int64') // 10**9  # Convertendo para segundos desde a época

    # Separar os dados conhecidos e desconhecidos
    known_lat = data_filtered.dropna(subset=['location-lat']).copy()
    unknown_lat = data_filtered[data_filtered['location-lat'].isna()].copy()

    # Criar escaladores separados
    scaler_input = MinMaxScaler(feature_range=(0, 1))  # Para timestamp e location-long
    scaler_lat = MinMaxScaler(feature_range=(0, 1))  # Para location-lat

    # Ajustar os escaladores usando apenas os dados conhecidos
    scaler_input.fit(known_lat[['timestamp', 'location-long']])
    scaler_lat.fit(known_lat[['location-lat']])  # Normaliza apenas a latitude conhecida

    # Normalizar os dados de entrada (timestamp e longitude)
    unknown_lat_scaled = scaler_input.transform(unknown_lat[['timestamp', 'location-long']])

    # Prever e substituir apenas os valores ausentes na latitude
    for i, row in unknown_lat.iterrows():
        X_single = unknown_lat_scaled[unknown_lat.index.get_loc(i)].reshape(1, 1, -1)

        print(f"Previsão de latitude para a linha {i} com os dados normalizados: {X_single}")

        # Prever a latitude normalizada
        predicted_latitude = model.predict(X_single)
        predicted_value_normalized = predicted_latitude[0, 0]

        # Desnormalizar a previsão
        predicted_value = scaler_lat.inverse_transform([[predicted_value_normalized]])[0, 0]

        # Verifica se a previsão é válida
        if not pd.isna(predicted_value) and np.isfinite(predicted_value):
            # Substituir apenas valores nulos pela previsão desnormalizada
            data_filtered.at[i, 'location-lat'] = predicted_value
            print(f"Latitude prevista para a linha {i} (desnormalizada): {predicted_value}")

    # Restaurar as outras colunas do dataset original
    data.update(data_filtered[['location-lat']])

    # Salvar o resultado final
    data.to_csv('data_imputed_with_latitude.csv', index=False, encoding='utf-8')
    print("Valores de latitude preenchidos e dataset salvo com sucesso!")

# Carregar o dataset e rodar a imputação
data = pd.read_csv('Migration of red-backed shrikes from the Iberian Peninsula (data from Tttrup et al. 2017).csv')
impute_latitude(data)
