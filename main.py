# Importação das bibliotecas principais
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import numpy as np
from datetime import datetime
from geopy.distance import geodesic
from mpl_toolkits.basemap import Basemap

# 1. Leitura dos Dados
df = pd.read_csv('Migration of red-backed shrikes from the Iberian Peninsula (data from Tttrup et al. 2017).csv')

# Visualizar as primeiras linhas do DataFrame para checar se os dados estão carregados corretamente
print(df.head())

# 2. Pré-processamento dos Dados
# Conversão da coluna de timestamp para datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

# Checar por valores nulos
print(df.isnull().sum())

# Filtrar colunas relevantes
df = df[['timestamp', 'location-long', 'location-lat', 'individual-local-identifier']]

# Filtrando dados para um único indivíduo (exemplo: o primeiro identificado)
individual_id = df['individual-local-identifier'].unique()[0]
df_individual = df[df['individual-local-identifier'] == individual_id]

# Visualizando os dados filtrados
print(df_individual.head())

# 3. Criação de Geodataframe
# Criar uma geometria de pontos a partir das colunas de latitude e longitude
geometry = [Point(lon, lat) for lon, lat in zip(df_individual['location-long'], df_individual['location-lat'])]

# Criar um GeoDataFrame
gdf = gpd.GeoDataFrame(df_individual, geometry=geometry)

# Checar o GeoDataFrame
print(gdf.head())

# 4. Visualização do Caminho de Migração
plt.figure(figsize=(10, 6))
plt.plot(gdf['location-long'], gdf['location-lat'], marker='o', color='blue', label='Caminho de migração')

# Adicionar título e rótulos
plt.title(f'Caminho de Migração da Ave {individual_id}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Exibir o gráfico
plt.legend()
plt.grid(True)
plt.show()

# 5. Mapeamento Geográfico com Basemap (Opcional)
plt.figure(figsize=(10, 7))
m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)

# Desenhar o mapa
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='lightgreen', lake_color='aqua')
m.drawparallels(np.arange(-90., 91., 30.))
m.drawmeridians(np.arange(-180., 181., 60.))

# Converter as coordenadas para o formato do mapa
x, y = m(gdf['location-long'].values, gdf['location-lat'].values)

# Plotar as coordenadas da migração
m.plot(x, y, marker='o', color='blue', markersize=5, linewidth=1)

# Adicionar título e exibir o mapa
plt.title(f'Migração da Ave {individual_id}')
plt.show()

# 6. Análise de Padrões de Migração
# Calcular a diferença de tempo entre as observações
gdf['time-delta'] = gdf['timestamp'].diff().fillna(pd.Timedelta(0))

# Calcular distâncias entre os pontos sucessivos (usando a fórmula de Haversine)
def haversine(lon1, lat1, lon2, lat2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

distances = [0]  # A primeira distância é 0
for i in range(1, len(gdf)):
    dist = haversine(gdf.iloc[i-1]['location-long'], gdf.iloc[i-1]['location-lat'],
                     gdf.iloc[i]['location-long'], gdf.iloc[i]['location-lat'])
    distances.append(dist)

gdf['distance_km'] = distances

# Exibir resumo da distância e tempo
print(gdf[['timestamp', 'distance_km', 'time-delta']].head())

# 7. Análise Estatística
# Calcular estatísticas descritivas sobre distâncias e tempos
print(gdf['distance_km'].describe())

# Verificar a distribuição do tempo de migração
sns.histplot(gdf['time-delta'].dt.total_seconds() / 3600, bins=20, kde=True)
plt.title('Distribuição do Tempo de Migração (em horas)')
plt.xlabel('Tempo (horas)')
plt.ylabel('Frequência')
plt.show()

# 8. Salvar os Resultados
gdf.to_csv('resultado_migracao_shrike.csv', index=False)

print("Processo concluído e resultados salvos em 'resultado_migracao_shrike.csv'.")
