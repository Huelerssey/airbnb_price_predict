import folium
from folium.plugins import HeatMap, FastMarkerCluster
import geopandas as gpd
from src.data_utility import carregar_dados_abnb

# Carregar os dados da geometria da cidade do Rio de Janeiro (capital)
gdf_geometria_capital_rj = gpd.read_file('geo/geometria_capital_do_rio_de_janeiro.json', driver='GeoJSON')

# Função para criar um mapa de cluster
def criar_mapa_cluster(dataframe, zoom: int):
    media_latitude = dataframe['latitude'].mean()
    media_longitude = dataframe['longitude'].mean()
    fmap = folium.Map(location=[media_latitude, media_longitude], zoom_start=zoom, tiles='cartodbpositron')
    mapa_cluster = FastMarkerCluster(dataframe[['latitude', 'longitude']].values.tolist())
    fmap.add_child(mapa_cluster)
    return fmap

# Função para criar um mapa de heatmap
def criar_mapa_heatmap(dataframe, zoom: int):
    media_latitude = dataframe['latitude'].mean()
    media_longitude = dataframe['longitude'].mean()
    fmap = folium.Map(location=[media_latitude, media_longitude], zoom_start=zoom, tiles='cartodbpositron')
    heat_data = [[row['latitude'], row['longitude']] for idx, row in dataframe.iterrows()]
    HeatMap(heat_data).add_to(fmap)
    return fmap

# Carregar os dados do seu dataframe
dados = carregar_dados_abnb()

# Criar o mapa de cluster e heatmap
mapa_cluster = criar_mapa_cluster(dados, zoom=11)
mapa_heatmap = criar_mapa_heatmap(dados, zoom=11)

# Adicionar a geometria da capital do RJ aos mapas
folium.GeoJson(gdf_geometria_capital_rj).add_to(mapa_cluster)
folium.GeoJson(gdf_geometria_capital_rj).add_to(mapa_heatmap)

# Salvar os mapas como arquivos HTML
mapa_cluster.save('mapas/cluster_map.html')
mapa_heatmap.save('mapas/heat_map.html')