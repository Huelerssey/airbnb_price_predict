import geopandas as gpd

# Carregar os dados de geolocalização do estado do RJ
data_rj = gpd.read_file('geo/RJ_Municipios_2022.shp')

# Filtrar apenas a geometria da capital do RJ (Rio de Janeiro)
gdf_capital_rj = data_rj[data_rj['NM_MUN'] == 'Rio de Janeiro']

# Salvar um arquivo GeoJSON com os dados da geometria da capital do RJ
filename_capital_rj = 'geo/geometria_capital_do_rio_de_janeiro.json'
gdf_capital_rj.to_file(filename_capital_rj, driver='GeoJSON')
