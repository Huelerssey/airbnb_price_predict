import time
import joblib
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


## 1 - INTRODUÇÃO - ##

"""
Projeto de ciência de dados com o objetivo de construir um modelo de previsão com
machine learning que seja capaz de prever o preço de uma diária do airbnb no RJ.
"""

## 1 - INTRODUÇÃO - ##

## 2 - OBTENÇÃO DOS DADOS ##

"""
https://www.kaggle.com/datasets/allanbruno/airbnb-rio-de-janeiro
"""

# importando base de dados
base_airbnb = pd.read_csv("dataset/dataset_final.csv", low_memory=False)

## 2 - OBTENÇÃO DOS DADOS ##

## 3 - ENTENDIMENTO DA ÁREA/NEGÓCIO - ##

"""
print(base_airbnb.head(5))
id: Identificador único da listagem.
listing_url: URL da listagem no site do Airbnb.
scrape_id: Identificador único da raspagem (scrape) que coletou esses dados.
last_scraped: Data da última raspagem.
name: Nome da listagem.
summary: Resumo ou descrição curta da listagem.
space: Descrição do espaço da listagem.
description: Descrição completa da listagem.
experiences_offered: Tipos de experiências oferecidas com a listagem.
neighborhood_overview: Visão geral do bairro.
price: Preço da diária.
availability: Disponibilidade da listagem.
number_of_reviews: Número de avaliações da listagem.
review_scores_rating: Classificação de avaliação da listagem.
property_type: Tipo de propriedade (por exemplo, apartamento, casa, etc.).
"""

## 3 - ENTENDIMENTO DA ÁREA/NEGÓCIO - ##

## 4 - LIMPEZA E TRATAMENTO DE DADOS - ##

"""
print(base_airbnb.info())
RangeIndex: 902210 entries, 0 to 902209
Columns: 109 entries, Unnamed: 0 to calculated_host_listings_count_shared_rooms
dtypes: float64(31), int64(16), object(62)
"""

#Colunas restantes após a análise qualitativa
colunas = [
    'host_is_superhost', 'host_listings_count',	'latitude',
    'longitude', 'property_type', 'room_type', 'accommodates',
    'bathrooms',	'bedrooms',	'beds',	'bed_type',	'amenities',
    'price', 'security_deposit',	'cleaning_fee',	'guests_included',
    'extra_people',	'minimum_nights', 'maximum_nights',	'number_of_reviews',
    'review_scores_rating',	'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',	
    'review_scores_location', 'review_scores_value', 'instant_bookable',
    'is_business_travel_ready', 'cancellation_policy', 'ano', 'mes',
]

#deleta colunas com valores nulos acima de 30 mil
base_airbnb = base_airbnb.loc[:,colunas]
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)

#exclui todos as linhas de uma coluna que possui valor none
base_airbnb = base_airbnb.dropna() 

# transforma dados da coluna prince em float
base_airbnb['price'] = base_airbnb['price'].apply(lambda x: float(x.replace('$', '').replace(',', '')))

# transforma dados da coluna extra_people em float
base_airbnb['extra_people'] = base_airbnb['extra_people'].apply(lambda x: float(x.replace('$', '').replace(',', '')))

# converter colunas booleanas em int (0 ou 1)
def converter_colunas_bool(df, colunas):
    for coluna in colunas:
        df[coluna] = df[coluna].map({'t': 1, 'f': 0})
    return df

#armazena as colunas que serão convertidas
colunas_para_converter = ['host_is_superhost', 'instant_bookable']

# aplica a fução de conversão
base_airbnb = converter_colunas_bool(base_airbnb, colunas_para_converter)

# lista de colunas que não fazem mais sentido após a análise exploratória
colunas_excluir = [
    'instant_bookable', 'guests_included', 'maximum_nights',
    'number_of_reviews', 'is_business_travel_ready', 'host_listings_count',
    'minimum_nights', 'bed_type', 'cancellation_policy'
]

# excluir colunas selecionadas
base_airbnb = base_airbnb.drop(columns=colunas_excluir)

## 4 - LIMPEZA E TRATAMENTO DE DADOS - ##

## 5 - ANÁLISE EXPLORATÓRIA DE DADOS - ##

## FUNÇÕES AUXILIARES ##

# retorna o limite inferior e o limite superior
def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return (q1 - 1.5 * amplitude, q3 + 1.5 * amplitude)

# Exclui outliers e retorna o novo dataframe e também a quantidade de linhas removidas
def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), : ]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas

# plota um diagrama de caixa com as informações da coluna já entre o limite
def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15,5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    return plt.show()

# plota um histograma da coluna passada como parâmetro
def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)
    return plt.show()

# plota um gráfico de barras com os dados da coluna já entre os dois limites
def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))
    return plt.show()

# plota um gráfico de barras para auxiliar a avaliação em colunas de texto
def grafico_aux_txt(data, coluna: str, figsize=(15, 5)):
    print(base_airbnb[coluna].value_counts())
    plt.figure(figsize=figsize)
    grafico = sns.countplot(data=data, x=coluna)
    grafico.tick_params(axis='x', rotation=90)
    return plt.show()

## FUNÇÕES AUXILIARES ##

## COLUNAS NUMÉRICAS ##

#price
# diagrama_caixa(base_airbnb['price'])
#histograma(base_airbnb['price'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print(f'{linhas_removidas} linhas removidas da coluna price')

#extra_people
# diagrama_caixa(base_airbnb['extra_people'])
# histograma(base_airbnb['extra_people'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print(f'{linhas_removidas} linhas removidas da coluna extra_people')

#host_listings_count
# diagrama_caixa(base_airbnb['host_listings_count'])
# grafico_barra(base_airbnb['host_listings_count'])
# movido para etapa de limpeza (foi excluido)

#accommodates
# diagrama_caixa(base_airbnb['accommodates'])
# grafico_barra(base_airbnb['accommodates'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print(f'{linhas_removidas} linhas removidas da coluna accommodates')

#bathrooms
# diagrama_caixa(base_airbnb['bathrooms'])
# grafico_barra(base_airbnb['bathrooms'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print(f'{linhas_removidas} linhas removidas da coluna bathrooms')

#bedrooms
# diagrama_caixa(base_airbnb['bedrooms'])
# grafico_barra(base_airbnb['bedrooms'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print(f'{linhas_removidas} linhas removidas da coluna bedrooms')

#beds
# diagrama_caixa(base_airbnb['beds'])
# grafico_barra(base_airbnb['beds'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print(f'{linhas_removidas} linhas removidas da coluna beds')

#guests_included
# diagrama_caixa(base_airbnb['guests_included'])
# grafico_barra(base_airbnb['guests_included'])

# print(limites(base_airbnb['guests_included']))
# plt.figure(figsize=(15, 5))
# sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())
# plt.show()
# movido para etapa de limpeza (foi excluido)


#minimum_nights
# diagrama_caixa(base_airbnb['minimum_nights'])
# grafico_barra(base_airbnb['minimum_nights'])
# movido para etapa de limpeza (foi excluido)

#maximum_nights
# diagrama_caixa(base_airbnb['maximum_nights'])
# grafico_barra(base_airbnb['maximum_nights'])
# movido para etapa de limpeza (foi excluido)

#number_of_reviews
# diagrama_caixa(base_airbnb['number_of_reviews'])
# grafico_barra(base_airbnb['number_of_reviews'])
# movido para etapa de limpeza (foi excluido)

## COLUNAS DE TEXTO ##

# print(base_airbnb.dtypes)
# print('-'*60)
# print(base_airbnb.iloc[0])

#property_type
#print(base_airbnb['property_type'].value_counts()) conta quantos valores existem para cada tipo de texto
# grafico_aux_txt(base_airbnb, 'property_type')


tabela_tipos_casa = base_airbnb['property_type'].value_counts() #descreve a categoria e seu valor
#print(tabela_tipos_casa.index) descreve apenas as caregorias
#print(tabela_tipos_casa['Apartament']) descreve os valores da categoria passada

# Agrupa todos as categorias com valor menor que 2000 em uma lista
colunas_agrupar = []
for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)

# inserir todos os valores da lista colunas_agrupar na categoria outros da coluna property_type
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Other'

#room_type
# grafico_aux_txt(base_airbnb, 'room_type')

#bed_type
# grafico_aux_txt(base_airbnb, 'bed_type')
# movido para etapa de limpeza (foi excluido)

#cancellation_policy
# grafico_aux_txt(base_airbnb, 'cancellation_policy')
# movido para etapa de limpeza (foi excluido)

#amenities
# criar uma nova coluna composta apenas pela quantidade de amenities
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)

# deletar a coluna antiga que foi substituida
base_airbnb = base_airbnb.drop('amenities', axis=1)

# diagrama_caixa(base_airbnb['n_amenities'])
# grafico_barra(base_airbnb['n_amenities'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print(f'{linhas_removidas} linhas removidas da coluna n_amenities')

# salva o dataset modelado
base_airbnb.to_pickle("arquivos_pkl/dataset_airbnb_modelado.pkl")

## PARA COLUNAS CATEGÓRICAS ##

# transforma colunas categóricas em numéricas
colunas_categoria = ['property_type', 'room_type']
base_airbnb = pd.get_dummies(data=base_airbnb, columns=colunas_categoria)

# Amostra aleatória de 300.000 linhas
# base_airbnb_sample = base_airbnb.sample(n=300000, random_state=42)

# correlação das colunas com o preço
# print(base_airbnb.corr()['price'])
# host_is_superhost                                 -0.136795
# host_listings_count                               -0.138097
# latitude                                          -0.174621
# longitude                                         -0.121438
# accommodates                                       0.398132
# bathrooms                                          0.354889
# bedrooms                                           0.408713
# beds                                               0.289161
# price                                              1.000000
# extra_people                                      -0.186581
# minimum_nights                                     0.186452
# ano                                               -0.000694
# mes                                               -0.000073
# n_amenities                                       -0.085493
# property_type_Apartment                            0.117570
# property_type_Bed and breakfast                   -0.024236
# property_type_Condominium                         -0.021423
# property_type_Guest suite                         -0.040343
# property_type_Guesthouse                          -0.022910
# property_type_Hostel                              -0.033197
# property_type_House                               -0.080641
# property_type_Loft                                -0.043684
# property_type_Other                               -0.034693
# property_type_Serviced apartment                  -0.008575
# room_type_Entire home/apt                          0.324517
# room_type_Hotel room                              -0.009420
# room_type_Private room                            -0.302371
# room_type_Shared room                             -0.086653
# cancellation_policy_flexible                       0.091993
# cancellation_policy_moderate                      -0.089644
# cancellation_policy_strict                        -0.005938
# cancellation_policy_strict_14_with_grace_period   -0.021134
# bed_type_Other                                    -0.022128
# bed_type_Real Bed                                  0.022128

# base de dados pronta para a IA
# print(base_airbnb.info())
# Index: 557640 entries, 1 to 902209
# Data columns (total 34 columns):
#  #   Column                                           Non-Null Count   Dtype
# ---  ------                                           --------------   -----
#  0   host_is_superhost                                557640 non-null  int64
#  1   host_listings_count                              557640 non-null  float64
#  2   latitude                                         557640 non-null  float64
#  3   longitude                                        557640 non-null  float64
#  4   accommodates                                     557640 non-null  int64
#  5   bathrooms                                        557640 non-null  float64
#  6   bedrooms                                         557640 non-null  float64
#  7   beds                                             557640 non-null  float64
#  8   price                                            557640 non-null  float64
#  9   extra_people                                     557640 non-null  float64
#  10  minimum_nights                                   557640 non-null  int64
#  11  ano                                              557640 non-null  int64
#  12  mes                                              557640 non-null  int64
#  13  n_amenities                                      557640 non-null  int64
#  14  property_type_Apartment                          557640 non-null  bool
#  15  property_type_Bed and breakfast                  557640 non-null  bool
#  16  property_type_Condominium                        557640 non-null  bool
#  17  property_type_Guest suite                        557640 non-null  bool
#  18  property_type_Guesthouse                         557640 non-null  bool
#  19  property_type_Hostel                             557640 non-null  bool
#  20  property_type_House                              557640 non-null  bool
#  21  property_type_Loft                               557640 non-null  bool
#  22  property_type_Other                              557640 non-null  bool
#  23  property_type_Serviced apartment                 557640 non-null  bool
#  24  room_type_Entire home/apt                        557640 non-null  bool
#  25  room_type_Hotel room                             557640 non-null  bool
#  26  room_type_Private room                           557640 non-null  bool
#  27  room_type_Shared room                            557640 non-null  bool
#  28  cancellation_policy_flexible                     557640 non-null  bool
#  29  cancellation_policy_moderate                     557640 non-null  bool
#  30  cancellation_policy_strict                       557640 non-null  bool
#  31  cancellation_policy_strict_14_with_grace_period  557640 non-null  bool
#  32  bed_type_Other                                   557640 non-null  bool
#  33  bed_type_Real Bed                                557640 non-null  bool
# dtypes: bool(20), float64(8), int64(6)

# Vizualização de mapa das propriedades
# amostra = base_airbnb.sample(n=50000) #amostra de 50 mil unidades das colunas
# centro_mapa = {'lat': amostra.latitude.mean(), 'lon': amostra.longitude.mean()} #setar o centro do mapa
# mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5,
#                         center=centro_mapa, zoom=10,
#                         mapbox_style='stamen-terrain')
#mapa.show()

## 5 - ANÁLISE EXPLORATÓRIA DE DADOS - ##

## 6 - MODELANDO UMA INTELIGÊNCIA ARTIFICIAL - ##

# # Separando as características e o alvo
# X_completo_csv = base_airbnb_sample.drop(columns=['price'])
# y_completo_csv = base_airbnb_sample['price']

# # Dividindo os dados em conjuntos de treinamento e teste
# X_treino_completo_csv, X_teste_completo_csv, y_treino_completo_csv, y_teste_completo_csv = train_test_split(X_completo_csv, y_completo_csv, test_size=0.2, random_state=42)

# # Função para treinar e avaliar um modelo
# def treinar_e_avaliar_modelo(modelo, X_treino, y_treino, X_teste, y_teste):
#     modelo.fit(X_treino, y_treino)
#     y_pred = modelo.predict(X_teste)
#     mse = mean_squared_error(y_teste, y_pred)
#     r2 = r2_score(y_teste, y_pred)
#     print(f'Erro Quadrático Médio: {mse}')
#     print(f'Coeficiente de Determinação (R2): {r2}')
#     return mse, r2

# # Modelos
# modelos = {
#     'Árvores Extras': ExtraTreesRegressor(n_estimators=100, random_state=42),
#     'Floresta Aleatória': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Lasso': Lasso(),
#     'Regressão Linear': LinearRegression()
# }

# # Treinando e avaliando cada modelo
# for nome_modelo, instancia_modelo in modelos.items():
#     print(f'\nAvaliando {nome_modelo}:')
#     treinar_e_avaliar_modelo(instancia_modelo, X_treino_completo_csv, y_treino_completo_csv, X_teste_completo_csv, y_teste_completo_csv)

## 6 - MODELANDO UMA INTELIGÊNCIA ARTIFICIAL - ##

## 7 - RESULTADOS - ##

# Avaliando Árvores Extras:
# Erro Quadrático Médio: 1605.5039857450938
# Coeficiente de Determinação (R2): 0.9771319515351796

# Avaliando Floresta Aleatória:
# Erro Quadrático Médio: 1718.4106154042179
# Coeficiente de Determinação (R2): 0.9755237622675297

# Avaliando Lasso:
# Erro Quadrático Médio: 48425.88675110813
# Coeficiente de Determinação (R2): 0.31024430022681415

# Avaliando Regressão Linear:
# Erro Quadrático Médio: 47549.581169059275
# Coeficiente de Determinação (R2): 0.3227259874092885

## 7 - RESULTADOS - ##

## 8 - ESCOLHENDO O MELHOR MODELO E COLOCANDO EM PRODUÇÃO - ##

# definindo dados de treino e teste
y = base_airbnb['price']
x = base_airbnb.drop('price', axis=1)

# dividindo a base entre treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2 ,random_state=42)

# cria o modelo
modelo_extratrees = ExtraTreesRegressor(n_estimators=10, random_state=42)

# treina o modelo
modelo_extratrees.fit(x_treino, y_treino)

# testa o modelo
y_pred = modelo_extratrees.predict(x_teste)

#avalia o modelo
mse = mean_squared_error(y_teste, y_pred)
r2 = r2_score(y_teste, y_pred)
print(f'Erro Quadrático Médio: {mse}')
print(f'Coeficiente de Determinação (R2): {r2}')

#armazena o modelo treinado para produção
joblib.dump(modelo_extratrees, "arquivos_pkl/modelo_airbnb_treinado.pkl")

## 8 - ESCOLHENDO O MELHOR MODELO E COLOCANDO EM PRODUÇÃO - ##