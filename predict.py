import time
import joblib
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


#Cria o dataframe onde as informações serão inseridas
base_airbnb = pd.DataFrame()
caminho_bases = pathlib.Path('dataset')

#cria uma coluna preenchida com as informações de mês e ano
meses = {
    'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4,
    'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8,
    'set': 9, 'out': 10, 'nov': 11, 'dez': 12
}

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]

    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))

    df = pd.read_csv(caminho_bases / arquivo.name, low_memory=False)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = pd.concat([base_airbnb, df])

"""
print(list(base_airbnb.columns)) mostra as colunas 
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';') gera um excel com as primeiras mil linhas e todas as colunas separadas por ;
print(base_airbnb[['experiences_offered']].value_counts()) conta valores de uma coluna
print((base_airbnb['host_listings_count']==base_airbnb['host_total_listings_count']).value_counts()) compara duas colunas
print(base_airbnb['square_feet'].isnull().sum()) verificar quuantidades de linhas com valores nulos
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

base_airbnb = base_airbnb.dropna() #exclui todos as linhas de uma coluna que possui valor none
#print(base_airbnb.shape) mostra a quantidade de linhas e colunas exstentes no dataframe

"""
até este momento a coluna price e extra_people eram pra ser do tipo int
ou float mas estão como objects
"""

"""
print(base_airbnb.dtypes) #tipos de colunas
print('-'*60)
print(base_airbnb.iloc[0])
para comprar as colunas com os tipos de valores que elas possuem, fazendo uma analise qualitativa
"""

"""
Se observar no arquivo csv o que tornava os numeros um objeto era o sifrão e a virgula separando
a casa decimal, e com estes comandos concertamos esse problema
"""

base_airbnb['price'] = base_airbnb['price'].str.replace('$', '', regex=True)
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '', regex=True)
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)

base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '', regex=True)
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '', regex=True)
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)

#print(base_airbnb.corr()) mostra a correlação entre os dados

"""
com os seguintes comandos é possível plotar um gráfico de corelação
para poder analisar se alguma métrica está muito fora da curva para
poder ser excluida.
plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(numeric_only=True), annot=True, cmap='Greens')
plt.show()
"""

#Análise exploratória e tratamento de Outliers
"""
Usando a lógia estatistica, deveremos decidir se pode ser excluido: 
valores abaixo de Q1 - 1.5 x amplitude
valores acima de Q3 + 1.5 x amplitude
onde amplitude = Q3 - Q1
Q1 e Q3 são o primeiro e o terceiro quartil da amostra, respectivamente.
"""

def limites(coluna):
    """retorna o limite inferior e o limite superior"""
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return (q1 - 1.5 * amplitude, q3 + 1.5 * amplitude)
    #print(base_airbnb['price'].describe()) ajuda a conferir se o calculo está correto

def excluir_outliers(df, nome_coluna):
    """Exclui outliers e retorna o novo dataframe e também a quantidade de linhas removidas"""
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), : ]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas

def diagrama_caixa(coluna):
    """plota um diagrama de caixa com as informações da coluna já entre o limite"""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15,5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    return plt.show()

def histograma(coluna):
    """plota um histograma da coluna passada como parâmetro"""
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)
    return plt.show()

def grafico_barra(coluna):
    """plota um gráfico de barras com os dados da coluna já entre os dois limites"""
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))
    return plt.show()

def grafico_aux_txt(data, coluna: str, figsize=(15, 5)):
    """plota um gráfico de barras para auxiliar a avaliação em colunas de texto"""
    print(base_airbnb[coluna].value_counts())
    plt.figure(figsize=figsize)
    grafico = sns.countplot(data=data, x=coluna)
    grafico.tick_params(axis='x', rotation=90)
    return plt.show()

"""Com as funções auxiliares construidas, é hora de usar para analises e exclusão de outliers"""

#para colunas com valores numéricos:

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
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print(f'{linhas_removidas} linhas removidas da coluna host_listings_count')

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

# neste caso precimos de uma análise adicional, o gráfico aponta o lim sup e inf com o msm valor
# print(limites(base_airbnb['guests_included']))
# plt.figure(figsize=(15, 5))
# sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())
# plt.show()
# devido aos dados não preenchidos corretamente da base de dados, o mais correto é excluir essa coluna
base_airbnb = base_airbnb.drop('guests_included', axis=1)

#minimum_nights
# diagrama_caixa(base_airbnb['minimum_nights'])
# grafico_barra(base_airbnb['minimum_nights'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print(f'{linhas_removidas} linhas removidas da coluna minimum_nights')

#maximum_nights
# diagrama_caixa(base_airbnb['maximum_nights'])
# grafico_barra(base_airbnb['maximum_nights'])
# devido a falta de sentido desses dados, eles também seram excluidos
base_airbnb = base_airbnb.drop('maximum_nights', axis=1)

#number_of_reviews
# diagrama_caixa(base_airbnb['number_of_reviews'])
# grafico_barra(base_airbnb['number_of_reviews'])
# como a base de dados favorece muito quem já está a mais tempo na plataforma devido ao numero de rewis ela será excluida
base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)

"""para colunas com categorias de texto:"""
# print(base_airbnb.dtypes) #tipos de colunas
# print('-'*60)
# print(base_airbnb.iloc[0])

#property_type
#print(base_airbnb['property_type'].value_counts()) conta quantos valores existem para cada tipo de texto
# grafico_aux_txt(base_airbnb, 'property_type')
"""
compradando o gráfico com a tabela, é visível que existem muitas categorias com poucos valores
tornando as tais, irrelevante para o nosso modelo, sendo assim, todas as categorias com valores
a baixo de "other" serão transformados em other
"""
tabela_tipos_casa = base_airbnb['property_type'].value_counts() #descreve a categoria e seu valor
#print(tabela_tipos_casa.index) descreve apenas as caregorias
#print(tabela_tipos_casa['Apartament']) descreve os valores da categoria passada

"""Agrupa todos as categorias com valor menos que 2000 em uma lista"""
colunas_agrupar = []
for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)
#print(colunas_agrupar)

"""inserir todos os valores da lista colunas_agrupar na categoria outros do dataframe"""
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Other'

"""comprando os novos valores"""
# print(tabela_tipos_casa)
# print('-' * 60)
# print(base_airbnb['property_type'].value_counts())

"""plotando o novo gráfico com os valores reagrupados na coluna outros"""
# grafico_aux_txt(base_airbnb, 'property_type')

#room_type
# grafico_aux_txt(base_airbnb, 'room_type')

#bed_type
# grafico_aux_txt(base_airbnb, 'bed_type')
"""de acordo com a análise, será necessário agrupar as categorias de menor porte em outros"""
# tabela_tipos_cama = base_airbnb['bed_type'].value_counts()
# colunas_agrupar_camas = []
# for tipo in tabela_tipos_cama.index:
#     if tabela_tipos_cama[tipo] < 9000:
#         colunas_agrupar_camas.append(tipo)

# for tipo in colunas_agrupar_camas:
#     base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Other'
# excluindo bed_type para teste
base_airbnb = base_airbnb.drop('bed_type', axis=1)

#cancellation_policy
# grafico_aux_txt(base_airbnb, 'cancellation_policy')
"""de acordo com a análise, será necessário agrupar as categorias de menor porte em outros"""
tabela_tipos_politica = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar_politica = []
for tipo in tabela_tipos_politica.index:
    if tabela_tipos_politica[tipo] < 10000:
        colunas_agrupar_politica.append(tipo)

for tipo in colunas_agrupar_politica:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'

# grafico_aux_txt(base_airbnb, 'cancellation_policy')

#amenities
"""
Como existe uma diversidade muito grande de amenities, em vez de considerar uma caracteristica por uma
o critério relevante será a quantidade de amenities por imóvel.
print(base_airbnb['amenities'].iloc[0].split(',')) todos os amenites de uma linha em lista.
print(len(base_airbnb['amenities'].iloc[0].split(','))) numeros de amenities de uma linha.
de acordo com a análise, criamos uma lista com todos os amenities, criamos uma nova coluna composta apenas
pela quantidade de amenities e por fim deletamos a coluna antiga.
"""

base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
base_airbnb = base_airbnb.drop('amenities', axis=1)

"""Agora que a nova coluna de amenities virou uma coluna numerica, vamos fazer a análise devidamente"""
# diagrama_caixa(base_airbnb['n_amenities'])
# grafico_barra(base_airbnb['n_amenities'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print(f'{linhas_removidas} linhas removidas da coluna n_amenities')

"""Vizualização de mapa das propriedades"""
# amostra = base_airbnb.sample(n=50000) #amostra de 50 mil unidades das colunas
# centro_mapa = {'lat': amostra.latitude.mean(), 'lon': amostra.longitude.mean()} #setar o centro do mapa
# mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5,
#                         center=centro_mapa, zoom=10,
#                         mapbox_style='stamen-terrain')
#mapa.show()

"""
Etapa de Encoding:
Features de valores True ou False seram subistituidas por 1 == True e 0 == False.
Features de categoria (features em que os valores da coluna são textos), 
vamos usar o método de variáveis dummies.
"""

#print(base_airbnb.columns)
#tratando colunas de verdadeiro ou falso
"""
como a coluna instant_bookable tornou o modelo mais complexo do que deveria
ela será excluida para teste.
"""
base_airbnb = base_airbnb.drop('instant_bookable', axis=1)

colunas_tf = ['host_is_superhost', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()

for coluna in base_airbnb_cod:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f', coluna] = 0
#print(base_airbnb_cod.iloc[0]) mostra a primeira linha de todas as colunas

#tratando colunas de categorias
colunas_categoria = ['property_type', 'room_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categoria)

#modelo de previsão
"""
1 -> Decidir se o problema em questão é de classificação ou regrção
- regressão é um problema de valor
- classificação é um problema de V/F

2 -> Escolher as métricas para avaliar o modelo (r² e rsme)
- funções em python já prontas
- r² é a capacidade em porcentagem de o quanto o modelo consegue descrever o preço final
- rsme é a percentagem de erro (se o modelo erra muito para os valores)

3 -> Escolher dentre todos os modelos de machine learning qual deles será usado
- regrssão linear:
traça uma linha reta de intersessão entre a menor distancia entre os pontos 
- random forest regressor:
cria várias arvores de decisões aleatória escolhendo a melhor pergunta e traçando a média entre as arvores
- extra trees:
cria várias arvores de decisões aleatória, escolhendo sempre uma pergunta aletória

4 -> Separar base de dados para treino e teste
- 80% treino
- 20% teste
- onde X são as caracteristicas do imóvel e Y será o preço

5 -> Comparar os resultados e escolher o modelo vencedor
- calcula-se o r² e o rsme para cada modelo
- escolhemos 1 métrica para ser a principal, digamos que seja o r²
- nesse caso, usaremos o modelo com maior r² e o rsme fica como fator de desempate
- modelo que demore menos
- modelo que preicse de menos informações para funcionar

6 - > Analisar o melhor modelo mais a fundo
- identificar a importancia de cada feature para ver oportunidades de melhoria . Se a coluna
não for utilizada pelo modelo ou pouco importante, podemos testar retirar ela para ver
o resultado, que pode melhorar ou piorar, isso será avaliado de acordo:
- métricas escolhidas (r² e rsme)
- velocidade do modelo
- simplicidade do modelo

7 -> fazer ajustes no melhor modelo
- a cada etapa, treinamos e testamos o modelo. sempre comparando com o resultado original
e o resultado anterior
- verificar se é possível chegar no mesmo resultado ou em um muito próximo com o modelo mais simples
e rápido
- tentar encontrar uma melhoria de previsão (maior r² ou menos rsme)
"""

#bloco auxiliar para avaliar modelos
def avaliar_modelo(nome_modelo, y_teste, previsao):
    start_time = time.time()
    r2 = r2_score(y_teste, previsao)
    rsme = np.sqrt(mean_squared_error(y_teste, previsao))
    end_time = time.time()
    execution_time = end_time - start_time
    return f'Modelo: {nome_modelo}\nr²: {r2:.2%}\nrsme: {rsme:.2f}\ntempo: {execution_time}'

modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {
    'RandomForest': modelo_rf,
    'LinearRegression': modelo_lr,
    'ExtraTrees': modelo_et
}

"""
resultado no terminal

Modelo RandomForest:
r²: 97.24%
rsme: 44.05
Modelo LinearRegression:
r²: 32.70%
rsme: 217.54
Modelo ExtraTrees:
r²: 97.50%
rsme: 41.89

Nesse caso o melhor modelo é o ExtraTrees por que possui o maior r² e o menor rsme.
"""

#separar os dados em treino e teste + Treino do modelo
base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)
x_test = x_test.reindex(columns=x_train.columns)
modelo_et.fit(x_train, y_train)
previsao = modelo_et.predict(x_test)

print(avaliar_modelo('ExtraTrees', y_test, previsao))


#ajustes e melhorias no melhor modelo

# print(modelo_et.feature_importances_) #grau de importancia de cada coluna
# print(x_train.columns) #colunas referentes ao dataframe ordenada pela importancia a cima

# importancia_features = pd.DataFrame(modelo_et.feature_importances_, x_train.columns)
# importancia_features = importancia_features.sort_values(by=0, ascending=False)

# plt.figure(figsize=(15, 5))
# ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
# ax.tick_params(axis='x', rotation=90)
# plt.show()

"""
de acordo com o gráfico das importancias, existem várias features(colunas) que
causam pouco ou nenhum impacto numérico, dando a possibildiade de excluir elas
para tornar o modelo cada vez mais simples
"""

joblib.dump(modelo_et, 'modelo.joblib')