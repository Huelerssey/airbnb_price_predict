import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


base_airbnb = pd.read_pickle("arquivos_pkl/dataset_airbnb_modelado.pkl")

## DOMINUIR COMPLEXIDADE DO MODELO ##

# Index(['host_is_superhost', 'host_listings_count', 'latitude', 'longitude',
#        'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price',
#        'extra_people', 'minimum_nights', 'ano', 'mes', 'n_amenities',
#        'property_type_Apartment', 'property_type_Bed and breakfast',
#        'property_type_Condominium', 'property_type_Guest suite',
#        'property_type_Guesthouse', 'property_type_Hostel',
#        'property_type_House', 'property_type_Loft', 'property_type_Other',
#        'property_type_Serviced apartment', 'room_type_Entire home/apt',
#        'room_type_Hotel room', 'room_type_Private room',
#        'room_type_Shared room', 'cancellation_policy_flexible',
#        'cancellation_policy_moderate', 'cancellation_policy_strict',
#        'cancellation_policy_strict_14_with_grace_period', 'bed_type_Other',
#        'bed_type_Real Bed'],
#       dtype='object')

# colunas para exluir como teste
colunas_excluir = [
    "price", 'cancellation_policy_flexible',
    'cancellation_policy_moderate', 'cancellation_policy_strict',
    'cancellation_policy_strict_14_with_grace_period', 'bed_type_Other',
    'bed_type_Real Bed', 'host_is_superhost', 'host_listings_count', 'minimum_nights'
    ]

# definindo dados de treino e teste
y = base_airbnb['price']
x = base_airbnb.drop(columns=colunas_excluir)

# dividindo a base entre treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3 ,random_state=42)

# cria o modelo
modelo_extratrees = ExtraTreesRegressor(n_estimators=100, random_state=42)

# treina o modelo
modelo_extratrees.fit(x_treino, y_treino)

# testa o modelo
y_pred = modelo_extratrees.predict(x_teste)

mse = mean_squared_error(y_teste, y_pred)
r2 = r2_score(y_teste, y_pred)
print(f'Erro Quadrático Médio: {mse}')
print(f'Coeficiente de Determinação (R2): {r2}')

#armazena o modelo treinado para produção
joblib.dump(modelo_extratrees, "arquivos_pkl/teste.joblib")