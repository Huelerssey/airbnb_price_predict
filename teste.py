import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import joblib
from sklearn.model_selection import train_test_split


base_airbnb = pd.read_pickle("arquivos_pkl/dataset_airbnb_modelado.pkl")

## DOMINUIR COMPLEXIDADE DO MODELO ##



# definindo dados de treino e teste
y = base_airbnb['price']
x = base_airbnb.drop('price', axis=1)

# dividindo a base entre treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2 ,random_state=42)

# cria o modelo
modelo_extratrees = ExtraTreesRegressor(n_estimators=100, random_state=42)

# treina o modelo
modelo_extratrees.fit(x_treino, y_treino)

# testa o modelo
y_pred = modelo_extratrees.predict(x_teste)

#armazena o modelo treinado para produção
joblib.dump(modelo_extratrees, "arquivos_pkl/teste.pkl")