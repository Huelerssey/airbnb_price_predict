import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregando o arquivo CSV do conjunto de dados modelado
full_dataset_csv = pd.read_pickle("arquivos_pkl/dataset_modelado_pkl.pkl")

# Separando as características e o alvo
X_full_csv = full_dataset_csv.drop(columns=['price'])
y_full_csv = full_dataset_csv['price']

# Dividindo os dados em conjuntos de treinamento e teste
X_train_full_csv, X_test_full_csv, y_train_full_csv, y_test_full_csv = train_test_split(X_full_csv, y_full_csv, test_size=0.2, random_state=42)

# Função para treinar e avaliar um modelo
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')
    return mse, r2

# Modelos
models = {
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Lasso': Lasso(),
    'Linear Regression': LinearRegression()
}

# Treinando e avaliando cada modelo
for model_name, model_instance in models.items():
    print(f'\nEvaluating {model_name}:')
    train_and_evaluate_model(model_instance, X_train_full_csv, y_train_full_csv, X_test_full_csv, y_test_full_csv)

# Evaluating Extra Trees:
# Mean Squared Error: 1602.224437762044
# R2 Score: 0.9771786638840028

# Evaluating Random Forest:
# Mean Squared Error: 1714.246310130716
# R2 Score: 0.9755830766857201

# Evaluating Lasso:
# Mean Squared Error: 48425.88675110814
# R2 Score: 0.31024430022681393

# Evaluating Linear Regression:
# Mean Squared Error: 47555.81640671667
# R2 Score: 0.32263717559804106