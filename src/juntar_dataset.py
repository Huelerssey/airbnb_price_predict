import pandas as pd
import pathlib


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

# salva um arquivo csv com todas planilhas em uma só
base_airbnb.to_csv("dataset/dataset_final.csv")