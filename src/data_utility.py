import pandas as pd
import streamlit as st
from joblib import load


# função que otimiza o carregamento dos dados da tabela
@st.cache_data
def carregar_dados_abnb():
    tabela_abnb = pd.read_pickle("arquivos_pkl/dataset_airbnb_modelado.pkl")
    return tabela_abnb

# função que otimiza o carregamento dos dados do modelo
@st.cache_data
def carregar_modelo_abnb():
    modelo_abnb = load("arquivos_pkl/dataset_airbnb_modelado.pkl")
    return modelo_abnb
