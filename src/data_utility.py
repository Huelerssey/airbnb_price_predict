import pandas as pd
import streamlit as st


# função que otimiza o carregamento dos dados
@st.cache_data
def carregar_dados():
    tabela = pd.read_pickle("arquivos_pkl/nome_do_arquivo.pkl")
    return tabela
