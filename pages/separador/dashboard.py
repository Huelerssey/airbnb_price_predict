import streamlit as st
from src.data_utility import carregar_dados_abnb


def dashboard():

    # carregar dados
    dados = carregar_dados_abnb()
    st.write("Hello world 4")