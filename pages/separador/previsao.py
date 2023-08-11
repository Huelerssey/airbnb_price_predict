import streamlit as st
from src.data_utility import carregar_modelo_abnb


def previsao():

    #carregar modelo
    modelo = carregar_modelo_abnb()
    st.write("Hello world 2")