import streamlit as st
import json
from streamlit_lottie import st_lottie


def home():
    
    # Colunas que organizam a página
    col1, col2 = st.columns(2)

    # animações
    with open("animacoes/pagina_inicial1.json") as source:
        animacao_1 = json.load(source)

    with open("animacoes/pagina_inicial2.json") as source:
        animacao_2 = json.load(source)
    
    # conteúdo a ser exibido na coluna 1
    with col1:
        st_lottie(animacao_1, height=350, width=400)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("<h5 style='text-align: justify;'> Ao longo deste projeto, vamos aprender mais sobre a ciência de dados aplicada ao mercado imobiliário. Se você está interessado em saber como a tecnologia pode desvendar os mistérios dos preços, este projeto é para você. Vamos embarcar juntos nesta jornada de descoberta!</h5>", unsafe_allow_html=True)

    # conteúdo a ser exibido na coluna 2
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("<h5 style='text-align: justify;'> Bem-vindo ao projeto de Previsão de Preços de Diárias no Airbnb no Rio de Janeiro! Neste projeto, vamos explorar o desafio de prever o preço das acomodações usando dados reais e técnicas avançadas de Machine learning.</h5>", unsafe_allow_html=True)
        st_lottie(animacao_2, height=400, width=440)