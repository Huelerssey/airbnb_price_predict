import streamlit as st
import pandas as pd
from babel.numbers import format_currency
from src.data_utility import carregar_modelo_abnb
from src.data_utility import carregar_dados_abnb
from streamlit_extras.colored_header import colored_header
import plotly.express as px


def previsao():

    #carregar dados
    dataset = carregar_dados_abnb()

    #carregar modelo
    modelo = carregar_modelo_abnb()

    # conteiner do titulo
    with st.container():
        
        #titulo
        st.markdown("<h1 style='text-align: center;'>📊 Modelo de Previsão de Preço 📊</h1>", unsafe_allow_html=True)

    # marcador azul
    colored_header(
    label="",
    description="",
    color_name="light-blue-70"
    )

    with st.container():

        # cria duas colunas
        coluna1, coluna2 = st.columns(2)

        # coluna 1
        with coluna1:

            # host is superhost
            host_is_superhost_selection = st.selectbox("O anfitrião é um Superhost?", ["Sim", "Não"])

            # Convertendo para 0 ou 1
            host_is_superhost = 1 if host_is_superhost_selection == "Sim" else 0

            # latitude
            latitude = st.number_input("Latitude (6 digitos):", value=-00.0000)

            #longitude
            longitude = st.number_input("Longitude (6 digitos):", value=-00.0000)

            # Numero de hóspedes
            accommodates = st.number_input("Número de Hóspedes:", value=1)

            # numero de banheiros
            bathrooms = st.number_input("Número de Banheiros:", value=1)

            # numero de quartos
            bedrooms = st.number_input("Número de Quartos:", value=1)

            # numero de camas
            beds = st.number_input("Número de Camas:", value=1)

            # custo por pessoa extra
            extra_people = st.number_input("Máximo de Preço por Pessoa Extra que pretende Pagar:", value=0.0)

            # ano
            ano = st.number_input("Ano (entre 2018 a 2020):", value=2018)

            # mes
            mes_selection = st.selectbox("Mês:", [
                "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
                "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
            ])

            # Convertendo para o valor numérico correspondente
            mes = mes_selection.index(mes_selection) + 1 

            # numero de comodidades
            n_amenities = st.number_input("Número de Comodidades:", value=1)
            
            # tipo de propriedade
            property_type = st.selectbox("Tipo de Propriedade:", [
                'Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite', 'Guesthouse', 'Hostel', 'House', 'Loft', 'Other', 'Serviced apartment'
            ])
            
            # tipo de quarto
            room_type = st.selectbox("Tipo de Quarto:", [
                'Entire home/apt', 'Hotel room', 'Private room', 'Shared room'
            ])
            
            # Criando a lista de colunas para tipos de propriedade 
            property_type_columns = [
                'property_type_Apartment', 'property_type_Bed and breakfast', 'property_type_Condominium', 'property_type_Guest suite', 'property_type_Guesthouse', 'property_type_Hostel', 'property_type_House', 'property_type_Loft', 'property_type_Other', 'property_type_Serviced apartment'
            ]

            # Criando a lista de colunas para tipos de quarto
            room_type_columns = [
                'room_type_Entire home/apt', 'room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room'
            ]
            
            # Criando um DataFrame com os dados de entrada
            input_data = pd.DataFrame({
                'host_is_superhost': [host_is_superhost],
                'latitude': [latitude],
                'longitude': [longitude],
                'accommodates': [accommodates],
                'bathrooms': [bathrooms],
                'bedrooms': [bedrooms],
                'beds': [beds],
                'extra_people': [extra_people],
                'ano': [ano],
                'mes': [mes],
                'n_amenities': [n_amenities],
                **dict.fromkeys(property_type_columns, 0),
                **dict.fromkeys(room_type_columns, 0),
            })

            input_data[f'property_type_{property_type}'] = 1
            input_data[f'room_type_{room_type}'] = 1

            # Botão para fazer a previsão
            if st.button("Fazer Previsão"):
                # Fazendo a previsão com o modelo
                preco_previsto = modelo.predict(input_data)[0]

                # Formatar o valor como moeda brasileira
                preco_formatado = format_currency(preco_previsto, 'BRL', locale='pt_BR')

                # Exibir o resultado
                st.success(f"O preço desta diária deveria custar: {preco_formatado}")

        with coluna2:

            st.subheader("Gráfico Auxiliar de Referencia (Lat e Long)")
            
            dados = dataset.sample(n=10000)
            # Mapa de dispersão das propriedades com base em latitude e longitude
            fig_location = px.scatter_mapbox(dados, lat="latitude", lon="longitude", color="price",
                                            color_continuous_scale=px.colors.sequential.Viridis,
                                            size_max=15, zoom=10, title=None,
                                            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'price': 'Preço'})

            # Configuração do estilo do mapa
            fig_location.update_layout(mapbox_style="carto-positron")

            st.plotly_chart(fig_location)
    