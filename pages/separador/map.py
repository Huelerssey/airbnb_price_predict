import streamlit as st
from src.data_utility import carregar_dados_abnb
from streamlit_extras.colored_header import colored_header


def map():
    
    #carregar dados
    dados = carregar_dados_abnb()
    
    #titulo
    st.markdown("<h1 style='text-align: center;'>🗺️ Distribuição de Propriedades no Airbnb RJ 🗺️</h1>", unsafe_allow_html=True)

    # marcador azul
    colored_header(
    label="",
    description="",
    color_name="light-blue-70"
    )

    # Organização da página
    col1, col2, col3 = st.columns(3)

    # Opções do selectbox
    opcoes_graficos = [
        'Mapa de Cluster - Rio de Janeiro',
        'Mapa de Calor (Heatmap) - Rio de Janeiro',
    ]

    with col2:
        # Selectbox para selecionar o mapa a ser exibido
        grafico_selecionado = st.selectbox('Selecione o mapa a ser exibido', opcoes_graficos, label_visibility='hidden')
    with st.container():
        def exibir_mapa(caminho_arquivo):
            # Abre o arquivo HTML e lê o conteúdo como uma string
            with open(caminho_arquivo, 'r') as arquivo:
                conteudo_html_mapa = arquivo.read()
        
            # Cria um contêiner div com largura e altura de 100% e coloca o conteúdo HTML do mapa dentro dele
            conteudo_html = f"""<div style="width:100%; height:100%">{conteudo_html_mapa}</div>"""

            # Exibe o mapa no Streamlit usando o HTML
            st.components.v1.html(conteudo_html, scrolling=True)
            st.markdown("""
            <style>
            .main iframe {
                width: 100%;
                min-height: 60vh;
                height: 100vh:
            }
            </style>
            """, unsafe_allow_html=True)

    # Mapeia a opção selecionada para o caminho do arquivo correspondente
    mapeamento_mapas = {
        'Mapa de Cluster - Rio de Janeiro': 'mapas/cluster_map.html',
        'Mapa de Calor (Heatmap) - Rio de Janeiro': 'mapas/heat_map.html',      
    }

    # Verifica se a opção selecionada é válida e exibe o mapa correspondente
    if grafico_selecionado in mapeamento_mapas:
        caminho_arquivo = mapeamento_mapas[grafico_selecionado]
        exibir_mapa(caminho_arquivo)
    else:
        st.warning('Selecione uma opção válida.')
