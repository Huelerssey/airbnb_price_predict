import streamlit as st
import json
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import pages.separador.pagina_1 as PaginaUm
import pages.separador.pagina_2 as PaginaDois
import pages.separador.pagina_3 as PaginaTres


# configurações da pagina
st.set_page_config(
    page_title='MultiPagesAPP',
    #https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app
    page_icon='✅',
    layout='wide'
)

#aplicar estilos de css a pagina
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# animações
with open("animacoes/animacao_lottie.json") as source:
    animacao_1 = json.load(source)

# Menu de navegação lateral
with st.sidebar:
    #exibir animação
    st_lottie(animacao_1, height=100, width=270)
    st.write("---")
    #https://github.com/victoryhb/streamlit-option-menu
    opcao_selecionada = option_menu(
        #https://icons.getbootstrap.com
        menu_title="Menu Inicial",
        menu_icon="justify",
        options=["Página 1", "Página 2", "Página 3", "---"],
        icons=['bookmark', 'bookmark', 'bookmark'],
        default_index=0,
        orientation='vertical',
        # styles={
        #     "container": {"padding": "0!important", "background-color": "#1a202c"},  # azul escuro
        #     "icon": {"color": "#a0aec0", "font-size": "25px"},  # azul claro
        #     "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "color": "#a0aec0", "--hover-color": "#a0aec0"},  # azul claro
        #     "nav-link-selected": {"background-color": "#2d3748", "color": "white"},  # azul mais escuro e branco
        # }
    )
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
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # footer da barra lateral
    st.write("---")
    st.markdown("<h5 style='text-align: center; color: lightgray;'>Developed By: Huelerssey Rodrigues</h5>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between;">
        <div>
            <a href="https://github.com/Huelerssey" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" width="100" />
            </a>
        </div>
        <div>
            <a href="https://www.linkedin.com/in/huelerssey-rodrigues-a3145a261/" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" width="100" />
            </a>
        </div>
        <div>
            <a href="https://api.whatsapp.com/send?phone=5584999306130" target="_blank">
                <img src="https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white" width="100" />
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Retorna a pagina 1
if opcao_selecionada == "Página 1":
    PaginaUm.pagina1()

# Retorna a pagina 2
elif opcao_selecionada == "Página 2":
    PaginaDois.pagina2()

# Retorna a pagina 3
elif opcao_selecionada == "Página 3":
    PaginaTres.pagina3()