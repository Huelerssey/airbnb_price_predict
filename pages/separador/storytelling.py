import streamlit as st
from streamlit_extras.colored_header import colored_header
from src.data_utility import carregar_dados_abnb
import plotly.express as px


def storytelling():

    # carregar dados
    dados = carregar_dados_abnb()
    st.markdown("<h1 style='text-align: center;'>📌 Construção do Projeto 📌</h1>", unsafe_allow_html=True)

    # marcador vermelho
    colored_header(
    label="",
    description="",
    color_name="light-blue-70"
    )

    st.header("📌 Introdução")
    st.write("O projeto que estou desenvolvendo tem como objetivo prever o preço das diárias em acomodações listadas no Airbnb na cidade do Rio de Janeiro. Utilizei um conjunto de dados obtido através da Kaggle, que contém diversas informações sobre as propriedades listadas, tais como tipo de propriedade, localização, número de quartos, comodidades, entre outros.")
    st.image("imagens/1.jpg")
    st.write("")

    st.header("📌 Obtenção dos Dados")
    st.write("O conjunto de dados utilizado neste projeto foi obtido através do Kaggle, no seguinte [link](https://www.kaggle.com/datasets/allanbruno/airbnb-rio-de-janeiro). A análise e modelagem desses dados nos permitirão entender os principais fatores que influenciam os preços das diárias e criar um modelo de machine learning capaz de prever esses preços com base nas características da propriedade.")
    st.write("")

    st.header("📌 Entendimento da Área/Negócio")
    st.write("Iniciamos nossa análise com uma seleção cuidadosa das características mais relevantes. O dataset original possuía uma variedade maior de colunas, mas optamos por focar naquelas que pareciam ter maior impacto na previsão dos preços das diárias no Airbnb. Esta etapa de avaliação qualitativa nos permitiu reduzir a dimensão dos dados e concentrar nossos esforços nas informações mais promissoras. Vamos explorar as colunas selecionadas e entender como elas podem contribuir para a nossa análise.")
    texto = """
    **host_is_superhost**: Indica se o anfitrião é considerado um "superhost" pelo Airbnb.\n
    **host_listings_count**: Número de acomodações listadas pelo anfitrião.\n
    **latitude**: Latitude da propriedade.\n
    **longitude**: Longitude da propriedade.\n
    **property_type**: Tipo da propriedade (ex.: apartamento, casa, etc.).\n
    **room_type**: Tipo de quarto oferecido (ex.: quarto inteiro, quarto compartilhado, etc.).\n
    **accommodates**: Número de pessoas que a propriedade pode acomodar.\n
    **bathrooms**: Número de banheiros na propriedade.\n
    **bedrooms**: Número de quartos na propriedade.\n
    **beds**: Número de camas na propriedade.\n
    **bed_type**: Tipo de cama (ex.: cama comum, futon, etc.).\n
    **amenities**: Lista de comodidades oferecidas (ex.: Wi-Fi, cozinha, etc.).\n
    **price**: Preço da diária.\n
    **security_deposit**: Depósito de segurança exigido.\n
    **cleaning_fee**: Taxa de limpeza.\n
    **guests_included**: Número de hóspedes incluídos no preço da diária.\n
    **extra_people**: Custo extra por pessoa adicional.\n
    **minimum_nights**: Número mínimo de noites para a reserva.\n
    **maximum_nights**: Número máximo de noites para a reserva.\n
    **number_of_reviews**: Número de avaliações da propriedade.\n
    **review_scores_rating**: Avaliação geral da propriedade.\n
    **review_scores_accuracy**: Avaliação da precisão da descrição da propriedade.\n
    **review_scores_cleanliness**: Avaliação da limpeza da propriedade.\n
    **review_scores_checkin**: Avaliação do processo de check-in.\n
    **review_scores_communication**: Avaliação da comunicação com o anfitrião.\n
    **review_scores_location**: Avaliação da localização da propriedade.\n
    **review_scores_value**: Avaliação do valor da propriedade.\n
    **instant_bookable**: Indica se a reserva instantânea está disponível.\n
    **is_business_travel_ready**: Indica se a propriedade está pronta para viagens de negócios.\n
    **cancellation_policy**: Política de cancelamento da reserva.\n
    **ano**: Ano da listagem.\n
    **mes**: Mês da listagem.\n
    """
    st.write(texto)

    st.header("📌 Limpeza e Tratamento dos Dados")
    st.write("Nesta etapa, vamos nos concentrar em limpar e preparar os dados para análise, isso incluirá a seleção das colunas relevantes, a transformação de tipos de dados e a remoção de linhas ou colunas indesejadas.")
    texto2 = """
    **Transformar os Dados**: Converteremos os dados em objetos para strings e verificaremos se os dados categóricos têm uma representatividade pequena para agrupá-los como "outros".\n
    **Remover Colunas ou Linhas em Branco**: Excluiremos colunas ou linhas que contêm valores em branco ou que não fazem sentido para a análise.\n
    **Transformar Dados Categóricos**: Convertendo valores '0' e '1' para 'False' e 'True', respectivamente, e aplicando o LabelEncoder para categóricos com mais de duas categorias.\n
    **Tratar Colunas Específicas**: As colunas 'extra_people' e 'price' devem ser 100% float, então vamos excluiremos a linha correspondente caso esteja em um formato divergente.\n
    """
    st.write(texto2)
    st.subheader("Código para tratamento de dados")
    codigo1 = """
    #deleta colunas com valores nulos acima de 30 mil
    base_airbnb = base_airbnb.loc[:,colunas]
    for coluna in base_airbnb:
        if base_airbnb[coluna].isnull().sum() > 300000:
            base_airbnb = base_airbnb.drop(coluna, axis=1)

    #exclui todos as linhas de uma coluna que possui valor none
    base_airbnb = base_airbnb.dropna() 

    # transforma dados da coluna prince em float
    base_airbnb['price'] = base_airbnb['price'].apply(lambda x: float(x.replace('$', '').replace(',', '')))

    # transforma dados da coluna extra_people em float
    base_airbnb['extra_people'] = base_airbnb['extra_people'].apply(lambda x: float(x.replace('$', '').replace(',', '')))

    # converter colunas booleanas em int (0 ou 1)
    def converter_colunas_bool(df, colunas):
        for coluna in colunas:
            df[coluna] = df[coluna].map({'t': 1, 'f': 0})
        return df

    #armazena as colunas que serão convertidas
    colunas_para_converter = ['host_is_superhost', 'instant_bookable']

    # aplica a fução de conversão
    base_airbnb = converter_colunas_bool(base_airbnb, colunas_para_converter)

    # lista de colunas que não fazem mais sentido após a análise exploratória
    colunas_excluir = [
        'instant_bookable', 'guests_included', 'maximum_nights',
        'number_of_reviews', 'is_business_travel_ready', 'host_listings_count',
        'minimum_nights', 'bed_type', 'cancellation_policy'
    ]

    # excluir colunas selecionadas
    base_airbnb = base_airbnb.drop(columns=colunas_excluir)
    """
    st.code(codigo1, language='python')
    st.write("")

    st.header("📌 Análise Exploratória de Dados")
    st.write("A análise exploratória irá revela insights valiosos sobre a distribuição dos preços, a relação entre características diferentes e a presença de outliers. Através de visualizações e análises estatísticas, os dados vão ser preparados para a modelagem da inteligência artificial.")
    st.write("Por questões de organização, costumo criar um bloco inteiro com funções auxiliares para fazer as análiese e depois aplica-los aos dados. Aqui está a sessão dedicada as funções e gráficos:")
    codigo2 = """
    ## FUNÇÕES AUXILIARES ##

    # retorna o limite inferior e o limite superior
    def limites(coluna):
        q1 = coluna.quantile(0.25)
        q3 = coluna.quantile(0.75)
        amplitude = q3 - q1
        return (q1 - 1.5 * amplitude, q3 + 1.5 * amplitude)

    # Exclui outliers e retorna o novo dataframe e também a quantidade de linhas removidas
    def excluir_outliers(df, nome_coluna):
        qtde_linhas = df.shape[0]
        lim_inf, lim_sup = limites(df[nome_coluna])
        df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), : ]
        linhas_removidas = qtde_linhas - df.shape[0]
        return df, linhas_removidas

    # plota um diagrama de caixa com as informações da coluna já entre o limite
    def diagrama_caixa(coluna):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(15,5)
        sns.boxplot(x=coluna, ax=ax1)
        ax2.set_xlim(limites(coluna))
        sns.boxplot(x=coluna, ax=ax2)
        return plt.show()

    # plota um histograma da coluna passada como parâmetro
    def histograma(coluna):
        plt.figure(figsize=(15, 5))
        sns.distplot(coluna, hist=True)
        return plt.show()

    # plota um gráfico de barras com os dados da coluna já entre os dois limites
    def grafico_barra(coluna):
        plt.figure(figsize=(15, 5))
        ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
        ax.set_xlim(limites(coluna))
        return plt.show()

    # plota um gráfico de barras para auxiliar a avaliação em colunas de texto
    def grafico_aux_txt(data, coluna: str, figsize=(15, 5)):
        print(base_airbnb[coluna].value_counts())
        plt.figure(figsize=figsize)
        grafico = sns.countplot(data=data, x=coluna)
        grafico.tick_params(axis='x', rotation=90)
        return plt.show()

    ## FUNÇÕES AUXILIARES ##
    """
    st.code(codigo2, language='python')
    st.write("Com as funções auxiliares construidas, é hora de colocar a mão na massa!")
    st.write("")

    st.header("📌 Modelando a Inteligência Artificial")
    st.write("")

    st.header("📌 Apresentação de Resultados")
    st.write("")

    st.header("📌 Escolhendo o Melhor Modelo e Colocando em Produção")
    st.write("")
