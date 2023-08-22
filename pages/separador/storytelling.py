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
    st.write("")

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
    codigo3 = """
    ## COLUNAS NUMÉRICAS ##

    #price
    diagrama_caixa(base_airbnb['price'])
    histograma(base_airbnb['price'])
    base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
    print(f'{linhas_removidas} linhas removidas da coluna price')

    #extra_people
    diagrama_caixa(base_airbnb['extra_people'])
    histograma(base_airbnb['extra_people'])
    base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
    print(f'{linhas_removidas} linhas removidas da coluna extra_people')

    #host_listings_count
    diagrama_caixa(base_airbnb['host_listings_count'])
    grafico_barra(base_airbnb['host_listings_count'])
    # movido para etapa de limpeza (foi excluido)

    #accommodates
    diagrama_caixa(base_airbnb['accommodates'])
    grafico_barra(base_airbnb['accommodates'])
    base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
    print(f'{linhas_removidas} linhas removidas da coluna accommodates')

    #bathrooms
    diagrama_caixa(base_airbnb['bathrooms'])
    grafico_barra(base_airbnb['bathrooms'])
    base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
    print(f'{linhas_removidas} linhas removidas da coluna bathrooms')

    #bedrooms
    diagrama_caixa(base_airbnb['bedrooms'])
    grafico_barra(base_airbnb['bedrooms'])
    base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
    print(f'{linhas_removidas} linhas removidas da coluna bedrooms')

    #beds
    diagrama_caixa(base_airbnb['beds'])
    grafico_barra(base_airbnb['beds'])
    base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
    print(f'{linhas_removidas} linhas removidas da coluna beds')

    #guests_included
    diagrama_caixa(base_airbnb['guests_included'])
    grafico_barra(base_airbnb['guests_included'])
    # print(limites(base_airbnb['guests_included']))
    plt.figure(figsize=(15, 5))
    sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())
    plt.show()
    # movido para etapa de limpeza (foi excluido)

    #minimum_nights
    diagrama_caixa(base_airbnb['minimum_nights'])
    grafico_barra(base_airbnb['minimum_nights'])
    # movido para etapa de limpeza (foi excluido)

    #maximum_nights
    diagrama_caixa(base_airbnb['maximum_nights'])
    grafico_barra(base_airbnb['maximum_nights'])
    # movido para etapa de limpeza (foi excluido)

    #number_of_reviews
    diagrama_caixa(base_airbnb['number_of_reviews'])
    grafico_barra(base_airbnb['number_of_reviews'])
    # movido para etapa de limpeza (foi excluido)

    ## COLUNAS DE TEXTO ##

    # print(base_airbnb.dtypes)
    # print('-'*60)
    # print(base_airbnb.iloc[0])

    #property_type
    #print(base_airbnb['property_type'].value_counts()) conta quantos valores existem para cada tipo de texto
    grafico_aux_txt(base_airbnb, 'property_type')

    tabela_tipos_casa = base_airbnb['property_type'].value_counts() #descreve a categoria e seu valor
    #print(tabela_tipos_casa.index) descreve apenas as caregorias
    #print(tabela_tipos_casa['Apartament']) descreve os valores da categoria passada

    # Agrupa todos as categorias com valor menor que 2000 em uma lista
    colunas_agrupar = []
    for tipo in tabela_tipos_casa.index:
        if tabela_tipos_casa[tipo] < 2000:
            colunas_agrupar.append(tipo)

    # inserir todos os valores da lista colunas_agrupar na categoria outros da coluna property_type
    for tipo in colunas_agrupar:
        base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Other'

    #room_type
    grafico_aux_txt(base_airbnb, 'room_type')

    #bed_type
    grafico_aux_txt(base_airbnb, 'bed_type')
    # movido para etapa de limpeza (foi excluido)

    #cancellation_policy
    grafico_aux_txt(base_airbnb, 'cancellation_policy')
    # movido para etapa de limpeza (foi excluido)

    #amenities
    # criar uma nova coluna composta apenas pela quantidade de amenities
    base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)

    # deletar a coluna antiga que foi substituida
    base_airbnb = base_airbnb.drop('amenities', axis=1)

    diagrama_caixa(base_airbnb['n_amenities'])
    grafico_barra(base_airbnb['n_amenities'])
    base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
    print(f'{linhas_removidas} linhas removidas da coluna n_amenities')

    ## PARA COLUNAS CATEGÓRICAS ##

    # transforma colunas categóricas em numéricas
    colunas_categoria = ['property_type', 'room_type']
    base_airbnb = pd.get_dummies(data=base_airbnb, columns=colunas_categoria)
    """
    st.code(codigo3, language='python')
    
    st.write("---")
    # price
    st.markdown("<h4 style='text-align: center;'>Coluna de Preço</h4>", unsafe_allow_html=True)
    st.image("imagens/preco_1.png")
    st.image("imagens/preco_2.png")
    st.write("O gráfico mostra a mediana dos preços em torno de 200, com a maioria dos preços concentrados entre aproximadamente 50 e 400 R$. Existem muitos outliers acima do limite superior (bigode superior), indicando listagens com preços muito mais altos do que a média. Por esse motivo, usamos a função auxiliar para exclui-los.")
    
    st.write("---")
    # extra_people
    st.markdown("<h4 style='text-align: center;'>Coluna de Preço por Pessoa Extra</h4>", unsafe_allow_html=True)
    st.image("imagens/extra_people_1.png")
    st.image("imagens/extra_people_2.png")
    st.write("A mediana dos custos adicionais por pessoa extra é $0, o que indica que muitos anfitriões não cobram um custo adicional por hóspedes extras. A maioria dos custos para pessoas extras está concentrada entre $0 e $50 e existem alguns outliers acima do limite superior, indicando algumas listagens que cobram significativamente mais por hóspedes extras. Dado o grande número de listagens que não cobram por pessoas extras e a presença de valores mais altos, vamos permanecer com a coluna mas excluindo os outliers.")

    st.write("---")
    # host_listings_count
    st.markdown("<h4 style='text-align: center;'>Quantidade de Porpriedades de um Anfitrião</h4>", unsafe_allow_html=True)
    st.image("imagens/host_listing_1.png")
    st.image("imagens/host_listing_2.png")
    st.write("A mediana da quantidade de listagens por anfitrião é 1, indicando que muitos anfitriões têm apenas uma listagem no Airbnb, o que torna sem sentido continuar com esta coluna em nossa análise, assim decidimos exclui-la")

    st.write("---")
    # accommodates
    st.markdown("<h4 style='text-align: center;'>Número de Comodidades</h4>", unsafe_allow_html=True)
    st.image("imagens/accommodates_1.png")
    st.image("imagens/accommodates_2.png")
    st.write("A mediana do número de acomodações (número de pessoas que uma listagem pode acomodar) é 2, indicando que muitos espaços são projetados para acomodar duas pessoas. Existem alguns outliers acima do limite superior, indicando algumas listagens que podem acomodar um número significativamente maior de pessoas. Dado que a capacidade de acomodação é uma característica importante ao decidir um espaço para alugar, vamos manter essa coluna mas excluir os outliers.")

    st.write("---")
    # bathrooms
    st.markdown("<h4 style='text-align: center;'>Número de Banheiros</h4>", unsafe_allow_html=True)
    st.image("imagens/bathrooms_1.png")
    st.image("imagens/bathrooms_2.png")
    st.write("A mediana do número de banheiros é 1, o que indica que muitas listagens possuem apenas um banheiro, com a maioria das listagens possui entre 1 e 2 banheiros. Existe alguns outliers acima do limite superior, mostrando algumas propriedades com um número significativamente maior de banheiros. Como o número de banheiros em uma propriedade é geralmente uma característica importante para os hóspedes, pois pode afetar o conforto e a conveniência, especialmente para grupos maiores. Portanto, vamos excluir os outliers e manter a coluna.")

    st.write("---")
    # bedrooms
    st.markdown("<h4 style='text-align: center;'>Número de Quartos</h4>", unsafe_allow_html=True)
    st.image("imagens/bedrooms_1.png")
    st.image("imagens/bedrooms_2.png")
    st.write("A mediana do número de quartos é 1, o que sugere que muitas listagens são de unidades de um quarto e existem alguns outliers acima do limite superior, indicando algumas propriedades com um número significativamente maior de quartos. O número de quartos, como o número de banheiros, é uma característica crucial para muitos hóspedes ao decidir sobre um espaço de aluguel. Quartos adicionais podem oferecer mais privacidade e conforto para grupos maiores, o que pode influenciar o preço da diária, então vamos manter essa coluna e excluir os outliers.")

    st.write("---")
    # beds
    st.markdown("<h4 style='text-align: center;'>Número de Camas</h4>", unsafe_allow_html=True)
    st.image("imagens/beds_1.png")
    st.image("imagens/beds_2.png")
    st.write("A distribuição mostra que a maioria das listagens tem 1 cama, seguida por aquelas com 2 camas. O número de camas é diretamente relacionado à capacidade de acomodação de uma propriedade e é uma informação crucial para hóspedes que viajam em grupos ou famílias. Por exemplo, um casal pode procurar uma listagem com uma cama de casal, enquanto um grupo de amigos pode preferir várias camas individuais. Esta variável é, portanto, importante ao determinar os preços das diárias. Tendo isso em mente, vamos manter essa coluna e excluir os outliers.")

    st.write("---")
    # guests_included
    st.markdown("<h4 style='text-align: center;'>Número de Hóspedes</h4>", unsafe_allow_html=True)
    st.image("imagens/guest_included_1.png")
    st.image("imagens/guest_included_2.png")
    st.write("considerando que muitas listagens incluem apenas 1 ou 2 hóspedes, em vez de uma distribuição variada, não faz sentido mante-la em nossa análise então ela será excluida.")

    st.write("---")
    # minimum_nights
    st.markdown("<h4 style='text-align: center;'>Noites Mínimas</h4>", unsafe_allow_html=True)
    st.image("imagens/minimum_nights_1.png")
    st.image("imagens/minimum_nights_2.png")
    st.write("O número mínimo de noites é uma política estabelecida pelo anfitrião e pode variar dependendo de várias razões, incluindo a localização da propriedade, a época do ano ou as preferências pessoais do anfitrião. No entanto, com base na distribuição observada, a maior parte dos anfitriões parecem favorecer estadias curtas, tornando a amostra muito homogênea e desnecessário para a nossa análise, por isso ela será excluida.")

    st.write("---")
    # maximum_nights
    st.markdown("<h4 style='text-align: center;'>Noites Máximas</h4>", unsafe_allow_html=True)
    st.image("imagens/maximum_nights_1.png")
    st.image("imagens/maximum_nights_2.png")
    st.write("Temos um pico claro em torno do valor máximo, indicando que muitos anfitriões definem um valor padrão muito alto para o número máximo de noites (provavelmente para efetivamente não ter um limite). No entanto, a presença de valores tão altos pode não ser significativa para a nossa modelagem, por isso vamos excluir essa coluna também.")

    st.write("---")
    # number_of_reviews
    st.markdown("<h4 style='text-align: center;'>Numero de Reviws</h4>", unsafe_allow_html=True)
    st.image("imagens/number_of_reviews_1.png")
    st.image("imagens/number_of_reviews_2.png")
    st.write("O número de avaliações pode ser um indicador da popularidade ou confiabilidade de uma listagem. Listagens com muitas avaliações podem ser vistas como mais confiáveis ou populares entre os hóspedes. No entanto, como o tempo de existência do imóvel no airbnb favorece os mais antigos, vamos excluir essa coluna.")

    st.write("---")
    # amenities
    st.markdown("<h4 style='text-align: center;'>Numero de Comodidades</h4>", unsafe_allow_html=True)
    st.image("imagens/n_amenities_1.png")
    st.image("imagens/n_amenities_2.png")
    st.write("A mediana do número de comodidades (amenities) é aproximadamente 18, indicando que muitas listagens oferecem em torno de 18 comodidades. Um número maior de comodidades pode ser um indicativo de uma listagem mais luxuosa ou bem equipada, o que pode afetar o preço da diária, sendo assim, vamos manter essa coluna e excluir os outliers.")

    st.write("---")
    # property_type
    st.markdown("<h4 style='text-align: center;'>Tipo de Propriedade</h4>", unsafe_allow_html=True)
    st.image("imagens/propety_type_1.png")
    st.write("Os tipos de propriedade mais comuns no conjunto de dados são 'Apartment' e 'House', que representam a grande maioria das listagens. Como a variedade de tipos de propriedades no Airbnb é vasta, os tipos de propriedade que têm uma contagem muito baixa serão agrupadas na categoria 'Other' para simplificação.")

    st.write("---")
    # room_type
    st.markdown("<h4 style='text-align: center;'>Tipo de Quarto</h4>", unsafe_allow_html=True)
    st.image("imagens/room_type_1.png")
    st.write("O tipo de quarto mais comum no conjunto de dados é 'Entire home/apt', indicando que muitos anfitriões oferecem a propriedade inteira para os hóspedes. Como o tipo de quarto é uma característica crucial para muitos hóspedes, vamos manter essa coluna.")

    st.write("---")
    # bed_type
    st.markdown("<h4 style='text-align: center;'>Tipo de Cama</h4>", unsafe_allow_html=True)
    st.image("imagens/bed_type_1.png")
    st.write("O tipo de cama é uma característica que pode influenciar a experiência do hóspede. Uma 'Real Bed' é geralmente preferida por muitos hóspedes por oferecer um maior conforto. No entanto, não temos uma distribuição significativa dos tipos então não faz sentido mantê-la em nossa análise, por isso ela será excluída.")

    st.write("---")
    # cancellation_policy
    st.markdown("<h4 style='text-align: center;'>Política de Cancelamento</h4>", unsafe_allow_html=True)
    st.image("imagens/cancellation_policy_1.png")
    st.write("A política de cancelamento  determina as condições sob as quais os hóspedes podem cancelar suas reservas sem penalidades. Algumas políticas são mais flexíveis, permitindo cancelamentos até um dia antes da chegada, enquanto outras são mais estritas, exigindo um aviso prévio maior e aplicando penalidades mais severas por cancelamento, mas não há um critério rigoroso para aplica-la a uma propriedade. Então para simplificar o nosso modelo, ela também será excluida!")
    st.write("")

    st.header("📌 Modelando a Inteligência Artificial")
    st.write("Após a fase de preparação e limpeza dos dados, chegamos ao coração do nosso projeto: a modelagem de machine learning. Esta etapa é essencial, pois é aqui que aplicamos algoritmos para aprender os padrões nos dados e fazer previsões precisas. Decidi experimentar uma variedade de algoritmos de aprendizado de máquina, incluindo a Regressão Linear, o algoritmo Lasso, as Árvores Extras e a Floresta Aleatória. Cada um desses algoritmos tem suas próprias características e pode se destacar de maneiras diferentes dependendo da natureza dos dados.")
    st.write("Em vez de treinar e avaliar cada modelo separadamente, criamos uma função personalizada que assume a responsabilidade de treinar, testar e avaliar todos os modelos de uma só vez. Isso não apenas otimiza nosso tempo, mas também garante uma abordagem padronizada na avaliação de cada modelo.")
    codigo4 = """
    # Função para treinar e avaliar um modelo
    def treinar_e_avaliar_modelo(modelo, X_treino, y_treino, X_teste, y_teste):
        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_teste)
        mse = mean_squared_error(y_teste, y_pred)
        r2 = r2_score(y_teste, y_pred)
        print(f'Erro Quadrático Médio: {mse}')
        print(f'Coeficiente de Determinação (R2): {r2}')
        return mse, r2

    # Modelos
    modelos = {
        'Árvores Extras': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Floresta Aleatória': RandomForestRegressor(n_estimators=100, random_state=42),
        'Lasso': Lasso(),
        'Regressão Linear': LinearRegression()
    }

    # Treinando e avaliando cada modelo
    for nome_modelo, instancia_modelo in modelos.items():
        print(f'\nAvaliando {nome_modelo}:')
        treinar_e_avaliar_modelo(instancia_modelo, X_treino_completo_csv, y_treino_completo_csv, X_teste_completo_csv, y_teste_completo_csv)
    """
    st.code(codigo4, language='python')
    st.write("Com os resultados em mãos, podemos analisar o desempenho de cada modelo e fazer uma escolha informada sobre qual deles levar adiante para a etapa de produção. Esta será a versão final que estará disponível para os usuários finais, pronta para fazer previsões precisas e informadas sobre os preços das diárias no Airbnb.")
    st.write("")

    st.header("📌 Apresentação de Resultados")
    st.write("Após a modelagem, chegamos à etapa crucial de avaliar e apresentar os resultados. Nesta fase, revisamos o desempenho de cada modelo para entender sua eficácia em prever os preços das diárias no Airbnb. Utilizamos duas métricas chave: o Erro Quadrático Médio (MSE), que nos dá uma medida da diferença entre as previsões e os valores reais, e o Coeficiente de Determinação (R2), que indica a proporção da variância dos preços que é previsível a partir das características.")
    st.write("Para facilitar a compreensão e comparação dos modelos, optamos por apresentar o R2  em percentual. Isso nos dá uma ideia clara de quão bem cada modelo se ajusta aos dados. Um  R2 de 100% indicaria um ajuste perfeito.")
    st.image("imagens/resultados.png")
    st.write("")

    st.header("📌 Escolhendo o Melhor Modelo e Colocando em Produção")
    st.write("Após a rigorosa avaliação de vários modelos, chegamos à etapa final: selecionar o melhor modelo e prepará-lo para uso em produção. Esta é uma etapa crucial, pois o modelo escolhido será o que estará disponível para os usuários finais, fornecendo previsões em tempo real.")
    st.write("Com base nos resultados anteriores, o modelo de 'Árvores Extras' destacou-se como o mais promissor, apresentando um excelente equilíbrio entre precisão e eficiência. Decidimos, portanto, adotá-lo como nosso modelo final.")
    codigo5 = """
    # definindo dados de treino e teste
    y = base_airbnb['price']
    x = base_airbnb.drop('price', axis=1)

    # dividindo a base entre treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2 ,random_state=42)

    # cria o modelo
    modelo_extratrees = ExtraTreesRegressor(n_estimators=10, random_state=42)

    # treina o modelo
    modelo_extratrees.fit(x_treino, y_treino)

    # testa o modelo
    y_pred = modelo_extratrees.predict(x_teste)

    #avalia o modelo
    mse = mean_squared_error(y_teste, y_pred)
    r2 = r2_score(y_teste, y_pred)
    print(f'Erro Quadrático Médio: {mse}')
    print(f'Coeficiente de Determinação (R2): {r2}')

    #armazena o modelo treinado para produção
    joblib.dump(modelo_extratrees, "arquivos_pkl/modelo_airbnb_treinado.pkl")
    """
    st.code(codigo5, language='python')
    st.write("Com isso, concluímos a etapa de implementação do modelo em produção. Agora, o modelo está pronto para ser integrado a qualquer aplicação ou plataforma, fornecendo previsões de preços de diárias no Airbnb com base nas características fornecidas.")

    # marcador azul
    colored_header(
    label="",
    description="",
    color_name="blue-40"
    )

    # #footer
    with st.container():

        # cria 3 colunas
        col1, col2, col3 = st.columns([2,1,2])
        
        # coluna do meio
        col2.write("Developed By: [@Huelerssey](https://huelerssey-portfolio.website)")
