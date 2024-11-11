import pickle

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import app as st
import importlib.util
import base64

# Função para converter a imagem para base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Projeto - Análise de Renda",
     page_icon="📈",
     layout="wide",
)

st.markdown("<h1 style='text-align: left;'>Análise de Renda</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left;'>Streamlit com as análises gráficas e implementação dos modelos</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left;'>Modelos treinados:</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left;'>   -TPOT</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left;'>   -Random Forest</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left;'>   -Regressão linear</h3>", unsafe_allow_html=True)
# criando colunas
col1, col2 = st.columns([2, 3])
# --------------------------------- Sidebar ----------------------------------
with col1:
    st.write('--'*50)

    st.title('Analise bivariada das variáveis')

    st.markdown('### Visualização da relação entre as variáveis, cada ponto representa uma pessoa')
    st.markdown('### O gráfico da diagonal principal mostra a distribuição de cada variável')
    st.markdown('### Os gráficos fora da diagonal principal mostram a relação entre duas variáveis')
    st.markdown('### É útil para identificar relações entre variáveis')

    st.image(plt.imread('./output/pairplot.png'))


# --------------------------------- Sidebar ----------------------------------
with col2:
    st.write('--'*50)

    st.title('Barplots das variáveis')

    st.markdown('### Visualização da distribuição das variáveis categóricas\n'
                '### cada barra representa a média da renda para cada categoria com intervalo de confiança de 95%')

    st.image(plt.imread('./output/barplots.png'))



# --------------------------------- Sidebar ----------------------------------
st.write('--'*50)
col1, col2 = st.columns(2)



with col1:
    # load heatmap fig
    st.title('Heatmap de Pearson')
    heatmap_pearson = plt.imread('./output/heatmap_pearson.png')
    st.image(heatmap_pearson)

with col2:
    st.title('Heatmap de Spearman')
    # load heatmap fig
    heatmap_spearman = plt.imread('./output/heatmap_spearman.png')
    st.image(heatmap_spearman)


# --------------------------------- Sidebar ----------------------------------

st.write('--'*50)
# Exemplo de interface de entrada de valores
st.title("Predição com Modelos Treinados")

# Carregando os modelos treinados

# Carregar o modelo salvo
lr_model = joblib.load('./output/linear_regression_model.pkl')
best_rf_model = joblib.load('./output/random_search_model.pkl')


# Carregar o pipeline exportado e ajustado
path_to_pipeline = './output/tpot_pipeline.py'
spec = importlib.util.spec_from_file_location("tpot_pipeline", path_to_pipeline)
tpot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpot_module)

# Carregar o modelo treinado
model_path = './output/trained_pipeline.pkl'
tpot_module.load_model(model_path)



# Criar duas colunas
col1, col2 = st.columns(2)

# Inputs para os valores das features
with col1:
    posse_de_veiculo = st.checkbox('Posse de Veículo')

    qtd_filhos = st.number_input('Quantidade de Filhos', min_value=0, step=1)
    idade = st.number_input('Idade', min_value=0, step=1)
    tempo_emprego = st.number_input('Tempo de Emprego (anos)', min_value=0.0, step=0.1)
    qt_pessoas_residencia = st.number_input('Quantidade de Pessoas na Residência', min_value=1, step=1)

    # Seletor para sexo
    sexo_M = st.selectbox('Sexo', ['F', 'M'])
    sexo_M = 1 if sexo_M == 'M' else 0

with col2:
    posse_de_imovel = st.checkbox('Posse de Imóvel')


    # Seletor para tipo de renda
    tipo_renda = st.selectbox('Tipo de Renda', ['Bolsista', 'Empresário', 'Pensionista', 'Servidor público'])
    tipo_renda_Bolsista = 1 if tipo_renda == 'Bolsista' else 0
    tipo_renda_Empresário = 1 if tipo_renda == 'Empresário' else 0
    tipo_renda_Pensionista = 1 if tipo_renda == 'Pensionista' else 0
    tipo_renda_Servidor_publico = 1 if tipo_renda == 'Servidor público' else 0

    # Seletor para educação
    educacao = st.selectbox('Educação', ['Pós graduação', 'Secundário', 'Superior completo', 'Superior incompleto'])
    educacao_Pos_graduacao = 1 if educacao == 'Pós graduação' else 0
    educacao_Secundario = 1 if educacao == 'Secundário' else 0
    educacao_Superior_completo = 1 if educacao == 'Superior completo' else 0
    educacao_Superior_incompleto = 1 if educacao == 'Superior incompleto' else 0

    # Seletor para estado civil
    estado_civil = st.selectbox('Estado Civil', ['Separado', 'Solteiro', 'União', 'Viúvo'])
    estado_civil_Separado = 1 if estado_civil == 'Separado' else 0
    estado_civil_Solteiro = 1 if estado_civil == 'Solteiro' else 0
    estado_civil_Uniao = 1 if estado_civil == 'União' else 0
    estado_civil_Viuvo = 1 if estado_civil == 'Viúvo' else 0

    # Seletor para tipo de residência
    tipo_residencia = st.selectbox('Tipo de Residência', ['Casa', 'Com os pais', 'Comunitário', 'Estúdio', 'Governamental'])
    tipo_residencia_Casa = 1 if tipo_residencia == 'Casa' else 0
    tipo_residencia_Com_os_pais = 1 if tipo_residencia == 'Com os pais' else 0
    tipo_residencia_Comunitario = 1 if tipo_residencia == 'Comunitário' else 0
    tipo_residencia_Estudio = 1 if tipo_residencia == 'Estúdio' else 0
    tipo_residencia_Governamental = 1 if tipo_residencia == 'Governamental' else 0

# Criar um array com os valores inseridos
input_data = np.array([[
    posse_de_veiculo,
    posse_de_imovel,
    qtd_filhos,
    idade,
    tempo_emprego,
    qt_pessoas_residencia,
    sexo_M,
    tipo_renda_Bolsista,
    tipo_renda_Empresário,
    tipo_renda_Pensionista,
    tipo_renda_Servidor_publico,
    educacao_Pos_graduacao,
    educacao_Secundario,
    educacao_Superior_completo,
    educacao_Superior_incompleto,
    estado_civil_Separado,
    estado_civil_Solteiro,
    estado_civil_Uniao,
    estado_civil_Viuvo,
    tipo_residencia_Casa,
    tipo_residencia_Com_os_pais,
    tipo_residencia_Comunitario,
    tipo_residencia_Estudio,
    tipo_residencia_Governamental
]], dtype=np.float64)

# Realizar predições ao clicar no botão
if st.button('Realizar Predição'):
    pred_tpot = tpot_module.predict(input_data)
    pred_lr = lr_model.predict(input_data)
    pred_best_rf = best_rf_model.predict(input_data)

    st.write(f"Predição com TPOT: {pred_tpot[0]}")
    st.write(f"Predição com Regressão Linear: {pred_lr[0]}")
    st.write(f"Predição com Random Forest: {pred_best_rf[0]}")

st.write('--'*50)

col1, col2 = st.columns(2)

with col1:

    st.title('Análise de resíduos')

    # load residual plot fig
    residual_plot = plt.imread('./output/residuos.png')
    st.image(residual_plot)

    st.table(pd.read_csv('./output/test_results.csv'))

with col2:

    st.title('Ganho de R2 por tempo de execução')
    # load tempo_r2 fig
    tempo_r2 = plt.imread('./output/tempo_r2.png')
    st.image(tempo_r2)

st.write('--'*50)

st.title('importância das variáveis')

# CSS for styling
st.markdown(
    """
    <style>
    .image-container {
        text-align: center;
    }
    .image-container img {
        width: 100%; /* Adjust to your desired size */
        height: auto;
    }
    .col {
        padding: 10px;
    }
    .spacer {
        margin-top: 20px; /* Space between lines */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Get base64 encoded images
rf_image_base64 = get_image_base64('./output/importancia_variaveis_rf_RandomSearch.png')
lr_image_base64 = get_image_base64('./output/importancia_variaveis_lr.png')
tpot_image_base64 = get_image_base64('./output/importancia_variaveis_rf_tpot.png')


# Create three columns
col1, col2, col3 = st.columns(3)

# Display content in columns
with col1:
    st.markdown("<div class='image-container'><h3 style='text-align: left;'>Random Forest</h3></div>", unsafe_allow_html=True)
    st.image(plt.imread('./output/importancia_variaveis_rf_RandomSearch.png'))

with col2:
    st.markdown("<div class='image-container'><h3 style='text-align: left;'>Regressão Linear</h3></div>", unsafe_allow_html=True)
    st.image(plt.imread('./output/importancia_variaveis_lr.png'))

with col3:
    st.markdown("<div class='image-container'><h3 style='text-align: left;'>TPOT</h3></div>", unsafe_allow_html=True)
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    st.image(plt.imread('./output/importancia_variaveis_rf_tpot.png'))

# Add space between lines
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# Add a divider line
st.write('--' * 50)

# Add more space after the divider
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)







