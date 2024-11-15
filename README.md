# Previsão de Renda com Deploy em Streamlit

## Descrição do Projeto
Este projeto tem como objetivo prever a renda mensal de clientes com base em suas características pessoais. A previsão de renda é útil em diversas aplicações, como análise de crédito, segmentação de clientes e personalização de ofertas. 

O processo foi desenvolvido com base na metodologia **CRISP-DM**, abrangendo as etapas de entendimento do negócio, análise exploratória, modelagem, validação e implantação, resultando em uma aplicação interativa criada com **Streamlit**.

---

https://github.com/LucianoContri/Analise_de_Renda/assets/64796689/3590d8a6-46ff-4f88-accc-991b5a00365f

---

## Tecnologias e Ferramentas Utilizadas

- **Linguagens e Bibliotecas:**
  - Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TPOT, Joblib.

- **Frameworks para Visualização e Deploy:**
  - Streamlit.

- **Técnicas de Modelagem:**
  - Regressão Linear, Random Forest e TPOT para otimização automatizada de pipelines.

- **Validação Estatística:**
  - Testes de Shapiro-Wilk, Levene e Breusch-Pagan para análise de resíduos.

- **Métodos de Boosting e Random Search:**
  - Random Forest e Gradient Boosting com busca de hiperparâmetros.

- **Machine Learning Automatizado:**
  - TPOT para geração de pipelines otimizados.

- **Análise Exploratória:**
  - Relatórios interativos com YData Profiling, gráficos de correlação, dispersão e barras.

---

## Etapas e Técnicas Aplicadas

### 1. **Entendimento do Negócio**
   - Identificação da necessidade de previsão de renda para otimizar decisões comerciais e financeiras.

### 2. **Análise de Dados**
   - Avaliação univariada e bivariada das variáveis utilizando heatmaps, gráficos de dispersão e barras.
   - Criação de dummies para variáveis categóricas e análise de correlação (Pearson e Spearman).
   - Tratamento de dados ausentes e ajuste de formatos para variáveis categóricas e numéricas.

### 3. **Modelagem**
   - Divisão dos dados em conjuntos de treino e teste.
   - Normalização com **StandardScaler**.
   - Teste de diferentes abordagens:
     - **Regressão Linear:** Modelo base para comparação de desempenho.
     - **Random Forest com Randomized Search:** Identificação de melhores hiperparâmetros.
     - **TPOT:** Otimização automatizada para seleção do pipeline mais eficiente.
   - Avaliação de modelos com métricas como **R² Score**, além de testes de normalidade e homocedasticidade dos resíduos.

### 4. **Validação e Comparação**
   - Comparação de desempenho entre os modelos baseados em tempo de execução e precisão (R²).
   - Visualização interativa dos resíduos para verificar suposições estatísticas.
   - Identificação de variáveis mais importantes nos modelos de Random Forest e Regressão Linear.

### 5. **Implantação com Streamlit**
   - Deploy dos resultados e gráficos interativos utilizando **Streamlit**.
   - Integração do pipeline otimizado treinado pelo TPOT.
   - Criação de um vídeo demonstrativo no **README** explicando o funcionamento da aplicação.

---

## Destaques do Projeto

- **Machine Learning Automatizado (AutoML):**
  - Uso do TPOT para identificar o melhor pipeline, automatizando a escolha do modelo e hiperparâmetros.
  
- **Validação Estatística Rigorosa:**
  - Implementação de testes para avaliar a robustez dos modelos.

- **Interatividade:**
  - Deploy do modelo e gráficos no Streamlit, permitindo análise visual e demonstração prática dos resultados.

- **Melhorias Iterativas:**
  - Comparação de abordagens tradicionais (Regressão Linear) e avançadas (Random Forest e TPOT).

---

