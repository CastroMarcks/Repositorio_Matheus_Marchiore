# -*- coding: utf-8 -*-

# Análise Fatorial PCA
# MBA em Data Science e Analytics USP ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Instalando os pacotes

# !pip install pandas
# !pip install numpy
# !pip install factor_analyzer
# !pip install sympy
# !pip install scipy
# !pip install matplotlib
# !pip install seaborn
# !pip install plotly
# !pip install pingouin
# !pip install pyshp

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go

#%% Importando o banco de dados

casas = pd.read_excel("preco_casas.xlsx")
# Fonte: adaptado de https://www.kaggle.com/datasets/elakiricoder/jiffs-house-price-prediction-dataset

# É possível extrairmos fatores que capturem o preço de venda das casas?

#%% Estatísticas descritiva das variáveis

tab_desc = casas.describe()

#%% Analisando as correlações de Pearson

# Matriz de correlações

corr = casas.corr()

# Gráfico interativo

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text=corr.values,
        texttemplate='%{text:.3f}',
        colorscale='viridis'))

fig.update_layout(
    height = 750,
    width = 750,
    yaxis=dict(autorange="reversed"))

fig.show()

#%% Selecionando as variáveis de interesse para a análise

# Vamos deixar o preço das casas de fora da análise fatorial! 

casas_pca = casas.drop(columns=['property_value'])

#%% Teste de Esfericidade de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(casas_pca)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Definindo a PCA (procedimento inicial extraindo todos os fatores possíveis)

fa = FactorAnalyzer(n_factors=8, method='principal', rotation=None).fit(casas_pca)

#%% Obtendo os autovalores

autovalores = fa.get_eigenvalues()[0]

print(autovalores)

# Soma dos autovalores

round(autovalores.sum(), 2)

#%% Critério de Kaiser (raiz latente)

# Temos 3 autovalores maiores do que 1
# Vamos parametrizar a função para a extração de 3 fatores!

fa = FactorAnalyzer(n_factors=3, method='principal', rotation=None).fit(casas_pca)

#%% Eigenvalues, variâncias e variâncias acumuladas

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Gráfico da variância acumulada dos componentes principais

plt.figure(figsize=(12,8))
ax = sns.barplot(x=tabela_eigen.index, y=tabela_eigen['Variância'], data=tabela_eigen, palette='pastel')
ax.bar_label(ax.containers[0])
plt.title("Fatores Extraídos", fontsize=16)
plt.xlabel(f"{tabela_eigen.shape[0]} fatores que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=12)
plt.ylabel("Porcentagem de variância explicada", fontsize=12)
plt.show()

#%% Determinando as cargas fatoriais

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = casas_pca.columns

print(tabela_cargas)

#%% Analisando as cargas fatoriais em cada fator extraído

tabela_cargas_graph = tabela_cargas.reset_index()
tabela_cargas_graph = tabela_cargas_graph.melt(id_vars='index')

sns.barplot(data=tabela_cargas_graph, x='variable', y='value', hue='index', palette='bright')
plt.legend(title='Variáveis', bbox_to_anchor=(1,1), fontsize = '6')
plt.title('Cargas Fatoriais', fontsize='12')
plt.xlabel(xlabel=None)
plt.ylabel(ylabel=None)
plt.show()

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = casas_pca.columns

print(tabela_comunalidades)

#%% Extração dos fatores para as observações do banco de dados

fatores = pd.DataFrame(fa.transform(casas_pca))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Adicionando os fatores ao banco de dados

casas = pd.concat([casas.reset_index(drop=True), fatores], axis=1)

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = casas_pca.columns

print(tabela_scores)

#%% Analisando os scores fatoriais em cada fator extraído

tabela_scores_graph = tabela_scores.reset_index()
tabela_scores_graph = tabela_scores_graph.melt(id_vars='index')

sns.barplot(data=tabela_scores_graph, x='variable', y='value', hue='index', palette='rocket')
plt.legend(title='Variáveis', bbox_to_anchor=(1,1), fontsize = '6')
plt.title('Scores Fatoriais', fontsize='12')
plt.xlabel(xlabel=None)
plt.ylabel(ylabel=None)
plt.show()

#%% Vamos consolidar os 3 fatores em uma medida única (soma ponderada)

casas['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']

    casas['Ranking'] = casas['Ranking'] + casas[tabela_eigen.index[index]]*variancia

#%% Os preços alinham-se às características representadas nos fatores?

pg.rcorr(casas[['Ranking', 'property_value']], 
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Fim!