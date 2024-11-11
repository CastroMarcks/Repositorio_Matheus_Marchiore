# -*- coding: utf-8 -*-

# Data Wrangling
# MBA em Data Science e Analytics USP/ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Atividade nº 2 - Dataset WDI World Bank

# O dataset contém muitos indicadores sobre o desenvolvimento dos países
# Fonte: https://databank.worldbank.org/source/world-development-indicators

# O objetivo é analisar variáveis referentes à área da saúde

#%% Carregando os pacotes

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Importando os bancos de dados

# Analise o arquivo (2.2) WDI World Bank.xlsx e veja os missing values
# Será utilizado um argumento de ajuste de NAs já na importação (dados_wdi)

dados_wdi = pd.read_excel('(2.2) WDI World Bank.xlsx', na_values='..')
dados_grupo = pd.read_excel('(2.3) WDI Income Group.xlsx')
dados_paises = pd.read_excel('(2.4) WDI Country.xlsx')

#%% Informações básicas dos dados

dados_wdi.info()

#%% Elementos únicos das variáveis

dados_wdi['Country Name'].unique()
dados_wdi['Series Name'].unique()
dados_wdi['Topic'].unique()

#%% Alterando os nomes das variáveis

dados_wdi.rename(columns={'Country Name':'pais',
                          'Country Code':'cod_pais',
                          'Series Name': 'serie',
                          'Series Code': 'cod_serie',
                          '2021 [YR2021]': 'ano_2021',
                          'Topic': 'topico'}, inplace=True)

#%% Analisando as últimas linhas do dataset

dados_wdi['pais'].tail(n=20)

# As últimas linhas do banco de dados não são observações
# São as referências do banco de dados e não serão utilizadas

#%% Excluindo as linhas finais

# Neste caso, vamos selecionar aquelas que serão analisadas
# Foi adicionada uma posição a mais na sequência
dados_wdi = dados_wdi.iloc[0:383572,]

# Revisando os dados
dados_wdi['pais'].tail(n=20)

#%% Selecionando os tópicos de saúde

# Os elementos do mesmo tópico iniciam com seu agregador
# Vamos selecionar as variáveis com base neste critério
dados_saude = dados_wdi[dados_wdi['topico'].str.startswith('Health')]

#%% Colocando as séries nas colunas

# O banco de dados não apresenta a estrutura com variáveis em colunas
# Será ajustado para manter a estrutura mais comumente utilizada

# As séries se tornam variáveis e as observações são os países
dados_saude = pd.pivot(dados_saude, 
                       index=['pais','cod_pais'], 
                       columns=['serie'], 
                       values='ano_2021')

# Voltando para o índice numérico
dados_saude.reset_index(inplace=True)

#%% Limpeza de observações

# Muitas observações não são de países; são agrupamentos disponíveis na base
# Para não gerar viés nas análises, vamos remover tais observações

# O arquivo (2.4) WDI Country.xlsx contém a referência dos IDs dos países
# Trocando o nome da variável chave para a ligação entre DataFrames
dados_paises.rename(columns={'Country':'cod_pais'}, 
                    inplace=True)

# Realizando um merge
dados_saude = pd.merge(dados_saude, dados_paises, 
                       how='left', 
                       on='cod_pais')
# A variável foi para o final do dataset
# Os valores identificados como 'nan' não são países

# Vamos limpar por meio de um filtro de observações
dados_saude = dados_saude[~ dados_saude['Name'].isna()].reset_index(drop=True)

# Vamos remover a coluna que não será mais utilizada
dados_saude.drop(columns=['Name'], inplace=True)

#%% Limpeza de variáveis

# Muitas variáveis apresentam apenas NAs
# Vamos manter apenas aquelas que têm alguma informação disponível
dados_saude.dropna(axis=1, how='all', inplace=True)

# axis=1 -> refere-se às colunas
# how='all' -> drop se todos os elementos forem nan

#%% Adicionar a categoria "income group" ao dataset final

# O objetivo é adicionar uma variável de agrupamento que está em 'dados_grupo'

# Selecionando as variáveis de interesse
dados_grupo_select = dados_grupo[['Code', 'Income Group']].copy()

# Trocando o nome da chave para a ligação entre DataFrames
dados_grupo_select.rename(columns={'Code':'cod_pais'}, 
                          inplace=True)

# Realizando o merge
dados_saude = pd.merge(dados_saude, dados_grupo_select, 
                       how='left', 
                       on='cod_pais')

#%% Reorganizando a posição da coluna

# Removendo a variável de interesse
organizar = dados_saude.pop('Income Group')

# Inserindo na posição desejada
dados_saude.insert(2, 'Group', organizar)

# inserir na posição 2
# o nome será Group
# a variável que será inserida

#%% Por exemplo, supondo que trata-se de um estudo sobre diabetes 

# Obter as posições das variáveis
col_pos = dados_saude.columns

# A variável de interesse está na posição 23

# Estatísticas descritivas
dados_saude.iloc[:,23].describe()

# Estatísticas por grupos
estat_grupo = dados_saude.iloc[:,[2,23]].groupby('Group').mean().reset_index()

# Finalizando com o gráfico
plt.figure(figsize=(15,9), dpi = 600)
ax = sns.barplot(data=estat_grupo, x=estat_grupo.iloc[:,0], y=estat_grupo.iloc[:,1])
for container in ax.containers: ax.bar_label(container, fmt='%.2f', padding=3, fontsize=12)
plt.xlabel('Grupo',fontsize=15)
plt.ylabel('Diabetes prevalence (% of population ages 20 to 79)', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#%% FIM!