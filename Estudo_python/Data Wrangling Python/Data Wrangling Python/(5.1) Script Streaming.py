# -*- coding: utf-8 -*-

# Data Wrangling
# MBA em Data Science e Analytics USP/ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Atividade nº 5 - Datasets de filmes e séries

# Os datasets contêm avaliações de filmes e séries disponíveis em streamings
# Fonte: https://www.kaggle.com/datasets/ruchi798/tv-shows-on-netflix-prime-video-hulu-and-disney

#%% Carregando os pacotes

import pandas as pd

#%% Carregando os datasets

dados_filmes = pd.read_csv('(5.2) Filmes Streaming.csv', sep=',')
dados_series = pd.read_csv('(5.3) Séries Streaming.csv', sep=',')

dados_filmes.info()
dados_series.info()

#%% Selecionado colunas e realizando a junção dos bancos de dados

# Os dois datasets têm estruturas semelhantes quanto às variáveis
# Porém, o dataset sobre filmes tem colunas a mais
# Vamos fazer uma rápida organização dos datasets e juntá-los

# Seleção das variáveis de interesse
dados_filmes = dados_filmes.iloc[:,0:12]

# Junção dos dados
dados_completo = pd.concat([dados_filmes, dados_series], ignore_index=True)

# Remoção de variável
dados_completo.drop(columns=['Unnamed: 0'], inplace=True)

#%% Extrair obter as notas

# Extrair as notas
dados_completo['Ajuste_IMDB'] = dados_completo['IMDb'].str.slice(0, 4)
dados_completo['Ajuste_Rotten'] = dados_completo['Rotten Tomatoes'].str.slice(0, 3)

# Ajustar a string e transformar para float
dados_completo['Ajuste_IMDB'] = dados_completo['Ajuste_IMDB'].str.rstrip('/').astype('float')
dados_completo['Ajuste_Rotten'] = dados_completo['Ajuste_Rotten'].str.rstrip('/').astype('float')

# A função str.rstrip() remove o caractere ao final da string

#%% Gerando estatísticas sobre as notas

# Atribuindo labels
muda_numeros = {0: 'filme', 1: 'série'}
dados_completo = dados_completo.assign(tipo = dados_completo.Type.map(muda_numeros))

# Agrupando o dataset
descritivas = dados_completo.groupby(['tipo'])

# Gerando estatísticas por variável
descritivas['Ajuste_IMDB'].describe().T
descritivas['Ajuste_Rotten'].describe().T

#%% Criando um indicador dos "melhores" filmes e séries 

# Separando o banco de dados
melhores_series = dados_completo[dados_completo['tipo']=='série']
melhores_filmes = dados_completo[dados_completo['tipo']=='filme']

#%% Séries

# Vamos identificar aqueles que estão com melhores notas nas duas avaliações
# Vamos usar o percentil 95 das notas como referência
melhores_series[['Ajuste_IMDB', 'Ajuste_Rotten']].quantile(0.95)

# Gerando os dados

melhores_series = melhores_series.assign(Categ_IMDB = pd.qcut(melhores_series.Ajuste_IMDB,
                                                              q=[0, 0.95, 1.0],
                                                              labels=['menores',
                                                                      'maiores']))

melhores_series = melhores_series.assign(Categ_Rotten = pd.qcut(melhores_series.Ajuste_Rotten,
                                                                q=[0, 0.95, 1.0],
                                                                labels=['menores',
                                                                        'maiores']))

melhores_series = melhores_series[(melhores_series['Categ_IMDB']=='maiores') & 
                                  (melhores_series['Categ_Rotten']=='maiores')].sort_values(['Ajuste_Rotten', 'Ajuste_IMDB'], ascending=False)

#%% Filmes

# Vamos identificar aqueles que estão com melhores notas nas duas avaliações
# Vamos usar o percentil 95 das notas como referência
melhores_filmes[['Ajuste_IMDB', 'Ajuste_Rotten']].quantile(0.95)

# Gerando os dados

melhores_filmes = melhores_filmes.assign(Categ_IMDB = pd.qcut(melhores_filmes.Ajuste_IMDB,
                                                              q=[0, 0.95, 1.0],
                                                              labels=['menores',
                                                                      'maiores']))

melhores_filmes = melhores_filmes.assign(Categ_Rotten = pd.qcut(melhores_filmes.Ajuste_Rotten,
                                                                q=[0, 0.95, 1.0],
                                                                labels=['menores',
                                                                        'maiores']))

melhores_filmes = melhores_filmes[(melhores_filmes['Categ_IMDB']=='maiores') & 
                                  (melhores_filmes['Categ_Rotten']=='maiores')].sort_values(['Ajuste_IMDB', 'Ajuste_Rotten'], ascending=False)

#%% FIM!