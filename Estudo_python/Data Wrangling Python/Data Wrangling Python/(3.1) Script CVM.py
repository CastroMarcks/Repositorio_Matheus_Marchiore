# -*- coding: utf-8 -*-

# Data Wrangling
# MBA em Data Science e Analytics USP/ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Atividade nº 3 - Dataset Comissão de Valores Mobiliários (CVM)

# O dataset contém informações financeiras de companhias abertas brasileiras
# Fonte: https://dados.cvm.gov.br/dataset/cia_aberta-doc-dfp

# O objetivo é analisar a variação anual nas vendas e no lucro das empresas

#%% Carregando os pacotes

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% Importando os bancos de dados

# Os arquivos têm um encoding específico
# É possível ajustar já na importação do .csv

dados_cvm = pd.read_csv('(3.2) CVM Resultado.csv', 
                        sep=';',
                        encoding='latin1')

dados_cadastro = pd.read_csv('(3.3) CVM Dados Cadastrais.csv', 
                        sep=';',
                        encoding='latin1')

#%% Registros únicos das contas

contas = dados_cvm['DS_CONTA'].unique()

#%% Filtrar as observações de interesse

# Em análise detalhada, identificou-se pelo código das contas (CD_CONTA)
# A receita principal da empresa é 3.01 e lucro/prejuízo líquido é 3.11
dados_sel = dados_cvm.query('CD_CONTA == "3.01" | CD_CONTA == "3.11"')

#%% Vamos ajustar a variável de data (que está como texto)

data = pd.to_datetime(dados_sel['DT_FIM_EXERC']).dt.year
dados_sel.insert(5, 'ANO', data)

dados_sel.info()

#%% Organizar as observações

# Temos informações de 2021 e 2022, vamos colocá-las juntas para cada empresa
# Para melhor organização, vamos separar as contas de receitas e lucros
dados_sel = dados_sel.sort_values(by=['CD_CONTA', 'CD_CVM'], ascending=True)

#%% Análise de duplicidades de observações

contagem = dados_sel.groupby(['CD_CVM', 'CD_CONTA'])['VL_CONTA'].count()

# Há um resíduo no dataset, a empresa CD_CVM = 26077 tem duplicidades

#%% Exclusão do resíduo

# Em análise adicional, verificou-se que há "versões" de relatórios
# Vamos manter a última versão disponibilizada (VERSAO = 3)
dados_sel.query('~(CD_CVM == 26077 & VERSAO == 1)', inplace=True)

# A inversão do critério de filtro foi feita por meio do ~

#%% Vamos adicionar os setores das empresas para fazer análises específicas

# Ajustando a base dos dados cadastrais
cadastrais = dados_cadastro[['CD_CVM', 'SETOR_ATIV']].copy()
cadastrais = cadastrais[cadastrais['SETOR_ATIV'].notnull()] # elimina missings

# Vamos manter apenas registros únicos (evitando duplicidade no merge)
cadastrais.drop_duplicates(inplace=True)

# Realizando o merge
dados_sel = pd.merge(dados_sel, cadastrais, 
                     how="left", 
                     on="CD_CVM")

#%% Limpeza: vamos selecionar apenas as variáveis de interesse

dados_sel = dados_sel[['CD_CVM',
                       'DENOM_CIA', 
                       'SETOR_ATIV', 
                       'CD_CONTA', 
                       'ANO', 
                       'VL_CONTA']]

# Note que já reposicionamos na ordem desejada

#%% Para facilitar a leitura das informações, vamos substituir os labels

labels = {'3.01': 'Receita',
          '3.11': 'Resultado'}

dados_sel = dados_sel.assign(CD_CONTA = dados_sel.CD_CONTA.map(labels))

#%% Vamos calcular a variação percentual (variável de interesse no estudo)

# Criar uma variável com o valor defasado
dados_sel['VALOR_LAG'] = dados_sel.groupby(['CD_CVM', 'CD_CONTA'])['VL_CONTA'].shift(1)

# Criando uma variável com o resultado da variação
dados_sel['VARIACAO'] = ((dados_sel['VL_CONTA'] - dados_sel['VALOR_LAG']) / dados_sel['VALOR_LAG'])

# Vamos ajustar o arredondamento para melhor visualização
dados_sel['VARIACAO'] = round(dados_sel['VARIACAO'], 3)

#%% Estatísticas descritivas da VARIAÇÃO

# Existem valores muito extremos que influenciam as descritivas

# Vamos limpar as observações
dados_sel = dados_sel[~ dados_sel['VARIACAO'].isin([np.nan, np.inf, -np.inf])]

# Tabela de estatísticas descritivas da VARIAÇÃO
dados_sel['VARIACAO'].describe()

#%% Vamos excluir grandes variações

# Existem muitos valores extremos na distribuição da variável

# Exemplo: vamos excluir variações maiores do que 200% e menores do que -200%
# São indícios de variações significativas nos fundamentos da empresa

dados_sel = dados_sel[dados_sel['VARIACAO'].between(-2, 2, inclusive='both')]

#%% Novas estatísticas descritivas

dados_sel['VARIACAO'].describe()

#%% Informações mais detalhadas por tipo de conta (receita e resultado)

dados_sel.groupby(['CD_CONTA'])['VARIACAO'].describe().T

#%% Informações mais detalhadas por setor

# Por setor
desc_setor = dados_sel.groupby(['CD_CONTA','SETOR_ATIV']).agg({'VARIACAO':'mean'})
desc_setor = desc_setor.reset_index()

# Os números setoriais indicam que existem análises mais específicas a fazer
# Por exemplo, alguns setores podem ter poucas observações (média com viés)

# Contagem de informações por setor
# Critério escolhido: vamos manter apenas setores com no mínimo 6 observações
n_setor = dados_sel[['SETOR_ATIV', 'VARIACAO']].groupby('SETOR_ATIV').count()

n_setor_seleciona = (n_setor
                     .query('VARIACAO >= 6')
                     .rename(columns={'VARIACAO':'CONTAGEM'})).reset_index()

# Ajuste do banco de dados
desc_setor = (desc_setor
.merge(n_setor_seleciona, how = 'left', on = 'SETOR_ATIV')
.query('~CONTAGEM.isna()'))

# Visualizando graficamente
plt.figure(figsize=(18,12), dpi = 600)
sns.barplot(data=desc_setor, y='SETOR_ATIV', x='VARIACAO', hue='CD_CONTA')

#%% FIM!