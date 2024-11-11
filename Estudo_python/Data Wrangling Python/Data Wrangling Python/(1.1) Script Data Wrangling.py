# -*- coding: utf-8 -*-

# Data Wrangling
# MBA em Data Science e Analytics USP/ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Atividade nº 1

# Aplicar funções que são frequentemente utilizadas na manipulação de dados

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install xlrd

#%% Importando os pacotes

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% Importando os bancos de dados

dados_tempo = pd.read_excel('(1.2) dataset_principal.xls')
dados_merge = pd.read_excel('(1.3) dataset_join.xls')
# "dados_tempo" - Fonte: Fávero & Belfiore (2024, Cap. 12)

#%% Visualizando informações básicas do dataset

## Algumas formas para visualizar informações do dataset

# Configurar para printar objetos no console

pd.set_option("display.max.columns", None)

print(dados_tempo)

# Somente os nomes das variáveis

dados_tempo.columns

# Somente as primeiras "n" observações + nomes das variáveis

dados_tempo.head(n=5)

# Somente as últimas "n" observações + nome das variáveis

dados_tempo.tail(n=3)

# Informações detalhadas sobre as variáveis

dados_tempo.info()

# object = variável de texto
# int ou float = variável numérica (métrica)
# category = variável categórica (qualitativa)

#%% Alterando os nomes das variáveis

# No dataset de exemplo, os nomes das variáveis contêm:
# Espaços, maiúsculas, acentos e caracteres especiais...
# É melhor não utilizá-los, pois podem gerar conflito e dificultam a escrita

# Função "rename": utilizada para alterar o nome das variáveis

# Renomeando todas as variáveis pelos nomes

dados_tempo = dados_tempo.rename(columns={'Estudante':'estudante',
                                          'Tempo para chegar à escola (minutos)':'tempo',
                                          'Distância percorrida até a escola (quilômetros)': 'distancia',
                                          'Quantidade de semáforos': 'semaforos',
                                          'Período do dia': 'periodo',
                                          'Perfil ao volante': 'perfil'})

# A seguir, vamos criar um objeto com nomes trocados
# Porém, a referência da variável está pela posição da coluna
# Em datasets com muitas variáveis, esta função facilita a escrita do código
# Lembrando: no Python as contagens de colunas e linhas iniciam-se em zero!

# Renomeando as variáveis pela sua posição (criando um objeto)
# Não é necessário trocar todos os nomes, pode ser um ou alguns deles

dados_novo = dados_tempo.rename(columns={dados_tempo.columns[0]: 'obs',
                                         dados_tempo.columns[1]: 'temp',
                                         dados_tempo.columns[5]: 'perf'})

# Para reescrever o mesmo objeto, poderia utilizar argumento inplace=True

dados_novo.rename(columns={'distancia': 'dist',
                           'semaforos': 'sem',
                           'periodo': 'per'},
                  inplace=True)

#%% Selecionando observações e variáveis de interesse

# Selecionando com base nas posições (1º arg.: linhas, 2º arg.: colunas)
# ATENÇÃO: no Python as contagens de colunas e linhas iniciam-se em zero!

dados_tempo.iloc[3,]
dados_tempo.iloc[:,4] # argumento : indicam vazio na linha
dados_tempo.iloc[2:5,] # note que exclui a posição final
dados_tempo.iloc[:,3:5] # note que exclui a posição final
dados_tempo.iloc[2:4,3:5] # note que exclui as posições finais
dados_tempo.iloc[5,4]

# Detalhar uma variável em específico pelo nome

dados_tempo['tempo']
var_tempo = dados_tempo['tempo']

dados_tempo.perfil
var_perfil = dados_tempo.perfil

# Se for mais de uma variável, inserir o argumento como uma lista

dados_tempo[['tempo', 'perfil']]
var_tempo_perfil = dados_tempo[['tempo', 'perfil']]

# Selecionando variáveis por meio de um início em comum

selec_1 = dados_tempo.loc[:, dados_tempo.columns.str.startswith('per')]

# Selecionando variáveis por meio de um final em comum

selec_2 = dados_tempo.loc[:, dados_tempo.columns.str.endswith('o')]

#%% Criação e alteração de variáveis e seus conteúdos

## 1. Vamos adicionar uma variável a um dataset existente
# Aqui as observações do dataset e variável devem estar igualmente ordenadas!

idade = pd.Series([25,28,30,19,20,36,33,48,19,21])
dados_novo['idade'] = idade

## 2. Adicionando linhas ao banco de dados
# A união ocorrerá pelo nome da coluna, mesmo estando em ordem distinta

nova_obs = pd.DataFrame({'per': ['Tarde'],
                         'obs': ['Roberto'],
                         'temp': [40]})

dados_concat = pd.concat([dados_novo, nova_obs])
dados_concat = pd.concat([dados_novo, nova_obs]).reset_index(drop=True)

# Foram gerados nan (valores faltantes - missing values)

## 3. Vamos criar uma variável em função de outras duas
# O valor será arredondado na mesma linha de código

dados_novo['sem_km'] = round((dados_novo['sem'] / dados_novo ['dist']), 2)

## 4. Vamos trocar os rótulos de determinadas variáveis
# Vamos usar a função 'assign' que adiciona variáveis ao dataset
# A função 'map' tem o objetivo de aplicar uma função a todos elementos da lista

# 4.1 Trocando textos por textos
labels = {'calmo': 'perfil_A',
          'moderado': 'perfil_B',
          'agressivo': 'perfil_C'}

df_labels = dados_tempo.assign(novo_perfil = dados_tempo.perfil.map(labels))
df_labels.info()

# 4.2 Trocando textos por números (ATENÇÃO: Não fazer ponderação arbitrária!)
numeros = {'calmo': 1,
           'moderado': 2,
           'agressivo': 3}

df_numeros = dados_tempo.assign(novo_perfil = dados_tempo.perfil.map(numeros))
df_numeros.info()

# 4.3 Trocando números por textos
textos = {0: 'zero',
          1: 'um',
          2: 'dois',
          3: 'três'}

df_texto = dados_tempo.assign(novo_semaforos = dados_tempo.semaforos.map(textos))
df_texto.info()

## 5. Vamos categorizar aplicando critérios detalhados por meio de condições

dados_tempo['faixa'] = np.where(dados_tempo['tempo']<=20, 'rápido',
                       np.where((dados_tempo['tempo']>20) & (dados_tempo['tempo']<=40), 'médio',
                       np.where(dados_tempo['tempo']>40, 'demorado',
                                'demais')))

## 6. Outra forma de categorizar é por meio dos quartis de variáveis (q=4)

dados_tempo['quartis'] = pd.qcut(dados_tempo['tempo'], q=4, labels=['1','2','3','4'])

## 7. Em certas circunstâncias será necessário trocar o tipo da variável
# Para evitar a ponderação arbitrária no df_numeros, vamos alterar o tipo

df_numeros['novo_perfil'] = df_numeros['novo_perfil'].astype('category')
df_numeros.info()

## 8. Por fim, vamos excluir algumas colunas sem uso
# Por exemplo, em df_numeros não vamos usar 'periodo' e 'perfil'

df_numeros.drop(columns=['periodo', 'perfil'], inplace=True)

#%% Organizando as observações do dataset por meio de critérios

# Organizando em ordem crescente

df_org_1 = dados_tempo.sort_values(by=['tempo'], ascending=True).reset_index(drop=True)

# Organizando em ordem decrescente

df_org_2 = dados_tempo.sort_values(by=['tempo'], ascending=False).reset_index(drop=True)

# Também é possível organizar variáveis texto

df_org_3 = dados_tempo.sort_values(by=['estudante'], ascending=True).reset_index(drop=True)
df_org_4 = dados_tempo.sort_values(by=['perfil'], ascending=False).reset_index(drop=True)

# Organizando por mais de um critério

df_org_5 = dados_tempo.sort_values(by=['perfil', 'distancia'], 
                                   ascending=[False, True]).reset_index(drop=True)

#%% Resumindo os dados

## 1. Visualizando estatísticas descritivas

# Tabela de descritivas para variáveis quantitativas

dados_tempo.describe()

# Estatísticas individuais

dados_tempo['tempo'].count() # contagem
dados_tempo['tempo'].mean() # média
dados_tempo['tempo'].median() # mediana
dados_tempo['tempo'].min() # mínimo
dados_tempo['tempo'].max() # máximo
dados_tempo['tempo'].std() # desvio padrão
dados_tempo['tempo'].var() # variância
dados_tempo['tempo'].quantile([0.25, 0.75]) # quartis
dados_tempo['tempo'].sum() # soma

# Matriz de correlações de Pearson

dados_tempo[['tempo', 'distancia', 'semaforos']].corr()

# Tabela de frequências para variáveis qualitativas

dados_tempo['periodo'].value_counts() # frequências absolutas
dados_tempo['perfil'].value_counts(normalize=True) # frequências relativas

# Tabela de frequências cruzadas para pares de variáveis qualitativas

pd.crosstab(dados_tempo['periodo'], dados_tempo['perfil'])
pd.crosstab(dados_tempo['periodo'], dados_tempo['perfil'], normalize=True)

## 2. Obtendo informações de valores únicos das variáveis

dados_tempo['tempo'].unique()
dados_tempo['periodo'].unique()
dados_tempo['perfil'].nunique() # quantidade de valores únicos

## 3. Criando um banco de dados agrupado (um critério)

dados_periodo = dados_tempo.groupby(['periodo'])

# Gerando estatísticas descritivas

dados_periodo.describe()

# Caso a tabela gerada esteja com visualização ruim no print, pode transpor

dados_periodo.describe().T

# Tamanho de cada grupo

dados_periodo.size()

# Criando um banco de dados agrupado (mais de um critério)

dados_criterios = dados_tempo.groupby(['periodo', 'perfil'])

# Gerando as estatísticas descritivas

dados_criterios.describe().T

# Tamanho de cada grupo

dados_criterios.size()

# Especificando estatísticas de interesse

dados_periodo.agg({'tempo': 'mean',
                   'distancia': 'mean',
                   'periodo': 'count'})

#%% Filtros de observações

# Vamos conhecer a função query para realizar os filtros

# Variáveis textuais e categóricas

filtro_calmo = dados_tempo[dados_tempo['perfil'] == 'calmo']
filtro_quartil = dados_tempo[dados_tempo['quartis'] == '1']

dados_tempo.query('perfil == "calmo"')
dados_tempo.query('quartis == "1"')

# Interseção entre critérios (&)

filtro_intersecao = dados_tempo[(dados_tempo['perfil'] == 'calmo') & (dados_tempo['periodo'] == 'Tarde')]

dados_tempo.query('perfil == "calmo" & periodo == "Tarde"')

# União entre critérios (|)

filtro_uniao = dados_tempo[(dados_tempo['perfil'] == 'calmo') | (dados_tempo['periodo'] == 'Tarde')]

dados_tempo.query('perfil == "calmo" | periodo == "Tarde"')

# Critério de diferente (!=)

filtro_difer = dados_tempo[(dados_tempo['perfil'] != 'calmo')]

dados_tempo.query('perfil != "calmo"')

# Utilizando operadores em variáveis métricas

filtro_tempo_1 = dados_tempo[dados_tempo['tempo'] >= 25]

filtro_tempo_2 = dados_tempo[(dados_tempo['tempo'] > 30) & (dados_tempo['distancia'] <= 25)]

filtro_tempo_3 = dados_tempo[dados_tempo['tempo'].between(25, 40, inclusive='both')]
# inclusive: "both", "neither", "left" ou "right"

dados_tempo.query('tempo >= 25')
dados_tempo.query('tempo > 30 & distancia <= 25')
dados_tempo.query('tempo.between(25, 40, inclusive="both")')

# Comparando com valores de outro objeto (isin())

nomes = pd.Series(["Gabriela", "Gustavo", "Leonor", "Ana", "Júlia"])
filtro_contidos = dados_tempo[dados_tempo['estudante'].isin(nomes)]

dados_tempo.query('estudante.isin(@nomes)') # note o @ referenciando o objeto

# Usando o critério "não" (inverte o argumento)

filtro_tempo_4 = dados_tempo[~(dados_tempo['tempo'] >= 25)]

filtro_perfil_demais = dados_tempo[~(dados_tempo['perfil'] == 'moderado')]

filtro_nao_contidos = dados_tempo[~(dados_tempo['estudante'].isin(nomes))]

dados_tempo.query('~(tempo >= 25)')
dados_tempo.query('~(perfil == "moderado")')
dados_tempo.query('~(estudante.isin(@nomes))')

#%% Junção de bancos de dados (merge)

# É necessária uma "chave" que faça a ligação entre os dois bancos de dados
# Ou seja, é necessário pelo menos uma variável em comum nos datasets

# Inicialmente, deixar as colunas "chave" com o mesmo nome nos dois datasets

dados_merge.rename(columns={'Estudante':'estudante'}, inplace=True)

# Parâmetros de configuração na função merge:
    # how: é a direção do merge (quais IDs restam na base final)
    # on: é a coluna com a chave para o merge

# Note que existe a seguinte diferença em termos de observações:
    # dados_tempo: contém Antônio, mas não o Marcos
    # dados_merge: contém Marcos, mas não Antônio

# Left
# Observações de dados_merge -> dados_tempo
# Ficam os IDs de dados_tempo

merge_1 = pd.merge(dados_tempo, dados_merge, how='left', on='estudante')

# Right
# Observações de dados_tempo -> dados_merge
# Ficam os IDs de dados_merge

merge_2 = pd.merge(dados_tempo, dados_merge, how='right', on='estudante')

# Outer
# Observações das duas bases de dados constam na base final 
# Ficam todos os IDs presentes nas duas bases

merge_3 = pd.merge(dados_tempo, dados_merge, how='outer', on='estudante')

# Inner
# Somente os IDs que constam nas duas bases ficam na base final 
# É a interseção de IDs entre as duas bases de dados

merge_4 = pd.merge(dados_tempo, dados_merge, how='inner', on='estudante')

# Verificando apenas a diferença entre os bancos de dados (comparação)

merge_5 = dados_tempo[~ dados_tempo.estudante.isin(dados_merge.estudante)]
merge_6 = dados_merge[~ dados_merge.estudante.isin(dados_tempo.estudante)]

# É importante analisar se há duplicidades de observações antes do merge

#%% Analisando duplicidades de observações

# Gerando o objeto após a remoção

dados_tempo.drop_duplicates()
# Interpretação: como retornou o mesmo DataFrame, não há duplicidades

# Contagem de linhas duplicadas

len(dados_tempo) - len(dados_tempo.drop_duplicates())

# Se fosse para estabelecer uma remoção com base em algumas variáveis

dados_tempo.drop_duplicates(subset=['estudante', 'perfil'])

#%% Excluindo valores faltantes (NA)

# Apresentando a contagem de NAs em cada variável

merge_3.isna().sum()

# Caso queira substituir NAs por algum elemento

merge_3 = merge_3.assign(quartis = merge_3.quartis.astype('object'))

# Texto

merge_3.fillna('elemento')

# Valor métrico
# ATENÇÃO: NÃO É UMA RECOMENDAÇÃO, APENAS ILUSTRA A DISPONIBILIDADE DO CÓDIGO!

merge_3['tempo'].fillna(merge_3['tempo'].mean())

# Excluindo observações que apresentem valores faltantes

merge_exclui = merge_3.dropna().reset_index(drop=True)

#%% Alterando a estrutura do banco de dados

# Colocando uma coluna abaixo de outra

df_estrutura = pd.melt(dados_tempo,
                       id_vars='estudante',
                       value_vars=['tempo', 'distancia'])

# Gerando um gráfico no novo DataFrame

plt.figure(figsize=(15,9), dpi = 600)
sns.barplot(data=df_estrutura, y='estudante', x='value', hue='variable')

#%% Encadeando funções

# Em certas circunstâncias é possível encadear uma função na outra
# Facilita a escrita e a leitura do código

(dados_tempo
.assign(sem_km = round((dados_novo['sem']/dados_novo ['dist']), 2))
.query('tempo >= 30')
.rename(columns={'periodo':'per'})
.groupby('per')
.agg({'sem_km':'mean',
      'distancia': 'mean',
      'per': 'count'}))

# Poderia criar um objeto normalmente!

df_ajustes = (dados_tempo
.assign(sem_km = round((dados_novo['sem']/dados_novo ['dist']), 2))
.query('tempo >= 30')
.rename(columns={'periodo':'per'})).sort_values('estudante').reset_index(drop=True)

#%% FIM!