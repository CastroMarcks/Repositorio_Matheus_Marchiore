# -*- coding: utf-8 -*-
"""
===============================================================================
MBA USP ESALQ
Big Data e Deployment de Modelos II

Prof. Helder Prado Santos
===============================================================================
"""

# %%

# Instalação dos pacotes necessário
!pip install pyspark==3.5.1
!pip install findspark
!pip install seaborn
!pip install pandas
!pip install matplotlib

# %%

# Importando as bibliotecas que utilizaremos durante a aula
from pyspark import SparkConf
from pyspark.sql import SparkSession
import findspark
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Encontrando as configurações para inicializar o Spark
findspark.init()

# %%

# Nome da aplicação Spark
app_name = "Aplicação Spark"

# Inicializando o objeto de configurações do Spark
conf = SparkConf()

# Configuração do tipo de cluster
conf.set("spark.master", "local[*]")
# Definição do nome da aplicação
conf.set("spark.app.name", app_name)
# Definição da quantidade de núcleos de processamento a serem usados
conf.set("spark.executor.cores", "2")
# Definição da quantidade de memória alocada para cada executor
conf.set("spark.executor.memory", "2g")
conf.set("spark.driver.memory", "2g")

# Criando a sessão Spark
spark = SparkSession.builder.config(conf=conf).getOrCreate()

print("Spark inicializado com sucesso!")

# %%

###############################################################################
#                              DATA WRANGLING                                 #
###############################################################################

# FONTE: https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cance
# llation-data-2009-2018

# Lendo um DataFrame Spark a partir de um arquivo CSV
df = spark.read.csv("./datasets/2016.csv", header=True, inferSchema=True)

# %%

# Exibindo os dados do DataFrame
df.show()

# %%

# Exibindo uma linha do DataFrame em formato vertical
df.show(1, vertical=True)

# %%

# Exibindo o esquema do DataFrame (nome das colunas e tipos)
df.printSchema()

# FL_DATE                      | DATA_VOO
# OP_CARRIER                   | TRANSPORTADORA
# OP_CARRIER_FL_NUM            | NUM_VOO_TRANSPORTADORA
# ORIGIN                       | ORIGEM
# DEST                         | DESTINO
# CRS_DEP_TIME                 | HORARIO_PARTIDA_PROGRAMADO
# DEP_TIME                     | HORARIO_PARTIDA_REAL
# DEP_DELAY                    | ATRASO_PARTIDA
# TAXI_OUT                     | TAXI_OUT
# WHEELS_OFF                   | HORARIO_DECOLAGEM
# WHEELS_ON                    | HORARIO_POUSO
# TAXI_IN                      | TAXI_IN
# CRS_ARR_TIME                 | HORARIO_CHEGADA_PROGRAMADO
# ARR_TIME                     | HORARIO_CHEGADA_REAL
# ARR_DELAY                    | ATRASO_CHEGADA
# CANCELLED                    | CANCELADO
# CANCELLATION_CODE            | CODIGO_CANCELAMENTO
# DIVERTED                     | DESVIADO
# CRS_ELAPSED_TIME             | TEMPO_DE_VOO_PROGRAMADO
# ACTUAL_ELAPSED_TIME          | TEMPO_DE_VOO_REAL
# AIR_TIME                     | TEMPO_DE_VOO_EM_AR
# DISTANCE                     | DISTANCIA
# CARRIER_DELAY                | ATRASO_DA_TRANSPORTADORA
# WEATHER_DELAY                | ATRASO_DE_CLIMA
# NAS_DELAY                    | ATRASO_NAS
# SECURITY_DELAY               | ATRASO_DE_SEGURANCA
# LATE_AIRCRAFT_DELAY          | ATRASO_DE_AERONAVE_TARDE

# %%

# Criando um DataFrame com colunas selecionadas
df_selecionado = df.select(["DISTANCE", "ACTUAL_ELAPSED_TIME", "ARR_DELAY"])

# Exibindo o DataFrame com as colunas selecionadas
df_selecionado.show()

# %%

# Visualizando as análises univariadas
df_selecionado.describe().show()

# %%

# contagem de linhas do dataset
total_linhas = df.count()

print(f"Total de linhas no dataset: {total_linhas} \n")

# %%

# verificando o número de partições utilizadas no processamento distribuído
df.rdd.getNumPartitions()

# %%

# coletando uma pequena amostra equivalente a 1%
pequena_amostra = df.sample(fraction=0.01)

# %%

# contagem de linhas do dataset reduzido
total_linhas_pequena_amostra = pequena_amostra.count()

print(f"Total de linhas no dataset: {total_linhas_pequena_amostra} \n")

# %%

# estratégia da extrapolação para calcular grandes volumes de dados
total_linhas_extrapoladas = int(total_linhas_pequena_amostra / 0.01)

print(f"Total de linhas no dataset: {total_linhas_extrapoladas} \n")

# %%

# Tratando dados temporais

#  Importando funções úteis do PySpark para tratamento de dados
import pyspark.sql.functions as F

# Documentação: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html

# Extraindo o ano da variável FL_DATE
df = df.withColumn("ano", F.year(df["FL_DATE"]))

# Extraindo o mês da variável FL_DATE
df = df.withColumn("mes", F.month(df["FL_DATE"]))

# Extraindo o dia do mês da variável FL_DATE
df = df.withColumn("dia", F.dayofmonth(df["FL_DATE"]))

# %%

# Exibindo os dados com as novas colunas
df.show(5, vertical=True)

# %%

# Removendo colunas indesejadas com o método drop
df = df.drop("Unnamed: 27")

# %%

# Exibindo os dados após a exclusão da coluna
df.show(5, vertical=True)

# %%

# Selecionando apenas as colunas temporais
df.select("FL_DATE", "ano", "mes", "dia").show()

# %%

# Verificando novamente o esquema dos dados
df.printSchema()

# %%

# Verificando valores nulos no dataset
# Lógica: contagem quando cada linha da coluna é um valor nulo
df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns]).show(
    vertical=True
)

# %%

# Carregando os dados das transportadoras
df_transportadoras = spark.read.csv(
    "./datasets/transportadoras.csv", header=True
)

# %%

# Visualizando as lindas e colunas do dataset
df_transportadoras.show(truncate=False)

# %%

# Vamos verificar novamente os dado do nosso banco de dados
df.show(1, vertical=True, truncate=False)

# %%

# Renomeando os nomes das colunas para igualar os dados dos dataframes
df_transportadoras = df_transportadoras.withColumnRenamed(
    "IATA Code", "OP_CARRIER"
)

# Verificando novamente o esquema dos dados
df_transportadoras.printSchema()

# %%

# Tipos de joins: inner, cross, outer,full, full_outer, left, left_outer,
# right, right_outer,left_semi, and left_anti.

# Unindo o dataframe original com o dataframe das transportados na coluna
# OP_CARRIER no tipo left (A <- B)

df = df.join(df_transportadoras, on="OP_CARRIER", how="left").drop(
    "OP_CARRIER"
)

# %%

# Verificando o schema dos dados após o junção
df.printSchema()

# %%

# Verificando os nomes das transportadoras no dataframe original
df.show(vertical=True, truncate=False)

# %%

###############################################################################
#                       AGRUPAMENTO E RESUMO DE DADOS                         #
###############################################################################

# %%

# Fluxo de funções: groupby -> agg

# Agrupando por mes e agregando pela média do atraso na chegada
df.groupby("mes").agg(F.avg("ARR_DELAY").alias("media_atraso_chegada")).show()

# %%

# Agrupando e resumindo múltiplas colunas
df_agrupado = df.groupby("mes").agg(
    # média do atraso na chegada
    F.avg("ARR_DELAY").alias("media_atraso_chegada"),
    # soma de voos cancelados
    F.sum("CANCELLED").alias("voos_cancelados"),
    # contagem da quantidade de voos (linhas)
    F.count(F.lit(1)).alias("quantidade_de_voos"),
)

# Visualizar o dataset agrupado
df_agrupado.show()

# %%

# Criando uma nova coluna com a porcentagem dos voos
df_agrupado = df_agrupado.withColumn(
    "porcentagem_voos_cancelados",
    # fórmula: (voos_cancelados/quantidade_de_voos)*100
    (df_agrupado["voos_cancelados"] / df_agrupado["quantidade_de_voos"]) * 100,
)

# Visualizar o dataset agrupado
df_agrupado.show()

# %%

# Filtrando apenas os dados dos voos cancelados
df_cancelados = df.where(df["CANCELLED"] == 1)

# Mostrando os dados de forma vertical
df_cancelados.show(1, vertical=True)

# %%

# Códigos de cancelamento

# A = Por transportadora
# B = Devido às condições climáticas
# C = Pelo sistema nacional de transporte aéreo
# D = Por razões de segurança

# Trocando os dados pelas informações reais
df_cancelados = df_cancelados.withColumn(
    "CANCELLATION_CODE",
    F.when(
        df_cancelados["CANCELLATION_CODE"] == "A", "Por transportadora"
    ).otherwise(df_cancelados["CANCELLATION_CODE"]),
)
df_cancelados = df_cancelados.withColumn(
    "CANCELLATION_CODE",
    F.when(
        df_cancelados["CANCELLATION_CODE"] == "B",
        "Devido às condições climáticas",
    ).otherwise(df_cancelados["CANCELLATION_CODE"]),
)
df_cancelados = df_cancelados.withColumn(
    "CANCELLATION_CODE",
    F.when(
        df_cancelados["CANCELLATION_CODE"] == "C",
        "Pelo sistema nacional de transporte aéreo",
    ).otherwise(df_cancelados["CANCELLATION_CODE"]),
)
df_cancelados = df_cancelados.withColumn(
    "CANCELLATION_CODE",
    F.when(
        df_cancelados["CANCELLATION_CODE"] == "D", "Por razões de segurança"
    ).otherwise(df_cancelados["CANCELLATION_CODE"]),
)

# %%

# Verificando os dados alterados
df_cancelados.show(vertical=True)

# %%

# Renomear coluna para melhor entendimento
df_cancelados = df_cancelados.withColumnRenamed(
    "CANCELLATION_CODE", "motivo_cancelamento"
)

# %%

# Agrupar os dados por motivo do cancelamento e a quantidade de voos
# relacionados
df_voos_cancelados_codigos = (
    # forma de agrupamento
    df_cancelados.groupby("motivo_cancelamento")
    # forma de resumir os dados
    .count()
    # renomeando a nova coluna criada
    .withColumnRenamed("count", "quantidade")
)

# Visulizar os dados
df_voos_cancelados_codigos.show(truncate=False)

# %%

# Passando as informações do cluster Spark para o kernel do python e criando um
# dataframe pandas
pdf_voos_cancelados_codigos = df_voos_cancelados_codigos.toPandas()

# %%

# Agora o dataframe pode ser tratado como um dataframe pandas

# Visualizando as primeiras linhas do dataset
pdf_voos_cancelados_codigos.head()

# %%

# Visualizando as variáveis univiaradas
pdf_voos_cancelados_codigos.describe()

# %%

# Verificando o tipo do dataframe pyspark
print(type(df_voos_cancelados_codigos))

# %%

# Verificando o tipo do dataframe pandas
print(type(pdf_voos_cancelados_codigos))

# %%

###############################################################################
#                         CRIAÇÃO DE GRÁFICOS                                 #
###############################################################################

# criando um gráfico de barras com os dados retornados

# definindo o tamanho da imagem
plt.figure(figsize=(16,12), dpi=300) 

# criando o gráfico de colunas
ax = sns.barplot(
    pdf_voos_cancelados_codigos.sort_values(by="quantidade", ascending=False),
    x="motivo_cancelamento",
    y="quantidade",
    color="#8e44ad"
)
# alterando título e os rótuos dos eixos x e y
ax.set_ylabel("Quantidade voos cancelados", fontsize=14)
ax.set_xlabel("Motivo dos cancelamentos", fontsize=14)
ax.set_title("Voos cancelados e seus motivos", fontsize=14)

# adicionando os labels nas colunas
ax.bar_label(ax.containers[0], fontsize=12)

# plotar o gráfico
plt.show()

#%%

# verificando dados das transportadoras

df_agrupado_companhias = df.groupby("Air Carrier Name").agg(
    # resumir o dado pela média do atraso na chegada e contagem de voos cancelados e total
    # média atraso na chegada
    F.avg("ARR_DELAY").alias("media_atraso"),
    # somatório de voos cancelados
    F.sum("CANCELLED").alias("voos_cancelados"),
    # contagem da quantidade de voos
    F.count(F.lit(1)).alias("quantidade_de_voos"),
)

# mostrar os dados
df_agrupado_companhias.show()

#%%

# passando as informações do driver para a memória do kernel e criando um dataframe pandas
pdf_agrupado_companhias = df_agrupado_companhias.toPandas()

# ajustando os dados para organizar os valores pela media do atraso
pdf_agrupado_companhias = pdf_agrupado_companhias.sort_values(
    by="media_atraso", ascending=True
).round(1)

#%%

# criando gráfico com os dados retornados

# definindo o tamanho da imagem
plt.figure(figsize=(16, 10), dpi=300)

# criando o gráfico de colunas
ax = sns.barplot(pdf_agrupado_companhias,
                 color="#8e44ad",
                 x="Air Carrier Name", 
                 y="media_atraso")

# alterando título e os rótuos dos eixos x e y
ax.set_ylabel("Média de atraso (min)", fontsize=14)
ax.set_xlabel("Código da trasportadora", fontsize=14)
ax.set_title("Média de atraso nas transportadoras", fontsize=14)

# adicionando os labels nas colunas
ax.bar_label(ax.containers[0], fontsize=12)
ax.tick_params(axis='x', rotation=90)

# adicionar linha auxiliar no eixo y = 0
plt.axhline(y=0, color="gray", linestyle="--")

# plotar o gráfico
plt.show()

# %%

###############################################################################
#                    LENDO MÚLTIPLOS DADOS ESTRUTURADOS                       #
###############################################################################

# Coletar a informação do diretório atual#
caminho_atual = os.getcwd()

# Lista de anos para carregar os dados
lista_anos = ["2016", "2017"]

# Lista que vai receber os caminhos completos
caminho_arquivos = []

# Percorrer cada ano da lista de anos
for ano in lista_anos:
    caminho = f"{caminho_atual}/datasets/{ano}.csv"
    caminho_arquivos.append(caminho)

print(caminho_arquivos)

# %%

# Criar um dataframe com múltiplas fontes de bases de dados
df = spark.read.csv(caminho_arquivos, header=True, inferSchema=True)

# %%

# Contagem de linhas do dataframe criado
df.count()

# %%

# Coletando o ano da variável FL_DATE
df = df.withColumn("ano", F.year(df["FL_DATE"]))

# Coletando o mês da variável FL_DATE
df = df.withColumn("mes", F.month(df["FL_DATE"]))

# %%

# Agrupando e resumindo agora os dados por ano
df.groupby("ano").agg(F.avg("ARR_DELAY").alias("media_atraso")).show()

# %%

# Agrupando e resumindo agora os dados por ano e por mês
df.groupby("ano", "mes").agg(
    F.avg("ARR_DELAY").alias("media_atraso")
).orderBy("ano", "mes").show()

# %%

# Retirando o dataframe spark da memória e do disco
df.unpersist()

# %%

# Finalizando a aplicação Spark
spark.stop()

# %%

# ############################## FIM DO SCRIPT ################################
