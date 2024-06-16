from __future__ import print_function
import os.path
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging as log
import awswrangler as wr
import boto3
#import quickstart as gsh
import os.path
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os 
import json
import random
import time
import datetime
from datetime import datetime



###------------------------------>  Configurações e Bibliotecas necessárias para conexão do ODBC Athena
class Athena: 
    
    def __init__(self, session: boto3.Session):
        
        self.session = session
    def read(self, database: str, file_path: str = None, query: str = None) -> pd.DataFrame:
       
        
        log.info(f'-----------< read query >-----------')
        log.info(f'Database: {database}')
        try:
            _query = open(file_path).read() if file_path else query
            df = wr.athena.read_sql_query(
                _query, 
                database=database,
                workgroup = 'sales-ops',
                boto3_session=self.session
            )
            log.info(f'DataFrame: {df.shape}')
            log.info(f'-------------< done >--------------')
            return df
        except Exception as e:
            log.error(f"Something went wrong executing query Exception: {e}")
#aqui não precisa alterar nada
session = boto3.Session(region_name='us-east-1')
#aqui não precisa alterar nada
athena = Athena(session)

###------------------------------>  Script Inicia extração dos dados aqui 

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of the spreadsheet.
SAMPLE_SPREADSHEET_ID = '1TpT5aOE1cQkqA2GOKRUN2vwXXHt_GD5JRPjGCVLvvIY'
SAMPLE_RANGE_NAME = 'crm!A:Z'


def sheets_to_csv(spreadsheet_id, range_name, csv_file_name):
    """Extrai valores de uma planilha Google Sheets e salva como CSV."""
    
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('client_secret.json'):
                print("Arquivo 'client_secret.json' não encontrado.")
                return
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get('values', [])

        if not values:
            print(f"Nenhum dado encontrado para a planilha '{spreadsheet_id}' com intervalo '{range_name}'.")
            return

        df = pd.DataFrame(values[1:], columns=values[0])
        df.to_csv(csv_file_name, index=False)
        print(f"Dados salvos como '{csv_file_name}'")

    except HttpError as error:
        print(f"Ocorreu um erro na API do Google Sheets: {error}")

# Exemplo de uso
sheets_to_csv(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME, 'histórico_extra_info.csv')

def expand_extra_info(csv_file_name):
    # Ler o arquivo CSV
    df = pd.read_csv(csv_file_name)

        
    if 'strategy_tag' in df.columns:
    # Manter linhas onde a coluna 'strategy_tag' contém a string 'stockout'
        df = df[df['strategy_tag'].str.contains('stockout', case=False, na=False)]


    # Processar cada linha na coluna 'extra_info'
    for index, row in df.iterrows():
        # Verificar se a célula está vazia ou não é um JSON
        if pd.isna(row['extra_info']) or not isinstance(row['extra_info'], str):
            continue

        try:
            # Tentar extrair o JSON
            extra_info = json.loads(row['extra_info'])

            # Tratar o JSON baseado na presença da chave 'extra_info'
            if 'extra_info' in extra_info and isinstance(extra_info['extra_info'], str):
                # Caso a chave 'extra_info' exista e seja uma string, decodificar o JSON aninhado
                extra_info_dict = json.loads(extra_info['extra_info'])
            else:
                # Caso contrário, usar o JSON extraído diretamente
                extra_info_dict = extra_info

            # Adicionar cada chave do dicionário como uma nova coluna no date_contactFrame
            for key, value in extra_info_dict.items():
                df.loc[index, key] = value

        except json.JSONDecodeError:
            print(f"Erro ao decodificar JSON na linha {index}")
            continue

    # Remover a coluna 'extra_info' original
    df.drop(columns=['extra_info'], inplace=True)

    # Salvar o date_contactFrame modificado no mesmo arquivo CSV
    df.to_csv(csv_file_name, index=False)
    print(f"CSV atualizado salvo em '{csv_file_name}'")
# Exemplo de uso
expand_extra_info('histórico_extra_info.csv')


###------------------------------>  Tratar dados e Coluna de Extra_Info do CRM
df = pd.read_csv('histórico_extra_info.csv')

# Inicializar uma lista para armazenar as novas linhas
linhas_expandidas = []

# Iterar sobre cada linha do date_contactFrame original
for index, row in df.iterrows():
    for sku in ['sku_1', 'sku_2', 'sku_3']:
        if pd.notna(row[sku]):
            # Adicionar uma nova linha na lista
            linhas_expandidas.append({'seller_id': row['seller_id'], 'date_contact': row['date_contact'], 'sku': row[sku]})


# Converter a lista em um date_contactFrame
df_expandido = pd.DataFrame(linhas_expandidas)

sheets_to_csv('1OCQAgLb-V3GH-OBJrGvHJZiXsMFGEjramwn4zdgDyjY', 'productsList', 'historicoskuspdf.csv')

dfpdf = pd.read_csv('historicoskuspdf.csv')

dfpdf = dfpdf[dfpdf['model'].str.contains('stock', na=False)]

dfpdf = dfpdf.drop(['id', 'model'], axis=1)

dfpdf = dfpdf.rename(columns={'date': 'date_contact'})

dfpdf = dfpdf[['seller_id','date_contact','sku']]

df_expandido = pd.concat([df_expandido, dfpdf], axis=0)

# Salvar o novo date_contactFrame em um arquivo CSV
df_expandido.to_csv('arquivo_formatado_stockout.csv', index=False)
print("Aquivo Formatado criado com sucesso")

###------------------------------> Filtrar o CRM por Skus unicos disparados, removendo por ordem ascendente de Data_disparo

df = pd.read_csv('arquivo_formatado_stockout.csv')

# Filtrar o DataFrame para manter apenas as linhas onde 'sku' tem 16 ou mais caracteres
df_filtrado = df[df['sku'].apply(lambda x: len(str(x)) >= 16)].copy()

# Converter a coluna 'date_contact' para datetime
df_filtrado['date_contact'] = pd.to_datetime(df_filtrado['date_contact'].str.replace('Z', ''), format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce').dt.tz_localize('UTC')

# Filtrar para manter apenas as linhas com 'date_contact' no mês atual
mes_atual = datetime.now().month
df_filtrado = df_filtrado[df_filtrado['date_contact'].dt.month == 4]

# # Ordenar o DataFrame por 'sku' e 'date_contact' em ordem crescente para ter a data antiga por primeiro
df_filtrado = df_filtrado.sort_values(by=['sku', 'date_contact'], ascending=[False, True])

# Remover duplicatas, mantendo a primeira ocorrência (date_contact mais antiga) para cada 'sku'
df_filtrado = df_filtrado.drop_duplicates(subset='sku', keep='first')

# Formatar a coluna 'date_contact' para mostrar apenas a data
df_filtrado['date_contact'] = df_filtrado['date_contact'].dt.strftime('%Y-%m-%d')

print('Quantidade de produtos únicos na lista final: {}'.format(df_filtrado['sku'].nunique()))

# Salvar o DataFrame filtrado em um arquivo CSV
df_filtrado.to_csv('df_filtrado_stockout.csv', index=False)


###------------------------------> Ínicio do agrupamento de skus por data de disparo, e execução das querys para Pedidos e histórico de Estoque

df_filtrado = pd.read_csv('df_filtrado_stockout.csv')

def agrupar_skus_por_date_contact(df):
    grouped_skus_by_date = {}

    for index, row in df.iterrows():
        date = row['date_contact'] 
        sku = row['sku']    

        if date not in grouped_skus_by_date:
            grouped_skus_by_date[date] = []

        grouped_skus_by_date[date].append(sku)

    return grouped_skus_by_date
agrupar_skus_por_date_contact(df_filtrado)

def generate_query_pedidos():
    grouped_date_contact = agrupar_skus_por_date_contact(df_filtrado)
    
    # Inicializar uma lista vazia para armazenar todos os date_contactframes
    all_results = []
    print('Executando query: pedidos pós disparo') 

    # Iterar por cada date_contact e conjunto de SKUs
    for date, skus in grouped_date_contact.items():
    
        formatted_skus = ', '.join([f"'{sku}'" for sku in skus])

        # Montar a query
        pedidos_pos = f"""
     SELECT DISTINCT
    oi.product_id,
    date_format(cast(oi.purchase_timestamp as date), '%Y-%m-%d') as purchase_timestamp,
    bp.full_name as produto,
    bp.seller_id as seller_id,
    oi.seller_order_item_id as order_id,
    oi.freight_value AS freight_value,
    oi.price AS pd_price
    FROM datalake_gold.bio_orderitem as oi 
    LEFT JOIN datalake_gold.bio_product as bp on bp.product_id = oi.product_id 
    WHERE oi.product_id in ({formatted_skus})
    AND oi.purchase_timestamp >= DATE('{date}')
    AND oi.purchase_timestamp < date_trunc('month', DATE('{date}') + interval '1' month)
    AND oi.cancelation_status = ''
    AND oi.region = 'br'
    AND oi.status_seller_order <> 'pending'
    """

        # Executar a query e armazenar o resultado
        result_df = athena.read('datalake', query=pedidos_pos)
        all_results.append(result_df)

    # Combinar todos os date_contactframes resultantes
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Salvar o date_contactframe combinado como CSV
    combined_df.to_csv('PedidosPOS.csv', index=False)
    
    return combined_df
if __name__ == '__main__': generate_query_pedidos()

def generate_query_pré_disparo():
    grouped_date_contact = agrupar_skus_por_date_contact(df_filtrado)
    
    all_results = []

    print('Executando query da quantidade pré disparo')
    for date, skus in grouped_date_contact.items():

        formatted_skus = ', '.join([f"'{sku}'" for sku in skus])

            # Aqui você construirá sua query usando 'formatted_skus' e 'date'
        query_estoque_no_disparo =f"""
               -- utilizando a tabela de data para completar os dias faltantes da tabela de stock history
        with
        produtoFinal as (
        with 
        calendar as (select cast(dia as date) dia
                        from datalake_silver.operations_orders_bi_ops_datas
                        where cast(dia as date) > date_add('day', -60, cast(at_timezone(current_date,'America/Sao_Paulo') as timestamp)) -- Ãºltimos 65 dias
                            and cast(dia as date) <= cast(at_timezone(current_date,'America/Sao_Paulo') as timestamp)),
                            
        products as (
                select distinct
                product_id as sku
                from datalake_gold.bio_product
             
            ),
                
        baseFinal as (
                select 
                c.dia,
                p.sku
                from calendar as c
                cross join products as p),    
                        
        stockNRow as ( -- apenas insere o número da linha para determinar o Ãºltimo valor de estoque por dia e sku
                                select (at_timezone(sh.updated_at,'America/Sao_Paulo')) as updated_at,
                                    sh.seller_product_id,
                                    bp.product_id as sku,
                                    sh.quantity,
                                    row_number() over(partition by sh.seller_product_id, cast(at_timezone(sh.updated_at,'America/Sao_Paulo') as date) order by at_timezone(sh.updated_at,'America/Sao_Paulo') desc) as nrow --nrow para identificar o Ãºltimo valor do dia
                                from datalake_silver.products_api_seller_products_stockhistory as sh
                                inner join datalake_gold.bio_product as bp
                                    on bp.seller_product_id = sh.seller_product_id 
                                where at_timezone(sh.updated_at,'America/Sao_Paulo') >= date_add('day', -120, at_timezone(current_timestamp,'America/Sao_Paulo'))
                                
                                ),
                        
        purchase as (
            select
                cast((at_timezone(purchase_timestamp, 'America/Sao_Paulo')) as date) as purchase_date,
                product_id as sku,
                round(sum(quantity)) as quantity,
                round(sum(gmv),2) as gmv
            from datalake_gold.bio_orderitem 
            where at_timezone(purchase_timestamp, 'America/Sao_Paulo') >= date_add('day', -120, at_timezone(current_timestamp,'America/Sao_Paulo')) 
            group by 1, 2
            ) , 
            
        stockBefore60 as (
                            select  --cast(updated_at at time zone 'America/Sao_paulo' as date) as updated_at,
                                date_add('day',-60, at_timezone(current_timestamp,'America/Sao_Paulo')) as updated_at,
                                sh.seller_product_id,
                                bp.product_id as sku,
                                max_by(quantity, at_timezone(sh.updated_at,'America/Sao_Paulo')) as quantity,
                                0 as nrow --crio essa coluna nrow para poder fazer o union com a stockNRow
                                from datalake_silver.products_api_seller_products_stockhistory as sh
                                    inner join datalake_gold.bio_product as bp 
                                    on bp.seller_product_id = sh.seller_product_id 
                                where at_timezone(sh.updated_at,'America/Sao_Paulo')  < date_add('day', -60, at_timezone(current_timestamp,'America/Sao_Paulo')) -- Ãºltimos 30 dias
                                    group by 1,2,3),
                            
        latestStock as (
                                select * from stockNRow where nrow = 1 --apenas Ãºltimo valor de estoque do dia de produtos aprovados e ativos 
                                union 
                                select * from stockBefore60 
                            ),
        
        priceNRow as (
                                select (at_timezone(b.updated_at,'America/Sao_Paulo')) as updated_at,
                                    b.seller_product_id,
                                    bp.product_id as sku,
                                    coalesce(b.offer, 0) as price,
                                    row_number() over(partition by b.seller_product_id, cast(at_timezone(b.updated_at,'America/Sao_Paulo') as date) order by at_timezone(b.updated_at,'America/Sao_Paulo') desc) as nrow --nrow para identificar o Ãºltimo valor do dia
                                from  datalake_silver.products_api_seller_products_pricehistory as b
                                inner join datalake_gold.bio_product bp
                                    on bp.seller_product_id = b.seller_product_id 
                                where at_timezone(b.updated_at,'America/Sao_Paulo') >= date_add('day', -60, at_timezone(current_timestamp,'America/Sao_Paulo')) -- Ãºltimos 30 dias
                                ),
        
        priceBefore60 as (
                            select  --cast(updated_at at time zone 'America/Sao_paulo' as date) as updated_at,
                                date_add('day',-60, at_timezone(current_timestamp,'America/Sao_Paulo')) as updated_at,
                                    b.seller_product_id,
                                    bp.product_id as sku,
                                coalesce (max_by(b.offer, at_timezone(b.updated_at,'America/Sao_Paulo')), 0) as price,
                                0 as nrow --crio essa coluna nrow para poder fazer o union com a stockNRow
                                from  datalake_silver.products_api_seller_products_pricehistory as b
                                inner join datalake_gold.bio_product bp
                                    on bp.seller_product_id = b.seller_product_id 
                                where at_timezone(b.updated_at,'America/Sao_Paulo')  < date_add('day', -60, at_timezone(current_timestamp,'America/Sao_Paulo'))-- Ãºltimos 30 dias
                                    group by 1,2,3),
        
        latestPrice as (
                                select * from priceNRow where nrow = 1 --apenas Ãºltimo valor de estoque do dia de produtos aprovados e ativos 
                                union 
                                select * from priceBefore60
                            ),
        
        prodInfo as (
                    select  
                    cast(updated_at at time zone 'America/Sao_Paulo' as date) as dateStart,
                    sku,
                    quantity as stock
                    from latestStock),
                    
        periodGap as (
                    select
                    dateStart,
                    sku,
                    stock as estoque,
                    coalesce(lag(dateStart, 1) over(partition by sku order by dateStart desc), cast(date_add('day', 1, at_timezone(current_timestamp,'America/Sao_Paulo')) as date)) as endDate
                            from prodInfo),
                                                            
        priceInfo as (
                select
                cast(updated_at at time zone 'America/Sao_Paulo' as date) as dateStart,
                sku,
                price
                from latestPrice),
            
        pricePeriodGap as (
                select
                dateStart,
                sku,
                price,
                coalesce(lag(dateStart, 1) over(partition by sku order by dateStart desc), cast(date_add('day', 1, at_timezone(current_timestamp,'America/Sao_Paulo')) as date)) as endDate
                from priceInfo)
        
        select
            bf.dia,
            bf.sku,
            pg.estoque as stock,
            ppg.price,
            coalesce(prc.quantity, 0) quantity,
            coalesce (prc.gmv, 0) gmv
        from baseFinal as bf
        join periodGap as pg 
            on bf.dia >= pg.dateStart and bf.dia < pg.endDate and bf.sku = pg.sku
        left join pricePeriodGap as ppg
            on bf.dia >= ppg.dateStart and bf.dia < ppg.endDate and bf.sku = ppg.sku
        left join purchase as prc 
            on prc.sku = bf.sku and prc.purchase_date = bf.dia
        order by 1 asc)

        select 
            dia,
            date_format(dia, '%Y-%m') as mes_ano,
            sku as product_id,
            pf.stock as estoque_pre
        from produtoFinal as pF
        inner join datalake_gold.bio_product as bp on bp.product_id = pF.sku
        where sku in ({formatted_skus})
        and dia = date('{date}') - INTERVAL '1' DAY
        order by dia desc
         """
        
        
        result_df = athena.read('datalake', query=query_estoque_no_disparo)
        all_results.append(result_df) 
        
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv('qntde_pré_disparo.csv', index=False)
    
    return combined_df
if __name__ == '__main__': generate_query_pré_disparo()

def generate_query_estoque_pós_disparo():
    grouped_date_contact = agrupar_skus_por_date_contact(df_filtrado)
    
    all_results = []
    print('Executando query da quantidade pós disparo')
    
    for date, skus in grouped_date_contact.items():

        formatted_skus = ', '.join([f"'{sku}'" for sku in skus])

            # Aqui você construirá sua query usando 'formatted_skus' e 'date'
        query_estoque_pós_disparo =f"""
               -- utilizando a tabela de data para completar os dias faltantes da tabela de stock history
        with
        produtoFinal as (
        with 
        calendar as (select cast(dia as date) dia
                        from datalake_silver.operations_orders_bi_ops_datas
                        where cast(dia as date) > date_add('day', -60, cast(at_timezone(current_date,'America/Sao_Paulo') as timestamp)) -- Ãºltimos 65 dias
                            and cast(dia as date) <= cast(at_timezone(current_date,'America/Sao_Paulo') as timestamp)),
                            
        products as (
                select distinct
                product_id as sku
                from datalake_gold.bio_product
             
            ),
                
        baseFinal as (
                select 
                c.dia,
                p.sku
                from calendar as c
                cross join products as p),    
                        
        stockNRow as ( -- apenas insere o número da linha para determinar o Ãºltimo valor de estoque por dia e sku
                                select (at_timezone(sh.updated_at,'America/Sao_Paulo')) as updated_at,
                                    sh.seller_product_id,
                                    bp.product_id as sku,
                                    sh.quantity,
                                    row_number() over(partition by sh.seller_product_id, cast(at_timezone(sh.updated_at,'America/Sao_Paulo') as date) order by at_timezone(sh.updated_at,'America/Sao_Paulo') desc) as nrow --nrow para identificar o Ãºltimo valor do dia
                                from datalake_silver.products_api_seller_products_stockhistory as sh
                                inner join datalake_gold.bio_product as bp
                                    on bp.seller_product_id = sh.seller_product_id 
                                where at_timezone(sh.updated_at,'America/Sao_Paulo') >= date_add('day', -120, at_timezone(current_timestamp,'America/Sao_Paulo'))
                                
                                ),
                        
        purchase as (
            select
                cast((at_timezone(purchase_timestamp, 'America/Sao_Paulo')) as date) as purchase_date,
                product_id as sku,
                round(sum(quantity)) as quantity,
                round(sum(gmv),2) as gmv
            from datalake_gold.bio_orderitem 
            where at_timezone(purchase_timestamp, 'America/Sao_Paulo') >= date_add('day', -120, at_timezone(current_timestamp,'America/Sao_Paulo')) 
            group by 1, 2
            ) , 
            
        stockBefore60 as (
                            select  --cast(updated_at at time zone 'America/Sao_paulo' as date) as updated_at,
                                date_add('day',-60, at_timezone(current_timestamp,'America/Sao_Paulo')) as updated_at,
                                sh.seller_product_id,
                                bp.product_id as sku,
                                max_by(quantity, at_timezone(sh.updated_at,'America/Sao_Paulo')) as quantity,
                                0 as nrow --crio essa coluna nrow para poder fazer o union com a stockNRow
                                from datalake_silver.products_api_seller_products_stockhistory as sh
                                    inner join datalake_gold.bio_product as bp 
                                    on bp.seller_product_id = sh.seller_product_id 
                                where at_timezone(sh.updated_at,'America/Sao_Paulo')  < date_add('day', -60, at_timezone(current_timestamp,'America/Sao_Paulo')) -- Ãºltimos 30 dias
                                    group by 1,2,3),
                            
        latestStock as (
                                select * from stockNRow where nrow = 1 --apenas Ãºltimo valor de estoque do dia de produtos aprovados e ativos 
                                union 
                                select * from stockBefore60 
                            ),
        
        priceNRow as (
                                select (at_timezone(b.updated_at,'America/Sao_Paulo')) as updated_at,
                                    b.seller_product_id,
                                    bp.product_id as sku,
                                    coalesce(b.offer, 0) as price,
                                    row_number() over(partition by b.seller_product_id, cast(at_timezone(b.updated_at,'America/Sao_Paulo') as date) order by at_timezone(b.updated_at,'America/Sao_Paulo') desc) as nrow --nrow para identificar o Ãºltimo valor do dia
                                from  datalake_silver.products_api_seller_products_pricehistory as b
                                inner join datalake_gold.bio_product bp
                                    on bp.seller_product_id = b.seller_product_id 
                                where at_timezone(b.updated_at,'America/Sao_Paulo') >= date_add('day', -60, at_timezone(current_timestamp,'America/Sao_Paulo')) -- Ãºltimos 30 dias
                                ),
        
        priceBefore60 as (
                            select  --cast(updated_at at time zone 'America/Sao_paulo' as date) as updated_at,
                                date_add('day',-60, at_timezone(current_timestamp,'America/Sao_Paulo')) as updated_at,
                                    b.seller_product_id,
                                    bp.product_id as sku,
                                coalesce (max_by(b.offer, at_timezone(b.updated_at,'America/Sao_Paulo')), 0) as price,
                                0 as nrow --crio essa coluna nrow para poder fazer o union com a stockNRow
                                from  datalake_silver.products_api_seller_products_pricehistory as b
                                inner join datalake_gold.bio_product bp
                                    on bp.seller_product_id = b.seller_product_id 
                                where at_timezone(b.updated_at,'America/Sao_Paulo')  < date_add('day', -60, at_timezone(current_timestamp,'America/Sao_Paulo'))-- Ãºltimos 30 dias
                                    group by 1,2,3),
        
        latestPrice as (
                                select * from priceNRow where nrow = 1 --apenas Ãºltimo valor de estoque do dia de produtos aprovados e ativos 
                                union 
                                select * from priceBefore60
                            ),
        
        prodInfo as (
                    select  
                    cast(updated_at at time zone 'America/Sao_Paulo' as date) as dateStart,
                    sku,
                    quantity as stock
                    from latestStock),
                    
        periodGap as (
                    select
                    dateStart,
                    sku,
                    stock as estoque,
                    coalesce(lag(dateStart, 1) over(partition by sku order by dateStart desc), cast(date_add('day', 1, at_timezone(current_timestamp,'America/Sao_Paulo')) as date)) as endDate
                            from prodInfo),
                                                            
        priceInfo as (
                select
                cast(updated_at at time zone 'America/Sao_Paulo' as date) as dateStart,
                sku,
                price
                from latestPrice),
            
        pricePeriodGap as (
                select
                dateStart,
                sku,
                price,
                coalesce(lag(dateStart, 1) over(partition by sku order by dateStart desc), cast(date_add('day', 1, at_timezone(current_timestamp,'America/Sao_Paulo')) as date)) as endDate
                from priceInfo)
        
        select
            bf.dia,
            bf.sku,
            pg.estoque as stock,
            ppg.price,
            coalesce(prc.quantity, 0) quantity,
            coalesce (prc.gmv, 0) gmv
        from baseFinal as bf
        join periodGap as pg 
            on bf.dia >= pg.dateStart and bf.dia < pg.endDate and bf.sku = pg.sku
        left join pricePeriodGap as ppg
            on bf.dia >= ppg.dateStart and bf.dia < ppg.endDate and bf.sku = ppg.sku
        left join purchase as prc 
            on prc.sku = bf.sku and prc.purchase_date = bf.dia
        order by 1 asc)

        SELECT 
    DATE_FORMAT(dia, '%Y-%m') AS mes_ano,
    sku,
    MAX(pf.stock) AS quantity_pos_disparo
FROM produtoFinal AS pf
INNER JOIN datalake_gold.bio_product AS bp ON bp.product_id = pf.sku
WHERE sku IN ({formatted_skus})
AND dia > DATE('{date}')
AND MONTH(dia) = MONTH(DATE('{date}'))
AND YEAR(dia) = YEAR(DATE('{date}'))
GROUP BY DATE_FORMAT(dia, '%Y-%m'), sku
ORDER BY quantity_pos_disparo DESC;
         """
        
        
        result_df = athena.read('datalake', query=query_estoque_pós_disparo)
        all_results.append(result_df) 
        
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv('qntde_pós_disparo.csv', index=False)
    
    return combined_df
if __name__ == '__main__': generate_query_estoque_pós_disparo()  
    

###------------------------------> Modelagagem dos dados para subtrair quantidade de pedidos pelo estoque anterior ao disparo

###  - - - Carregando os DataFrames
main_df = pd.read_csv('df_filtrado_stockout.csv')
df_pedidos = pd.read_csv('PedidosPOS.csv')
df_quantity = pd.read_csv('qntde_pré_disparo.csv')  
# df_total_gmv = pd.read_csv('total_gmv.csv')
df_estoque_pós = pd.read_csv('qntde_pós_disparo.csv')

# Contando pedidos por produto no df_pedidos
pedido_count = df_pedidos.groupby('product_id')['order_id'].count().reset_index()
pedido_count.rename(columns={'order_id': 'pedido_count'}, inplace=True)

# Juntando df_quantity com pedido_count
df_combined = pd.merge(df_quantity, pedido_count, on='product_id', how='left')

# Inicializa um DataFrame vazio para armazenar os pedidos ajustados
df_pedidos_incrementais = pd.DataFrame()
for index, row in df_combined.iterrows():
    product_id = row['product_id']
    estoque = int(row['estoque_pre'])
    pedidos = 0 if pd.isna(row['pedido_count']) else int(row['pedido_count'])

    # Filtra df_pedidos para o produto atual
    df_temp = df_pedidos[df_pedidos['product_id'] == product_id]

    if pedidos > estoque:
        # Mantém um número de linhas igual a 'pedidos - estoque'
        df_temp = df_temp.sample(pedidos - estoque)
    elif pedidos < estoque:
        # Se o estoque for maior que os pedidos, remove todas as linhas para esse SKU
        df_temp = pd.DataFrame()  # Cria um DataFrame vazio para esse SKU

    # Adiciona ao DataFrame ajustado
    df_pedidos_incrementais = pd.concat([df_pedidos_incrementais, df_temp])
    
    
df_pedidos_incrementais.to_csv('df_pedidos_incrementais_abr.csv', index=False)

###------------------------------> Geração do GPA baseada nos pedidos incrementais, procvs de informações entre as bases
def generate_query_GPA():
   
    # Carregar o CSV em um DataFrame
    df_pedidos_incrementais = pd.read_csv('df_pedidos_incrementais_abr.csv')

    # Verificar se 'order_id' é a coluna correta. Se não for, substitua por sua coluna alvo.
    order_ids = df_pedidos_incrementais['order_id'].unique()
    all_results = []

    ids_chunks = [order_ids[i:i + 2000] for i in range(0, len(order_ids), 2000)]

    for ids_chunk in ids_chunks:
        formatted_ids = ', '.join([f"'{id}'" for id in ids_chunk])
    
    

            # Montar a query
        gpa = f"""
        WITH financial_control AS (    
        SELECT
            seller_order_item_code,
            DATE_TRUNC('week', accounted_at) AS semana_competencia,
            DATE_TRUNC('month', accounted_at) AS mes_competencia,
            CAST(SUM(
                CASE
                    WHEN provision_type IN ('seller_transfer', 'seller_transfer_chargeback') THEN -1 * relative_amount
                END) AS DECIMAL(24,2)) AS Gmv,
            coalesce(CAST(SUM(
                CASE
                    WHEN provision_type IN ('seller_commission', 'seller_commission_chargeback', 'seller_commission_fine', 'seller_commission_fine_chargeback') THEN -1 * relative_amount
                    ELSE 0
                END) * (1 - 0.1125) AS DECIMAL(24,2)),0) AS Commission_net,
            coalesce(CAST(SUM(
                CASE
                    WHEN provision_type IN ('seller_flat_fee', 'seller_flat_fee_chargeback', 'seller_flat_fee_fine', 'seller_flat_fee_fine_chargeback') THEN -1 * relative_amount
                    ELSE 0
                END) * (1 - 0.1125) AS DECIMAL(24,2)),0) AS Flat_fee_net,
            coalesce(CAST(SUM(
                CASE
                    WHEN provision_type IN ('seller_markup', 'seller_markup_chargeback') THEN -1 * relative_amount
                    ELSE 0
                END) * (1 - 0.1125) AS DECIMAL(24,2)),0) AS Revenue_markup_net,
            coalesce(CAST(SUM(
                CASE
                    WHEN provision_type IN ('seller_subscription', 'seller_subscription_chargeback') THEN -1 * relative_amount
                    ELSE 0
                END) * (1 - 0.0565) AS DECIMAL(24,2)),0) AS Subscription_net,
            coalesce(CAST(SUM(
                CASE
                    WHEN provision_type IN ('marketplace_commission_discount', 'marketplace_commission', 'marketplace_commission_chargeback',
                                            'marketplace_commission_fine', 'marketplace_commission_fine_chargeback',
                                            'marketplace_flat_fee', 'marketplace_flat_fee_chargeback',
                                            'marketplace_flat_fee_fine', 'marketplace_flat_fee_fine_chargeback') THEN relative_amount
                    ELSE 0
                END) * (1 - 0.0925) AS DECIMAL(24,2)) * -1,0) AS Net_COGS,
            coalesce(CAST(SUM(
                CASE
                    WHEN provision_type IN ('seller_incentive_value', 'seller_incentive_value_chargeback', 'seller_subsidy', 'seller_subsidy_chargeback', 'seller_price_discount', 'seller_price_discount_chargeback', 'marketplace_subsidy', 'marketplace_subsidy_chargeback', 'seller_flat_freight_reduced', 'seller_flat_freight_reduced_chargeback', 'seller_freight_reduced', 'seller_freight_reduced_chargeback', 'seller_markup_incentive', 'seller_markup_incentive_chargeback', 'seller_operation_incentive', 'seller_operation_incentive_chargeback') THEN -1 * relative_amount
                    ELSE 0
                END) AS DECIMAL(24,2)),0) AS Sales_incentive_wihtout_ads,
            coalesce(CAST(SUM(
                CASE
                    WHEN provision_type IN ('seller_flat_freight_deduction', 'seller_flat_freight_deduction_chargeback',
                                            'seller_freight_buyer_deduction', 'seller_freight_buyer_deduction_chargeback',
                                            'seller_freight_increased', 'seller_freight_increased_chargeback',
                                            'carrier_quoted', 'carrier_quoted_chargeback', 'carrier_quoted_adjustment',
                                            'driver_first_mile', 'driver_complements_first_mile') THEN -1 * relative_amount
                    ELSE 0
                END) AS DECIMAL(24,2)),0) AS freight_result
        FROM datalake_silver.controller_api_accountingsellerstore_accountingsellerstore
        GROUP BY seller_order_item_code, DATE_TRUNC('week', accounted_at), DATE_TRUNC('month', accounted_at)
    )
    SELECT
        bio.seller_order_item_id,
        bio.seller_order_item_code,
        ROUND(SUM(fc.Commission_net + fc.Flat_fee_net + fc.Revenue_markup_net + fc.Subscription_net + fc.Net_COGS + fc.Sales_incentive_wihtout_ads + fc.freight_result), 2) AS gross_profit_adjusted
    FROM
        datalake_gold.bio_orderitem bio
    LEFT JOIN
        financial_control fc ON bio.seller_order_item_code = fc.seller_order_item_code
    WHERE bio.seller_order_item_id in ({formatted_ids})
    GROUP BY
        bio.seller_order_item_id, bio.seller_order_item_code;
        """
        result_df = athena.read('datalake', query=gpa)
        all_results.append(result_df)

    # Concatenar todos os resultados parciais em um único DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)

    # Salvar o date_contactframe combinado como CSV
    combined_df.to_csv('gpa.csv', index=False)
        
    return combined_df

if __name__ == '__main__': generate_query_GPA()



df_gpa = pd.read_csv('gpa.csv')
df_incrementais = pd.read_csv('df_pedidos_incrementais_abr.csv')

df_gpa_filtered = df_gpa[['seller_order_item_id', 'gross_profit_adjusted']]
merged_df = pd.merge(df_incrementais, df_gpa_filtered, left_on='order_id', right_on='seller_order_item_id', how='left')
merged_df['updated_at'] = datetime.now().strftime('%Y-%m-%d')

print("Base de pedidos incrementais final de Stockout gerada: df_incremental_abr.csv")
merged_df.to_csv('df_incremental_final_abr.csv')

merged_df_estoque = df_estoque_pós[['sku','quantity_pos_disparo']]
merged_df_estoque = pd.merge(df_quantity, merged_df_estoque, left_on='product_id', right_on='sku', how='left')
merged_df_estoque = merged_df_estoque.drop(['sku', 'mes_ano'], axis=1)
merged_df_estoque['updated_at'] = datetime.now().strftime('%Y-%m-%d')
print("Base de histórico de estoque final gerada: df_estoque_final_abr.csv")
merged_df_estoque.to_csv('df_estoque_final_abr.csv')


# ###------------------------------> Agrupamento das bases de todos os meses para upar na AWS 

resultados_incremental_jan = pd.read_csv('df_incremental_final_jan.csv')
resultados_incremental_fev = pd.read_csv('df_incremental_final_fev.csv')
resultados_incremental_mar = pd.read_csv('df_incremental_final_mar.csv')
resultados_incremental_abr = pd.read_csv('df_incremental_final_abr.csv')

# Criar uma lista com os DataFrames
dfs = [resultados_incremental_jan, resultados_incremental_fev, resultados_incremental_mar, resultados_incremental_abr]

# Concatenar os DataFrames da lista
df_final = pd.concat(dfs, ignore_index=True)

# Salvar o DataFrame resultante em um novo arquivo CSV
df_final.to_csv('df_incremental_all_year.csv', index=False)



resultados_estoque_jan = pd.read_csv('df_estoque_final_jan.csv')
resultados_estoque_fev = pd.read_csv('df_estoque_final_fev.csv')
resultados_estoque_mar = pd.read_csv('df_estoque_final_mar.csv')
resultados_estoque_abr = pd.read_csv('df_estoque_final_abr.csv')

# # Criar uma lista com os DataFrames
dfs = [resultados_estoque_jan, resultados_estoque_fev, resultados_estoque_mar, resultados_estoque_abr]

# Concatenar os DataFrames da lista
df_final = pd.concat(dfs, ignore_index=True)

# Salvar o DataFrame resultante em um novo arquivo CSV
df_final.to_csv('df_estoque_all_year.csv', index=False)
print("FIM")

