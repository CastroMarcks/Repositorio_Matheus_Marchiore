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



#os.system('python funcionalsheets_to_date_contactframe.py')

#aqui não precisa alterar nada
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



###------------------------------> Extração do Google Sheets

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

###------------------------------> Expansão do extra_info coluna

def expand_extra_info(csv_file_name):
    # Ler o arquivo CSV
    df = pd.read_csv(csv_file_name)
    
    
   # Converter a coluna 'date_contact' para datetime
    df['date_contact'] = pd.to_datetime(df['date_contact'].str.replace('Z', ''), format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce').dt.tz_localize('UTC')


    df = df[df['date_contact'].dt.month == 4]
    
    if 'strategy_tag' in df.columns:
    # Manter linhas onde a coluna 'strategy_tag' contém a string 'stockout'
        df = df[df['strategy_tag'].str.contains('campaign', case=False, na=False)]


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


# # Carregar o arquivo CSV
df = pd.read_csv('histórico_extra_info.csv')

# Inicializar uma lista para armazenar as novas linhas
linhas_expandidas = []

for index, row in df.iterrows():
    # Adicionar diretamente a linha relevante à lista, já que não precisamos verificar os SKUs
    linhas_expandidas.append({'seller_id': row['seller_id'], 'date_contact': row['date_contact']})

# # Converter a lista em um DataFrame
df_expandido = pd.DataFrame(linhas_expandidas)

# Salvar o novo DataFrame em um arquivo CSV
df_expandido.to_csv('arquivo_formatado_campanhas.csv', index=False)
print("Arquivo Formatado criado com sucesso")


###------------------------------> Limpezas nos dataframes para filtrarmos sellers_id unicos em ordem crescente de data_disparo

df_filtrado = pd.read_csv('arquivo_formatado_campanhas.csv')

# Converter a coluna 'date_contact' para datetime
df_filtrado['date_contact'] = pd.to_datetime(df_filtrado['date_contact'], errors='coerce')

# Ordenar o DataFrame por 'seller_id' e 'date_contact' em ordem crescente para ter a data antiga por primeiro
df_filtrado = df_filtrado.sort_values(by=['seller_id', 'date_contact'], ascending=[False, True])

# Remover duplicatas, mantendo a primeira ocorrência (date_contact mais antiga) para cada 'sku'
df_filtrado = df_filtrado.drop_duplicates(subset='seller_id', keep='first')

# Filtrar para manter apenas as linhas com 'date_contact' no mês atual
#mes_atual = datetime.now().month
df_filtrado = df_filtrado[df_filtrado['date_contact'].dt.month == 4]

# Formatar a coluna 'date_contact' para mostrar apenas a data
df_filtrado['date_contact'] = df_filtrado['date_contact'].dt.strftime('%Y-%m-%d')

print('Quantidade de sellers_id únicos na lista final: {}'.format(df_filtrado['seller_id'].nunique()))

# Salvar o DataFrame filtrado em um arquivo CSV
df_filtrado.to_csv('df_filtrado_campanhas.csv', index=False)


###------------------------------> Agrupamento dos sellers_id e data de disparo

df_filtrado = pd.read_csv('df_filtrado_campanhas.csv')
def agrupar_sellers_por_date_contact(df):
    grouped_sellers_by_date = {}

    for index, row in df.iterrows():
        date = row['date_contact'] 
        seller = row['seller_id']    

        if date not in grouped_sellers_by_date:
            grouped_sellers_by_date[date] = []

        grouped_sellers_by_date[date].append(seller)

    return grouped_sellers_by_date
agrupar_sellers_por_date_contact(df_filtrado)

###------------------------------> Geração de todos os optins dados pelos sellers disparados após data de disparo

def generate_query_optin():
    grouped_date_contact = agrupar_sellers_por_date_contact(df_filtrado)
    
    # Inicializar uma lista vazia para armazenar todos os DataFrames
    all_results = []
    print('Executando query: optins pós disparo') 

    # Iterar por cada data e conjunto de sellers
    for date, sellers in grouped_date_contact.items():
        

        # Divide a lista de sellers em pedaços de tamanho 2000
        sellers_chunks = [sellers[i:i + 2000] for i in range(0, len(sellers), 2000)]
        
        for sellers_chunk in sellers_chunks:
            formatted_sellers_chunk = ', '.join([f"'{seller}'" for seller in sellers_chunk])

            # Montar a query com um pedaço dos sellers
            optins_pos = f"""
            SELECT DISTINCT
            cp.sku,
            bp.seller_id,
            cc.id AS campaign_id,
            1 AS optin_active,
            DATE_FORMAT(MIN(ch.created_at), '%Y-%m-%d') AS data_optin,
            ct.label AS campaign_label,
            cp.subsidy_channel_percentage,
            cp.subsidy_olist_percentage,
            cp.subsidy_seller_percentage
        FROM datalake_silver.campaigns_api_campaigns_campaign AS cc 
        LEFT JOIN datalake_silver.campaigns_api_campaigns_campaignproduct as cp on cp.campaign_id = cc.id
        LEFT JOIN datalake_silver.campaigns_api_campaigns_campaignproducthistory AS ch ON cp.id = ch.campaign_product_id
        LEFT JOIN datalake_silver.campaigns_api_campaigns_tag AS ct ON cc.tag_id = ct.id
        LEFT JOIN datalake_gold.bio_product AS bp ON ch.sku = bp.seller_product_sku
        WHERE ch.status IN ('active', 'accepted')
        AND cc.updated_at = (
            SELECT MAX(updated_at)
            FROM datalake_silver.campaigns_api_campaigns_campaign AS ccc
            WHERE ccc.id = cc.id
        )
        AND cp.status in ('active', 'accepted')
        and cp.updated_at = (select max(cpp.updated_at) from datalake_silver.campaigns_api_campaigns_campaignproduct as cpp where cpp.id = cp.id)
        AND bp.seller_id IN ({formatted_sellers_chunk})
        -- Aqui ajustamos o filtro de data para o último dia do mês
        AND ch.created_at BETWEEN DATE('{date}') AND date_add('day', -1, date_add('month', 1, date_trunc('month', DATE('{date}'))))
        GROUP BY cp.sku, bp.seller_id, cc.id, cc.olister_responsible_email, ct.label,cp.subsidy_channel_percentage, cp.subsidy_olist_percentage, cp.subsidy_seller_percentage

             """

            # Executar a query e armazenar o resultado para o pedaço atual de sellers
            result_df = athena.read('datalake', query=optins_pos)
            all_results.append(result_df)

    # Combinar todos os DataFrames resultantes
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Salvar o DataFrame combinado como CSV
    combined_df.to_csv('OptinsPOS.csv', index=False)
    
    return combined_df
if __name__ == '__main__':
    generate_query_optin()
    
###------------------------------> Agrupamento dos sellers_id, Campaign_id e data de disparo

df_optins = pd.read_csv('OptinsPOS.csv')


def agrupar_sellers_optin_por_date_contact(df):
    grouped_sellers_by_date = {}

    for index, row in df.iterrows():
        date = row['data_optin']
        seller = row['seller_id']
        id = row['campaign_id']

        if date not in grouped_sellers_by_date:
            grouped_sellers_by_date[date] = []

        # Adiciona um dicionário contendo o par SKU e ID à lista da data correspondente
        grouped_sellers_by_date[date].append({'seller_id': seller, 'campaign_id': id})

    return grouped_sellers_by_date

###------------------------------> Geração do GMV baseado nos sellers_id e nas campaign_ids dos sellers disparados

def generate_query_GMV():
    grouped_date_contact = agrupar_sellers_optin_por_date_contact(df_optins)
    
    all_results = []
    print('Executando query: gmv optins disparo') 

    for date, items in grouped_date_contact.items():
        sellers_ids = [item['seller_id'] for item in items]
        ids = [item['campaign_id'] for item in items]

        # Divide a lista em pedaços de tamanho 2000
        sellers_chunks = [sellers_ids[i:i + 2000] for i in range(0, len(sellers_ids), 2000)]
        ids_chunks = [ids[i:i + 2000] for i in range(0, len(ids), 2000)]

        for sellers_id_chunk, ids_chunk in zip(sellers_chunks, ids_chunks):
            formatted_sellers = ', '.join([f"'{seller_id}'" for seller_id in sellers_id_chunk])  # Corrigido aqui
            formatted_ids = ', '.join([f"'{id}'" for id in ids_chunk])

            query_gmv  = f"""
            SELECT 
            so.seller_id,
            soi.campaign_id,
            soi.product_sku,
            soi.id AS order_id,
            oi.channel_slug,
            oi.channel_store,
            SUM(CAST(soi.freight_value AS DOUBLE) + CAST(soi.price AS DOUBLE)) AS gmv
        FROM datalake_silver.orders_api_seller_orders_sellerorder AS so
        LEFT JOIN datalake_silver.orders_api_seller_orders_sellerorderitem AS soi ON soi.seller_order_id = so.id
        LEFT JOIN datalake_silver.sellers_api_sellers_seller ss ON ss.id = so.seller_id
        LEFT JOIN datalake_gold.bio_orderitem AS oi ON oi.seller_order_item_id = soi.id 
        WHERE 1=1
        AND so.status <> 'pending'
        AND so.cancelation_status = ''
        AND so.region = 'br'
        AND ss.plan_type <> '1P'
        AND so.purchase_timestamp >= DATE('{date}') 
        AND so.purchase_timestamp <= date_add('day', -1, date_add('month', 1, date_trunc('month', DATE('{date}'))))
        AND soi.campaign_id in ({formatted_ids})  
        AND so.seller_id in ({formatted_sellers})
        GROUP BY 1,2,3,4,5,6

             """
            result_df = athena.read('datalake', query=query_gmv)
            all_results.append(result_df)
            
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv('gmvCampanha.csv', index=False)
if __name__ == '__main__':generate_query_GMV()


df_gmv_campanha = pd.read_csv('gmvCampanha.csv')

###------------------------------> Geração do GPA baseada nas orders das campanhas aceitas

def generate_query_GPA():
    # Carregar o CSV em um DataFrame
    df_pedidos_incrementais = pd.read_csv('gmvCampanha.csv')
    print('Executando query: GPA das orders')
    # Verificar se 'order_id' é a coluna correta. Se não for, substitua por sua coluna alvo.
    order_ids = df_pedidos_incrementais['order_id'].unique()
    all_results = []

    # Divide a lista de order_ids em pedaços de tamanho 2000
    ids_chunks = [order_ids[i:i + 2000] for i in range(0, len(order_ids), 2000)]

    for ids_chunk in ids_chunks:
        formatted_ids = ', '.join([f"'{id}'" for id in ids_chunk])

        # Montar a query com um pedaço dos ids
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
    bio.product_id as sku,
    bio.seller_order_item_id,
    bio.seller_order_item_code,
    ROUND(SUM(fc.Commission_net + fc.Flat_fee_net + fc.Revenue_markup_net + fc.Subscription_net + fc.Net_COGS + fc.Sales_incentive_wihtout_ads + fc.freight_result), 2) AS gpa,
    bio.campaign_id as campaign_id
FROM
    datalake_gold.bio_orderitem bio
LEFT JOIN
    financial_control fc ON bio.seller_order_item_code = fc.seller_order_item_code
WHERE bio.seller_order_item_id in ({formatted_ids})
GROUP BY
    bio.seller_order_item_id, bio.seller_order_item_code, bio.campaign_id, bio.product_id;;
    """
    # Executar a query e armazenar o resultado para o pedaço atual de ids
        result_df = athena.read('datalake', query=gpa)
        all_results.append(result_df)

    # Concatenar todos os resultados parciais em um único DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)

    # Salvar o DataFrame combinado como CSV
    combined_df.to_csv('gpa_campanha.csv', index=False)

    return combined_df
if __name__ == '__main__':generate_query_GPA()

###------------------------------> Manutenção dos Dataframes
df_optins = pd.read_csv('OptinsPOS.csv')
df_optins['campaign_key'] = df_optins['sku'].astype(str) +  df_optins['campaign_id'].astype(str)

df_gmv_campanha = pd.read_csv('gmvCampanha.csv')
df_gmv_campanha = df_gmv_campanha.drop_duplicates(subset=['order_id'])

df_gpa_campanha = pd.read_csv('gpa_campanha.csv')
df_gpa_campanha = df_gpa_campanha.drop_duplicates(subset=['seller_order_item_id'])

df_merged = pd.merge(df_gmv_campanha,df_gpa_campanha,  left_on='order_id', right_on='seller_order_item_id', how='left')
df_merged.to_csv('df_merged.csv', index=False)
df_merged['campaign_key'] = df_merged['sku'].astype(str) +  df_merged['campaign_id_x'].astype(str)

df_merged.drop('seller_order_item_code', axis=1, inplace=True)
df_merged.drop('seller_id', axis=1, inplace=True)


df_merged = pd.merge(df_optins, df_merged,  left_on='campaign_key', right_on='campaign_key', how='left')
df_merged.drop('seller_order_item_id', axis=1, inplace=True)
df_merged.drop('product_sku', axis=1, inplace=True)
df_merged['updated_at'] = datetime.now().strftime('%Y-%m-%d')

arquivo_formatado_campanhas = pd.read_csv('arquivo_formatado_campanhas.csv')
arquivo_formatado_campanhas_sem_duplicatas = arquivo_formatado_campanhas.drop_duplicates(subset=['seller_id'], keep='first')
df_merged = pd.merge(df_merged, arquivo_formatado_campanhas_sem_duplicatas,  left_on='seller_id', right_on='seller_id', how='left')
df_merged = df_merged.drop_duplicates(subset=['sku_x', 'seller_id', 'campaign_id', 'order_id'], keep='first')
df_merged.drop('sku_y', axis=1, inplace=True)
df_merged.drop('campaign_id_x', axis=1, inplace=True)
df_merged.drop('campaign_id_y', axis=1, inplace=True)
df_merged.rename(columns={'sku_x': 'sku'}, inplace=True)
df_merged.drop('campaign_key', axis=1, inplace=True)
print('Dataframe de campanhas por seller_id pronto: df_campanhas_sellersid_abr.csv')
df_merged.to_csv('df_campanhas_sellersid_abr.csv')


resultados_estoque_fev = pd.read_csv('df_campanhas_sellersid_fev.csv')
resultados_estoque_mar = pd.read_csv('df_campanhas_sellersid_mar.csv')
resultados_estoque_abr = pd.read_csv('df_campanhas_sellersid_abr.csv')



# # Criar uma lista com os DataFrames
dfs = [ resultados_estoque_fev, resultados_estoque_mar, resultados_estoque_abr]

# Concatenar os DataFrames da lista
df_final = pd.concat(dfs, ignore_index=True)

# Salvar o DataFrame resultante em um novo arquivo CSV
df_final.to_csv('df_campanha_all_year_seller_id.csv', index=False)

print("FIM")
