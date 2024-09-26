# %% Imports
import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import io 
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging as log
import awswrangler as wr
import boto3
import os
import random
import time

# Configurações da API
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
auth_base_uri = os.getenv('AUTH_BASE_URI')
rest_base_uri = os.getenv('REST_BASE_URI')

# %% Função para ler dados do Google Sheets
def leitura_sheets():
    creds = Credentials(
        token=os.getenv('TOKEN'),
        refresh_token=os.getenv('REFRESH_TOKEN'),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv('GSHEET_CLIENT_ID'),
        client_secret=os.getenv('GSHEET_CLIENT_SECRET'),
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    client = gspread.authorize(creds)
    spreadsheet = client.open_by_url(os.getenv('GSHEET_URL'))
    sheet = spreadsheet.sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    print("Colunas disponíveis no DataFrame:", df.columns)
    return df

# %% Função para consulta de jornada
def consulta_journey_hist(df):
    client_id_store = os.getenv('CLIENT_ID_STORE')
    client_secret_store = os.getenv('CLIENT_SECRET_STORE')
    auth_url_store = os.getenv('AUTH_URL_STORE')
    base_url_store = os.getenv('BASE_URL_STORE')

    client_id_tiny = os.getenv('CLIENT_ID_TINY')
    client_secret_tiny = os.getenv('CLIENT_SECRET_TINY')
    auth_url_tiny = os.getenv('AUTH_URL_TINY')
    base_url_tiny = os.getenv('BASE_URL_TINY')
    
    def obter_token(client_id, client_secret, auth_url):
        auth_payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }
        auth_response = requests.post(auth_url, json=auth_payload)
        if auth_response.status_code == 200:
            return auth_response.json().get("access_token")
        else:
            return None

    access_token_store = obter_token(client_id_store, client_secret_store, auth_url_store)
    access_token_tiny = obter_token(client_id_tiny, client_secret_tiny, auth_url_tiny)

    if not access_token_store and not access_token_tiny:
        print("Erro ao obter o token de acesso para ambas as APIs.")
        return pd.DataFrame()

    start_date = '2024-09-09T00:00:00Z'
    end_date = '2024-10-10T00:00:00Z'
    
    valid_columns = [
        "ActivityId", "ActivityName", "ActivityType", "ContactKey", 
        "CreatedDate", "EventName"
    ]
    columns_param = ",".join(valid_columns)

    def consultar_historia_jornada(access_token, base_url, definition_id):
        history_url = f"{base_url}interaction/v1/interactions/journeyhistory/download?format=csv&columns={columns_param}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "x-direct-pipe": "true"
        }
        payload = {
            "definitionIds": [definition_id],
            "activityTypes": ["REST", "WHATSAPPACTIVITY", "UPDATECONTACTDATA", "SMSSYNC"],
            "statuses": ["complete"]
        }
        response = requests.post(history_url, headers=headers, json=payload)
        return response

    final_df = pd.DataFrame()
    for index, row in df.iterrows():
        definition_id = row['Id']
        nome_jornada = row['Nome da jornada']
        print(f"Consultando jornada: {nome_jornada}")

        response = consultar_historia_jornada(access_token_store, base_url_store, definition_id)

        if response.status_code != 200 or not response.text:
            print(f"Tentando a API Tiny para {definition_id}")
            response = consultar_historia_jornada(access_token_tiny, base_url_tiny, definition_id)

        if response.status_code == 200 and response.text:
            csv_data = response.text
            temp_df = pd.read_csv(io.StringIO(csv_data))
            final_df = pd.concat([final_df, temp_df], ignore_index=True)
        else:
            print(f"Falha ao baixar o histórico da jornada para {definition_id}: {response.status_code} - {response.text}")

    if not final_df.empty:
        print("Histórico combinado da jornada baixado e salvo com sucesso.")
    else:
        print("Nenhum dado de histórico foi retornado.")

    return final_df

# %% Função para salvar dados no S3
def sandbox_inputador(df, nome_da_tabela):
    caminho_no_s3 = f"s3://dataplat-sandbox-datalake-sales-ops/tables/{nome_da_tabela}/"
    nome_do_banco_de_dados = "sandbox_datalake_sales_ops"

    wr.s3.to_parquet(
        df=df,
        path=caminho_no_s3,
        dataset=True,
        database=nome_do_banco_de_dados,
        table=nome_da_tabela,
        mode="overwrite"
    )

# %% Função para obter o token de acesso
def get_access_token():
    url = f'{auth_base_uri}v2/token'
    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info.get('access_token')
        expires_in = token_info.get('expires_in')
        print(f'Access token obtained. Expires in {expires_in} seconds.')
        return access_token, time.time() + expires_in - 60
    else:
        raise Exception(f'Failed to get access token: {response.status_code} - {response.text}')

# %% Função para consulta ao Data Lake usando Athena
def consulta_lake(consulta):
    df_consulta = athena.read('datalake', query=consulta)
    return df_consulta

# %% Função para salvar dados incrementais
def salvar_dados_incrementais(final_df, caminho_arquivo='combined_journey_history.csv'):
    if os.path.exists(caminho_arquivo):
        df_existente = pd.read_csv(caminho_arquivo)
        df_combinado = pd.concat([df_existente, final_df], ignore_index=True)
        df_combinado = df_combinado.drop_duplicates(subset='Id', keep='last')
    else:
        df_combinado = final_df

    df_combinado.to_csv(caminho_arquivo, index=False, encoding='utf-8')
    print(f"Dados salvos com sucesso no arquivo {caminho_arquivo}.")
    return df_combinado

# %% Configurações e funções adicionais
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

session = boto3.Session(region_name='us-east-1')
athena = Athena(session)

# %% Consulta no Athena
consulta = """SELECT 
    l.id AS lead_id,
    c.document_number AS cnpj
FROM "datalake_silver".leads_api_leadsv2_lead_relationship AS l
LEFT JOIN "datalake_silver".leads_api_leadsv2_company AS c 
    ON c.id = l.company_id
"""

definition_ids = leitura_sheets()
df = pd.DataFrame(definition_ids)
df_conjourney_hist = consulta_journey_hist(definition_ids)

# %% Processamento final e salvamento
df_lake = consulta_lake(consulta)
df_result = pd.merge(df_conjourney_hist, df_lake, left_on='ContactKey', right_on='lead_id', how='left')
df_result.drop(columns=['lead_id'], inplace=True)

df_result.rename(columns={
    'ActivityId': 'keys.jobid',
    'ActivityType': 'ActivityType',
    'ActivityName': 'ActivityName',
    'ContactKey': 'keys.subscriberkey',
    'CreatedDate': 'values.data_envio_format',
    'EventName': 'values.nome_email',
    'cnpj': 'values.cnpj'
}, inplace=True)

loc_0 = df_result.columns.get_loc('values.data_envio_format')
df_result_split = df_result['values.data_envio_format'].str.split(pat='T', expand=True).add_prefix('values.data_envio_format_')
df_result = pd.concat([df_result.iloc[:, :loc_0], df_result_split, df_result.iloc[:, loc_0:]], axis=1)
df_result = df_result.drop(columns=['values.data_envio_format_1'])

# Salvando os dados incrementais
df_conjourney_hist_completo = salvar_dados_incrementais(df_result)

# %% Preparação final e envio para o S3
df_conjourney_hist_completo.drop(columns=['Id'], inplace=True)
df_fuinil = pd.read_csv('Teste2.csv')
df_fuinil['ActivityType'] = "_Sent"
df_fuinil['ActivityName'] = "email"

for column in df_fuinil.columns:
    if column not in df_conjourney_hist_completo.columns:
       df_conjourney_hist_completo[column] = np.nan

df_testeaa = pd.concat([df_fuinil, df_conjourney_hist_completo], ignore_index=True)
df_testeaa.to_csv('Teste4.csv', index=False)

# %% Envio para o S3
df = pd.read_csv("Teste4.csv")
df = df.astype(str)
nome_da_tabela = 'CLM_sent_sfmc'
sandbox_inputador(df, nome_da_tabela)
