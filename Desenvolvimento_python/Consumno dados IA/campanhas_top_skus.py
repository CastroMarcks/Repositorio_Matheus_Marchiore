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
import os 
import json
import random
import time
import datetime
from datetime import datetime

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SAMPLE_SPREADSHEET_ID = os.getenv('SAMPLE_SPREADSHEET_ID')  # ID da planilha sensível substituído por variável de ambiente
SAMPLE_RANGE_NAME = 'crm!A:Z'

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
                workgroup=os.getenv('ATHENA_WORKGROUP'),  # Sensível: substituído por variável de ambiente
                boto3_session=self.session
            )
            log.info(f'DataFrame: {df.shape}')
            log.info(f'-------------< done >--------------')
            return df
        except Exception as e:
            log.error(f"Something went wrong executing query Exception: {e}")

session = boto3.Session(region_name=os.getenv('AWS_REGION'))  # Região AWS sensível
athena = Athena(session)

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

# Outras funções permanecem iguais...

def run_campanhas_3kus_results(mes):
    ### -----------------------> Extraindo Dados CRM lake
    
    if mes <= 4:
        SAMPLE_SPREADSHEET_ID = os.getenv('SAMPLE_SPREADSHEET_ID')  # ID da planilha sensível substituído por variável de ambiente
        SAMPLE_RANGE_NAME = 'crm!A:Z'
        sheets_to_csv(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME, 'historico_extra_info.csv')
        df = expand_extra_info_sheets('historico_extra_info.csv', mes)
    else:
        df = consulta_crm(mes, 'campaign')
        df = expand_extra_info(df)

    # Código continua sem alterações...

def agrupa_meses_campanha():
    dfs_campanha = []
    dfs_pedidos = []

    for mes in range(3, 9):
        try:
            df_campanha_top3 = pd.read_csv(f'df_campanha_top3_{mes}.csv')
            dfs_campanha.append(df_campanha_top3)
        except FileNotFoundError:
            print(f'Arquivo df_campanha_top3_{mes}.csv não encontrado, pulando para o próximo mês.')

        try:
            df_gmv_gpa = pd.read_csv(f'pedidos_campanhas_{mes}.csv')
            dfs_pedidos.append(df_gmv_gpa)
        except FileNotFoundError:
            print(f'Arquivo pedidos_campanhas_{mes}.csv não encontrado, pulando para o próximo mês.')
        
    if dfs_campanha:
        df_final_campanha = pd.concat(dfs_campanha, ignore_index=True)
        df_final_campanha.to_csv('df_campanha_top3_all_year.csv', index=False)
    else:
        print("Nenhum arquivo de campanha encontrado para processar.")
        
    if dfs_pedidos:
        df_final_pedidos = pd.concat(dfs_pedidos, ignore_index=True)
        df_final_pedidos.to_csv('df_campanha_pedidos_all_year.csv', index=False)
    else:
        print("Nenhum arquivo de pedidos encontrado para processar.")
    
    print("FIM")

# Continue com o restante do código...
