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

# The ID and range of the spreadsheet.
SAMPLE_SPREADSHEET_ID = os.getenv('SAMPLE_SPREADSHEET_ID')  # Substituído por variável de ambiente
SAMPLE_RANGE_NAME = 'crm!A:Z'

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
                workgroup=os.getenv('ATHENA_WORKGROUP'),  # Substituído por variável de ambiente
                boto3_session=self.session
            )
            log.info(f'DataFrame: {df.shape}')
            log.info(f'-------------< done >--------------')
            return df
        except Exception as e:
            log.error(f"Something went wrong executing query Exception: {e}")

session = boto3.Session(region_name=os.getenv('AWS_REGION'))  # Substituído por variável de ambiente
athena = Athena(session)

###------------------------------>  Script Inicia extração dos dados aqui 

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

def run_stockout_results(mes):
    print(f"verificando mes {mes}")
    ### -----------------------> Extraindo Dados CRM
    if mes <= 4:
        # The ID and range of the spreadsheet.
        SAMPLE_SPREADSHEET_ID = os.getenv('SAMPLE_SPREADSHEET_ID')  # Substituído por variável de ambiente
        SAMPLE_RANGE_NAME = 'crm!A:Z'
        sheets_to_csv(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME, 'historico_extra_info.csv')
        df = expand_extra_info_sheets('historico_extra_info.csv', mes)
    else:
        print(f"verificando mes {mes}")
        df = consulta_crm(mes, 'stockout')
        df = expand_extra_info(df)
        print(f"verificando df mes {mes}")
        print(df)

    # O restante do código segue igual...
