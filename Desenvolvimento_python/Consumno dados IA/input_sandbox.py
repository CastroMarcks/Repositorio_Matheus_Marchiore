import pandas as pd
import logging as log
import awswrangler as wr
import boto3


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

session = boto3.Session(region_name=os.getenv('AWS_REGION'))  # Região AWS sensível substituída por variável de ambiente
athena = Athena(session)

def sandbox_inputador(df, nome_da_tabela):
    caminho_no_s3 = f"s3://{os.getenv('S3_BUCKET')}/tables/{nome_da_tabela}/"  # Substituído por variável de ambiente
    nome_do_banco_de_dados = "sandbox_datalake_sales_ops"

    # Salvar o DataFrame no S3 e sobrescrever os dados existentes (se aplicável)
    wr.s3.to_parquet(
        df=df,
        path=caminho_no_s3,
        dataset=True,
        database=nome_do_banco_de_dados,
        table=nome_da_tabela,
        mode="overwrite"  # mode = "append" para adições
    )
