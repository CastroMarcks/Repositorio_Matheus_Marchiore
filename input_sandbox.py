import pandas as pd
import pandas as pd
import logging as log
import awswrangler as wr
import boto3
import pandas as pd


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

df = pd.read_csv("df_campanha_all_year_seller_id.csv") ##****

## Nossas tabelas na Sandbox----------------------
#select * from sandbox_datalake_sales_ops.oli_pedidos_incrementais (df_incremental_all_year.csv)
#select * from sandbox_datalake_sales_ops.oli_estoques (df_estoque_all_year.csv)
#select * from sandbox_datalake_sales_ops.oli_resultados_campanhas_top3 (df_campanha_top3_all_year.csv)
#select * from sandbox_datalake_sales_ops.oli_resultados_campanhas_sellers_id (df_campanha_all_year_seller_id.csv)


def sandbox_inputador (df):
    caminho_no_s3 = "s3://dataplat-sandbox-datalake-sales-ops/tables/oli_resultados_campanhas_sellers_id/" ##****
    nome_do_banco_de_dados = "sandbox_datalake_sales_ops"
    nome_da_tabela = "oli_resultados_campanhas_sellers_id" ##****

    # Salvar o DataFrame no S3 e sobrescrever os dados existentes (se aplicável)
    wr.s3.to_parquet(
        df=df,
        path=caminho_no_s3,
        dataset=True,
        database=nome_do_banco_de_dados,
        table=nome_da_tabela,
        mode="overwrite"  
        #mode = "append"
    )

df_final = sandbox_inputador(df)
print('Dados criados no sandbox com sucesso')
print("FIM")