from results_marketmap import run_marketmap_results
from input_sandbox import sandbox_inputador
import pandas as pd
from main_stockout import run_stockout_results
from main_stockout import agrupa_meses
from campanhas_top_skus import run_campanhas_3kus_results
from campanhas_top_skus import agrupa_meses_campanha
import os

if __name__ == "__main__":
    
    # # -------MARKET--MAP-------------------------------------------------------------------.
    print('Começa o script de resultados de Marketmap')
    run_marketmap_results()

    print('Começa o script para inputar dados na tabela de Pedidos de Marketmap')
    df = pd.read_csv("df_Marketmap_final.csv")
    nome_da_tabela = 'oli_pedidos_marketmap'
    sandbox_inputador(df, nome_da_tabela)
    print('Termina o script para inputar dados na tabela de Pedidos de Marketmap')

    print('Começa o script para inputar dados na tabela de Produtos de Marketmap')
    df = pd.read_csv("ProdutosCadastradosMarketMap.csv")
    nome_da_tabela = 'oli_produtos_marketmap'
    sandbox_inputador(df, nome_da_tabela)
    print('Termina o script para inputar dados na tabela de Produtos de Marketmap')
    
    print('Termina o script de resultados de Marketmap')

  # #-------STOCKOUT-------------------------------------------------------------------

    print('Começa o script de resultados de Stockout')

    for mes in range(5, 8):
        try:
            run_stockout_results(mes)
        except:
            continue
    agrupa_meses()

    print('Começa o script para inputar dados na tabela de Pedidos de Stockout')
    df = pd.read_csv("df_incremental_all_year.csv")

    nome_da_tabela = 'oli_pedidos_incrementais'
    sandbox_inputador(df, nome_da_tabela)
    print('Termina o script para inputar dados na tabela de Pedidos de Stockout')

    print('Começa o script para inputar dados na tabela de Estoque de Stockout')
    df = pd.read_csv("df_estoque_all_year.csv")
    nome_da_tabela = 'oli_estoques'
    sandbox_inputador(df, nome_da_tabela)
    print('Termina o script para inputar dados na tabela de Estoque de Stockout')
    
    print('Termina o script de resultados de Stockout')

# #-------CAMPANHA--3-SKUS-------------------------------------------------------------------

    print('Começa o script de resultados de Campanha 3 Skus')
    for mes in range(3, 8):
        try:
            run_campanhas_3kus_results(mes)
        except:
            print("Não rolou")
            continue

    agrupa_meses_campanha()

    print('Começa o script para inputar dados na tabela de Campanha 3 Skus')

    # Processamento do primeiro arquivo CSV
    df = pd.read_csv("df_campanha_top3_all_year.csv")
    print(df)
    nome_da_tabela = 'oli_resultados_campanhas_top3'
    sandbox_inputador(df, nome_da_tabela)

    # Processamento do segundo arquivo CSV
    df = pd.read_csv("df_campanha_pedidos_all_year.csv")
    nome_da_tabela = 'oli_pedidos_campanhas_top3'
    sandbox_inputador(df, nome_da_tabela)

    print('Termina o script para inputar dados na tabela de Campanha 3 Skus')
