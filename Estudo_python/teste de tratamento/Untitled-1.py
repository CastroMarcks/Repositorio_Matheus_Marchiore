# %%
import pandas as pd 

# %%
df = pd.read_csv('Planilha.csv')
df

# %%
#aqui quero extrairo ojason que tem na coluna extra_info

# %%
import pandas as pd
import json

# Carregando os dados do arquivo CSV
df = pd.read_csv('Planilha.csv')

# Função para converter JSON em dicionário
def parse_json(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}  # Retorna um dicionário vazio em caso de erro

# Aplicar a função na coluna 'extra_info' para extrair o JSON
df['extra_info'] = df['extra_info'].apply(parse_json)

# Normalizar o conteúdo JSON em colunas separadas
json_cols = df['extra_info'].apply(pd.Series)

# Unir as novas colunas ao DataFrame original
df = pd.concat([df, json_cols], axis=1).drop('extra_info', axis=1)

# Preencher todos os valores nulos com espaço em branco
df.fillna(' ', inplace=True)

# Encontrar colunas que terminam com números e mesclar os valores
import re

# Extrair sufixos numéricos das colunas e organizar em um dicionário
sufixos = {}
for col in df.columns:
    match = re.search(r'(\d+)$', col)
    if match:
        num = match.group(1)
        if num not in sufixos:
            sufixos[num] = []
        sufixos[num].append(col)

# Para cada grupo de sufixo, mesclar as colunas correspondentes
for num, columns in sufixos.items():
    df['merged_' + num] = df[columns].apply(lambda x: ';'.join(x.astype(str)), axis=1)
    df.drop(columns, axis=1, inplace=True)  # Remover as colunas originais se necessário

# Mostrar o DataFrame atualizado
df


# %% [markdown]
# 

# %%
# Selecionando as colunas fixas
fixed_columns = df.columns.difference(['merged_1', 'merged_2', 'merged_3'])

# Pivotando as colunas 'Mesclado1', 'Mesclado2' e 'Mesclado3' mantendo as outras colunas fixas
pivot_df = df.melt(id_vars=fixed_columns, value_vars=['merged_1', 'merged_2', 'merged_3'], var_name='Tipo_Mesclado', value_name='Mesclado')
df = pivot_df

# %%
import pandas as pd

def pivot_merged_columns(df):
    # Encontrar todas as colunas 'merged_'
    merged_cols = [col for col in df.columns if col.startswith('merged_')]
    
    # Criar um novo DataFrame para receber os dados pivoteados
    # Considerando que há uma coluna de identificação que não será pivotada
    id_vars = [col for col in df.columns if col not in merged_cols]
    df_long = pd.melt(df, id_vars=id_vars, value_vars=merged_cols, var_name='merged_type', value_name='merged_value')
    
    # Pivotar os dados para transformar valores de 'merged_type' em colunas distintas
    df_pivot = df_long.pivot_table(index=id_vars, columns='merged_type', values='merged_value', aggfunc='first').reset_index()
    
    # Planar as colunas do índice multi-nível resultante
    df_pivot.columns = [f'{lvl1}' if not lvl0 or lvl0 == 'index' else f'{lvl0}' for lvl0, lvl1 in df_pivot.columns]
    
    return df_pivot

# Usar a função no DataFrame
df_pivoted = pivot_merged_columns(df)
df_pivoted

# %%
id_vars = [col for col in df.columns if not col.startswith('merged_')]
df_long = pd.melt(df, id_vars=id_vars, var_name='merged_column', value_name='merged_values')

# Mostrar o DataFrame reorganizado
df_long

# %% [markdown]
# 

# %%
import pandas as pd
import json

# Carregando os dados do arquivo CSV
df = pd.read_csv('Planilha.csv')

# Função para converter JSON em dicionário, tratando possíveis valores nulos ou não-JSON
def parse_json(data):
    if pd.isna(data) or not data.strip():
        return {}  # Retorna um dicionário vazio para dados nulos ou vazios
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}  # Retorna um dicionário vazio em caso de erro de decodificação

# Aplicar a função na coluna 'extra_info' para extrair o JSON
df['extra_info'] = df['extra_info'].apply(parse_json)

# Normalizar o conteúdo JSON em colunas separadas
json_cols = df['extra_info'].apply(lambda x: pd.Series(x))

# Unir as novas colunas ao DataFrame original
df = pd.concat([df, json_cols], axis=1).drop('extra_info', axis=1)

# Preencher todos os valores nulos com espaço em branco
df.fillna(' ', inplace=True)

# Mesclar colunas que terminam com o mesmo número
import re

sufixos = {}
for col in df.columns:
    match = re.search(r'(\d+)$', col)
    if match:
        num = match.group(1)
        if num not in sufixos:
            sufixos[num] = []
        sufixos[num].append(col)

for num, columns in sufixos.items():
    df['merged_' + num] = df[columns].apply(lambda x: ';'.join(x.astype(str)), axis=1)
    df.drop(columns, axis=1, inplace=True)

# Identificar colunas que não são de mesclagem para usá-las como identificadores
id_vars = [col for col in df.columns if not col.startswith('merged_')]

# Transformar colunas mescladas em linhas
df_long = pd.melt(df, id_vars=id_vars, var_name='merged_column', value_name='merged_values')

# Mostrar o DataFrame reorganizado
df_long


