import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

# Pegar a chave para autenticação na API do Kaggle
file = open('./deteccao-fraude/kaggle.json')
json_file = json.load(file)

os.environ['KAGGLE_USERNAME'] = json_file['username']
os.environ['KAGGLE_KEY'] = json_file['key']


def select_sample():
    """
        Mostra o tamanho do dataset original e oferece a opção de escolher
        o tamanho do dataset para trabalhar    
    """
    
    len_df = df.shape[0]

    print(f'The original dataset have {len_df: ,} rows.')
    qtd = int(input('How many rows do you want to work with?: '))

    sampled_df = df.sample(n=qtd, random_state=9453421)
    sampled_df.to_parquet('./data/financial-data.parquet')

    print('Amostras selecionadas e salvas')


dataset = 'ealaxi/paysim1'
path = 'data'

api = KaggleApi()
api.authenticate()

if os.path.exists('./data/financial-data.parquet'):
    os.remove(path='./data/financial-data.parquet')

api.dataset_download_files(dataset, path, force = True, quiet = False, unzip = True)
df = pd.read_csv('./data/PS_20174392719_1491204439457_log.csv')

os.remove(path='./data/PS_20174392719_1491204439457_log.csv')
select_sample()
