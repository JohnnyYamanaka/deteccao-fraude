import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

# Pegar a chave para autenticação na API do Kaggle
file = open("../deteccao-fraude/kaggle.json")
json_file = json.load(file)

os.environ["KAGGLE_USERNAME"] = json_file["username"]
os.environ["KAGGLE_KEY"] = json_file["key"]

dataset = "ealaxi/paysim1"
path = "../deteccao-fraude/data"

api = KaggleApi()
api.authenticate()


def select_sample():
    """
    Mostra o tamanho do dataset original e oferece a opção de escolher
    o tamanho do dataset para trabalhar
    """
    if os.path.exists("../deteccao-fraude/data/financial-data.parquet"):
        os.remove(path="../deteccao-fraude/data/financial-data.parquet")

    api.dataset_download_files(dataset, path, force=True, quiet=False, unzip=True)

    df_raw = pd.read_csv("../deteccao-fraude/data/PS_20174392719_1491204439457_log.csv")

    os.remove(path="../deteccao-fraude/data/PS_20174392719_1491204439457_log.csv")

    len_df = df_raw.shape[0]

    print(f"The original dataset have {len_df: ,} rows.")
    qtd = int(input("How many rows do you want to work with?: "))

    sampled_df = df_raw.iloc[:qtd, :]
    sampled_df.to_parquet("../deteccao-fraude/data/financial-data.parquet")

    df_raw = df_raw.iloc[qtd:, :]
    df_raw.to_parquet("../deteccao-fraude/data/raw-data.parquet")

    print("Amostras selecionadas e salvas")


select_sample()
