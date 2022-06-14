import joblib
import streamlit as st
import pandas as pd

import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

from helpers.data_preparation import DataTransformation
from helpers.feature_engineering import FeatureEngineering
from helpers.evaluate import EvaluateModel

st.title('Detecção de Fraudes')
st.subheader('Bem vindo ao protótipo do detector de fraudes.')
st.write('Para começar, escolha a data das transações desejada.')
st.write('A primeira execução pode vai demorar mais que os outros para \
    carregar o dataset.')

@st.experimental_memo
def load_dataset():
    df = pd.read_parquet('../deteccao-fraude/data/raw-data.parquet')

    return df

def select_transaction_day(day: int):
    df = load_dataset()
    df['day'] = df['step'].apply(lambda x: x // 24) + 1
    df = df.query('day == @day')
    df.drop('day', axis=1, inplace = True)

    return df


model = joblib.load('../deteccao-fraude/model/model.joblib')

day = st.slider(label='Data', min_value=6, max_value=31, 
        help='Os cinco primeiros dias foram retirados porque o \
            modelo foi treinado com esses dados.')

if st.button(label='Prever'):
    df = select_transaction_day(day)

    if df.query('isFraud == 1')['isFraud'].count() != df.shape[0]:
        data_prep = DataTransformation(df)
        df = data_prep.transform_data()
        fe = FeatureEngineering(df)
        df = fe.transform_dataset()

        evaluate = EvaluateModel(model=model, dataset=df)
        roc, df_evaluate, total_loss = evaluate.predict()

        st.write(f'AUC Score: {roc: .4f}')
        st.write(f'Total of loss: $ {total_loss: ,.2f}')

        st.write('O modelo errou em classificar as seguintes transações: ')
        fraud = df_evaluate.query('financialLost < 0')
        st.dataframe(fraud.head())

    else:
        st.error(
            'Data escolhida tem somente uma classe, por favor escolher outra')
