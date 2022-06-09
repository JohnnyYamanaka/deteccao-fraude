# %% 
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.data_preparation import DataTransformation

warnings.filterwarnings('ignore')

# %%
df = pd.read_parquet('../deteccao-fraude/data/financial-data.parquet')
df.head()

# %%
data_prep = DataTransformation(df)
df = data_prep.transform_data()

# %%
df

# %%
# Ideia numero 1: tentar agrupar as horas em períodos de 3 horas cada
df['periodOfDay'] = df['hour'].apply(lambda x: x % 8)
df

# %%
# Reorganizando as colunas
columns_order = list(df.columns)
columns_order.insert(0, columns_order.pop())

df = df.reindex(columns = columns_order)
df.info()

# %%
# Verificando correlação das featuers com a variável alvo
df.corr()['isFraud'][:-1]

# %%
# Ideia numero 2: separar os maiores montantes para tentar prever com maior
# precisão montantes maiores

df['amount'].describe()
third_quartile = df['amount'].quantile(0.75)
third_quartile

# %%
plt.figure(figsize=(12, 6))
plt.title('Amount (K $)', fontsize=18)
sns.histplot(data=df, x=df['amount'] / 1000)
plt.vlines(third_quartile / 1000, 0, 11000, color='r', linestyles ="dotted")
plt.legend(['Third quantile (original dataset)', 'fraud amount'])

# %%
df_fraud = df.query('isFraud == 1').copy()
df_not_fraud = df.query('isFraud == 0').copy()

# %%
plt.figure(figsize=(12, 6))
plt.title('Amount in fraudulent transaction', fontsize=18)
sns.histplot(data=df_fraud, x=df_fraud['amount'] / 1000)
plt.vlines(third_quartile / 1000, 0, 650, color='r', linestyles ="dotted")
plt.legend(['Third quantile (original dataset)', 'fraud amount'])

plt.show()
# %%
plt.figure(figsize=(12, 6))
plt.title('Amount in not fraudulent transaction', fontsize=18)
sns.histplot(data=df_not_fraud, x=df_not_fraud['amount'] / 1000)
plt.vlines(third_quartile / 1000, 0, 11000, color='r', linestyles ="dotted")
plt.legend(['Third quantile (original dataset)', 'fraud amount'])


plt.show()

# %%
df_not_fraud['amount'].describe()

# %%
df_fraud['amount'].describe()

# %%
# Transações fraudulentas aperentam ser os que movimentam maiores montantes.
# Portanto, faz sentido criar uma feature para determinar se o montante é alto
# ou não. Para esse teste, vamos utilizar o terceiro quartil (75%)


# %%
df['higherValue'] = df['amount'].apply(lambda x: 1 if x >= third_quartile else 0)

columns_order = list(df.columns)
columns_order.insert(0, columns_order.pop())

df = df.reindex(columns = columns_order)
df
# %%
df['higherValue'].value_counts(normalize=True)

# %%
# Ideia numero 3: transações onde o valor é exatamente igual ao saldo antigo da
# conta de origem pode indicar fraude

total_fraud = df['isFraud'].sum()
zero_transaction = (df['amount'] == df['oldbalanceOrg']).sum()

print(total_fraud)
print(zero_transaction)

print(zero_transaction / total_fraud)

# %%
# Existe uma alta correlação entre essas duas colunas, logo vale a pena 
# criar essa nova feature

conditions = [
    (df['amount'] == df['oldbalanceOrg']),
    (df['amount'] != df['oldbalanceOrg'])
]

values =  [1, 0]

df['withdrawAll'] = np.select(conditions, values)

columns_order = list(df.columns)
columns_order.insert(0, columns_order.pop())

df = df.reindex(columns = columns_order)

# %%
df['withdrawAll'].sum()

# %%
df.corr()['isFraud']
# %%
df.head()

# %%
# Ideia numero 4: Criar um booleano para indicar que o valor do 
# oldbalance está acima do terceiro percentil também.

df['oldbalanceOrg'].describe()

# %%
old_balance_org_quantile = df['oldbalanceOrg'].quantile(0.75)

df['oldbalanceOrgQuantile'] = df['oldbalanceOrg'].\
    apply(lambda x: 1 if x >= old_balance_org_quantile else 0)


columns_order = list(df.columns)
columns_order.insert(0, columns_order.pop())

df = df.reindex(columns = columns_order)

# %%
df

# %%
# # Ideia numero 5: Criar feature com o amount ao quadrado
df['amountSquare'] = np.power(df['amount'], 2)

columns_order = list(df.columns)
columns_order.insert(0, columns_order.pop())

df = df.reindex(columns = columns_order)

# %%
# # Ideia numero 6: Criar feature com a diferença entre a média e o amount
df['amountDiff'] = np.mean(df['amount']) - df['amount']

columns_order = list(df.columns)
columns_order.insert(0, columns_order.pop())

df = df.reindex(columns = columns_order)

# %%
df.corr()['isFraud'][:-1].sort_values()

# %%
# Fim da Investigação
