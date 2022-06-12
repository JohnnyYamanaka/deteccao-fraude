# %%
#Importando as bibliotecas
from joblib import dump
import pandas as pd

from helpers import data_preparation, evaluate, feature_engineering

#Pre processamento
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Seleção de features
from sklearn.feature_selection import SelectFromModel

# Método para determinar os melhores hiper parâmetros dos modelos
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')
SEED = SEED = 175834773

# %%
# Importando o dataset
df = pd.read_parquet('../deteccao-fraude/data/financial-data.parquet')

# %%
#Preparando o dataset para o modelo...
data_prep = data_preparation.DataTransformation(df)
df = data_prep.transform_data()

fe = feature_engineering.FeatureEngineering(df)
fe.transform_dataset()
df

# %%
# Separando as veriáveis catégoricas e numéricas
cat_features = df.select_dtypes(include='object').copy().columns
cat_features = cat_features.append(
    pd.Index(['higherValue', 'higherOldBalance', 'withdrawAll']))

numeric_features = df.select_dtypes(include='number').copy().columns
numeric_features = numeric_features.delete([-3, -4, -5, -7])

# %%
onehot = OneHotEncoder()
standard = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
    ('one', onehot, cat_features),
    ('standard', standard, numeric_features)
])

# %%
# Treinando Regressão Logística...
rl_clf = LogisticRegression()

params_lr = {
    'penalty' : ('l1', 'l2', 'elasticnet'),
    'C' : [0.001, 0.01, 1, 10, 100, 1000]
}

random_search_lr = RandomizedSearchCV(
    estimator = rl_clf, 
    param_distributions=params_lr, 
    scoring='roc_auc',
    cv=5, 
    verbose=0, 
    random_state=SEED
)

pipe_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', random_search_lr)
])

eval_rl = evaluate.EvaluateModel(model=pipe_lr, dataset=df)
model_lr = eval_rl.run_model(SEED)

# %%
random_search_lr.best_params_

# %%
# Treinando XGBoost...
xgb_clf = XGBClassifier(verbosity=0)

params_xgb = {
    'max_depth': [2, 4, 6],
    'max_leaves' : [2, 4, 6, 8],
}

random_search_xgb = RandomizedSearchCV(
    estimator = xgb_clf,
    param_distributions = params_xgb,
    scoring='roc_auc',
    cv = 5,
    verbose = 0,
    random_state=SEED
)

pipe_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', random_search_xgb)
])

eval_xgb = evaluate.EvaluateModel(model=pipe_xgb, dataset=df)
model_xgb = eval_xgb.run_model(SEED)

# %%
random_search_xgb.best_params_

# %%
# Treinando Random Forest...
rf_clf = RandomForestClassifier(verbose=0, n_jobs=-1)

params_rf = {
    'n_estimators': [50, 100, 150],
    'criterion' : ('gini', 'entropy'),
    'max_depth' : [2, 4, 6],
    'min_samples_leaf' : [2, 4, 10, 20]
}

random_search_rf = RandomizedSearchCV(
    estimator = rf_clf,
    param_distributions = params_rf,
    scoring='roc_auc',
    cv = 5,
    verbose = 0,
    random_state=SEED   
)

pipe_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', random_search_rf)
])

eval_rf = evaluate.EvaluateModel(model=pipe_rf, dataset=df)
model_rf = eval_rf.run_model(SEED)

# %%
random_search_rf.best_params_

# %%
# Todos os modelos apresetaram o mesmo resultado... 
# Isso pode indicar que chegamos ao limite de melhorias que pode ser 
# realizada com o conjunto que temos.
# Portanto, vamos selecionar o modelo mais simples como o final.

X = df.drop('isFraud', axis=1).copy()
y = df['isFraud']


pipe_final = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(**random_search_rf.best_params_))
])

pipe_final.fit(X, y)


# %%
# Exportando o modelo e as features.
dump(pipe_final, '../deteccao-fraude/model/model.joblib')

# %%
