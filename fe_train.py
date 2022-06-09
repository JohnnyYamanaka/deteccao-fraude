# %% 
import pandas as pd
import numpy as np

from helpers.data_preparation import DataTransformation
from helpers.evaluate import EvaluateModel
from helpers.feature_engineering import FeatureEngineering

#Pre processamento
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
SEED = SEED = 175834773


# %%
df = pd.read_parquet('../deteccao-fraude/data/financial-data.parquet')
df.head()

# %%
data_prep = DataTransformation(df)
df = data_prep.transform_data()

# %%
feature_engineering = FeatureEngineering(df)
df = feature_engineering.transform_dataset()

# %%
df

# %%
cat_features = df.select_dtypes(include='object').copy().columns
cat_features.append(pd.Index(['higherValue', 'higherOldBalance', 
                                                    'withdrawAll']))

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
print('\ntestando Logistic Regression...')
pipe_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

evaluate_lr = EvaluateModel(pipe_lr, df)
evaluate_lr.run_model(SEED)

# %%
print('\ntestando Random Forest...')
pipe_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=SEED, n_jobs=-1))
])

evaluate_rf = EvaluateModel(pipe_rf, df)
evaluate_rf.run_model(SEED)

# %%
print('\ntestando XGBoost...')
pipe_xb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(random_state=SEED, verbosity=0))
])

evaluate_xgb = EvaluateModel(pipe_xb, df)
evaluate_xgb.run_model(SEED)

# %%
