# %%
import warnings
import pandas as pd

from helpers.data_preparation import DataTransformation
from helpers.evaluate import EvaluateModel
from IPython.display import display

# Pre processamento
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Modelos
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

SEED = 175834773
warnings.filterwarnings("ignore")

# %%
df = pd.read_parquet("../deteccao-fraude/data/financial-data.parquet")

# %%
data_prep = DataTransformation(df)
df = data_prep.transform_data()

# %%
display(df)

# %%
cat_features = df.select_dtypes(include="object").copy().columns
numeric_features = df.select_dtypes(include="number").copy().columns

numeric_features = numeric_features.delete(-1)

# %%
onehot = OneHotEncoder()
standard = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("one", onehot, cat_features),
        ("standard", standard, numeric_features),
    ]
)


# %%
print("\nrodando dummy classifier...")
pipe_dummy = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", DummyClassifier())]
)

evaluate_dummy = EvaluateModel(pipe_dummy, df)
evaluate_dummy.run_model(SEED)


# %%
print("\ntestando Logistic Regression...")
pipe_lr = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", LogisticRegression())]
)

evaluate_lr = EvaluateModel(pipe_lr, df)
evaluate_lr.run_model(SEED)

# %%
print("\ntestando Random Forest...")
pipe_rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
    ]
)

evaluate_rf = EvaluateModel(pipe_rf, df)
evaluate_rf.run_model(SEED)

# %%
print("\ntestando XGBoost...")
pipe_xb = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(random_state=SEED, verbosity=0)),
    ]
)

evaluate_xgb = EvaluateModel(pipe_xb, df)
evaluate_xgb.run_model(SEED)

# %%
print("\ntestando LightGBM...")
pipe_lgb = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMClassifier(random_state=SEED, verbosity=0)),
    ]
)

evaluate_lgb = EvaluateModel(pipe_lgb, df)
evaluate_lgb.run_model(SEED)
