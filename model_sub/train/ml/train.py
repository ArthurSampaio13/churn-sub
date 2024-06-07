# %%
import pandas as pd

import sqlalchemy
import sqlite3


from sklearn import model_selection
from feature_engine import imputation
from feature_engine import encoding
from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics

import scikitplot as skplt
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# %%
# SAMPLE
engine = sqlalchemy.create_engine("sqlite:///../../../data/gc.db")
conn = sqlite3.connect("../../../data/gc.db")
query = "SELECT * FROM tb_abt_sub"

df = pd.read_sql_query(query, conn)
# %%

#Out of time - Back Test
dt_oot = df[df['dtRef'].isin(['2022-01-01','2021-12-31'])].copy()
# %%

df_train = df[~df['dtRef'].isin(['2022-01-01','2021-12-31'])].copy()
df_train
# %%

features = df_train.columns.to_list()[2:-1]
target = 'flagSub'

df_train
# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_train[features], df_train[target], test_size=0.2, random_state=42)
# %%
#EXPLORE

cat_features = X_train.dtypes[X_train.dtypes=='object'].index.tolist()
num_features = list(set(X_train.columns) - set(cat_features))
# %%
# Missing numericos
is_na = X_train[num_features].isna().sum()
is_na[is_na > 0]
# %%

missing_0 = ['avgKDA']
missing_1 = ['winRateDust2', 'vlIdade', 'winRateOverpass', 'winRateInferno', 'winRateNuke', 'winRateVertigo', 'winRateAncient', 'winRateMirage', 'winRateTrain']

# %%
# Missing categoricos
is_na = X_train[cat_features].isna().sum()
is_na
# %%
# MODIFY
# Input missing_1
imput_1 = imputation.ArbitraryNumberImputer(arbitrary_number=-1, variables=missing_1)
# Input missing_0
imput_0 = imputation.ArbitraryNumberImputer(arbitrary_number=0, variables=missing_0)

# %%
# OneHotEncoding
onehot = encoding.OneHotEncoder(drop_last=True, variables=cat_features)
# %%
# MODEL
model = ensemble.RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1)

# %%
# PIPELINE
model_pipe = pipeline.Pipeline(steps=[("imput 0", imput_0),
                                      ("imput -1", imput_1),
                                      ("onehot", onehot),
                                      ("Modelo", model)])
# %%
# REGEX para remover [^A-Za-z0-9_]+
X_train.columns = X_train.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '', regex=True)
# %%
model_pipe.fit(X_train, y_train)
# %%
# METRICAS
y_train_pred = model_pipe.predict(X_train)
y_train_prob = model_pipe.predict_proba(X_train)

# %%
acc_train = round(100 * metrics.accuracy_score(y_train, y_train_pred), 2)
roc_train = metrics.roc_auc_score(y_train, y_train_prob[:,1])

print("acc_train", acc_train)
print("roc_train", roc_train)

# %%
print("Baseline", round(1 - y_train.mean(), 2)*100)
print("Acuraria", acc_train)

# %%
skplt.metrics.plot_roc(y_train, y_train_prob)
plt.show()
# %%
