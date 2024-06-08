# %%
import pandas as pd


import sqlalchemy
import sqlite3


from sklearn import model_selection
from feature_engine import imputation

from feature_engine import encoding


from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model


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
# REGEX para remover [^A-Za-z0-9_]+
X_train.columns = X_train.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '', regex=True)

# %%
# MODELS
rf_clf = ensemble.RandomForestClassifier(n_estimators=200, 
                                         min_samples_leaf=50, 
                                         n_jobs=-1, 
                                         random_state=42)

ada_clf = ensemble.AdaBoostClassifier(n_estimators=200, 
                                      learning_rate=0.8, 
                                      random_state=42)

dt_clf = tree.DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=50,
                                     random_state=42)

rl_clf = linear_model.LogisticRegressionCV(cv=4, 
                                           n_jobs=-1)
# %%
# PIPELINE COM GRIDSEARCH

param_grid = {'n_estimators':[50,100,200,250],
                                    'min_samples_leaf':[5,10,50,100]}

grid_search = model_selection.GridSearchCV(rf_clf, 
                             param_grid, 
                             n_jobs=1, 
                             cv=4, 
                             scoring='roc_auc',
                             verbose=3,
                             refit=True)

#{'min_samples_leaf': 5, 'n_estimators': 250}

pipeline_rf = pipeline.Pipeline(steps=[("imput 0", imput_0),
                                      ("imput -1", imput_1),
                                      ("onehot", onehot),
                                      ("Modelo", grid_search)])

# %%

def train_test_report(modelo, X_train, y_train, X_test, y_test, key_metric, is_prob=True):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)
    
    metric_result = key_metric(y_test, proba[:,1]) if is_prob else key_metric(y_test, pred)
    
    return metric_result
    

# %%

pipeline_rf.fit(X_train, y_train)

# %%
pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")

# %%
# METRICAS
y_train_pred = pipeline_rf.predict(X_train)
y_train_prob = pipeline_rf.predict_proba(X_train)

# %%
acc_train = round(100 * metrics.accuracy_score(y_train, y_train_pred), 2)
roc_train = round(100 * metrics.roc_auc_score(y_train, y_train_prob[:,1]), 2)

print("acc_train", acc_train)
print("roc_train", roc_train)

# %%
print("Baseline", round(1 - y_train.mean(), 2)*100)
print("Acuraria", acc_train)

# %%
# CURVA ROC
skplt.metrics.plot_roc(y_train, y_train_prob)
plt.show()
# %%
# KS
skplt.metrics.plot_ks_statistic(y_train, y_train_prob)
plt.show()
# Conseguindo separar muito bem os dados
# %%
# PRECISON RECALL
skplt.metrics.plot_precision_recall(y_train, y_train_prob)
plt.show()
# %%
# Os 100 primeiros tem 14x mais chance de assinar
skplt.metrics.plot_lift_curve(y_train, y_train_prob)
plt.show()
# %%
# TESTE
y_test_pred = pipeline_rf.predict(X_test)
y_test_prob = pipeline_rf.predict_proba(X_test)

acc_test = round(100 * metrics.accuracy_score(y_test, y_test_pred), 2)
roc_test = round(100 * metrics.roc_auc_score(y_test, y_test_prob[:,1]), 2)

print("acc_train", acc_test)
print("roc_train", roc_test)

# %%
print("Baseline", round(1 - y_test.mean(), 2)*100)
print("Acuraria", acc_test)

# %%
# CURVA ROC
skplt.metrics.plot_roc(y_test, y_test_prob)
plt.show()
# %%
# KS
skplt.metrics.plot_ks_statistic(y_test, y_test_prob)
plt.show()
# Conseguindo separar muito bem os dados
# %%
# PRECISON RECALL
skplt.metrics.plot_precision_recall(y_test, y_test_prob)
plt.show()
# %%
# Os 100 primeiros tem 14x mais chance de assinar
skplt.metrics.plot_lift_curve(y_test, y_test_prob)
plt.show()
# %%
skplt.metrics.plot_cumulative_gain(y_test,y_test_prob)
plt.show()
# %%
### Testando na OOT para ver se descolou
X_oot, y_oot = dt_oot[features], dt_oot[target]

y_prob_oot = pipeline_rf.predict_proba(X_oot)

roc_oot = round(100 * metrics.roc_auc_score(y_oot, y_prob_oot[:,1]), 2)

print("roc_train", roc_oot)

skplt.metrics.plot_lift_curve(y_oot, y_prob_oot)
plt.show()
# %%
skplt.metrics.plot_cumulative_gain(y_oot,y_prob_oot)
plt.show()