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

def report_model(X, y, model, metric, is_prob=True):
    if is_prob:
        y_pred = model.predict_proba(X)[:,1]
    else:
        y_pred = metric.predict(X)
    res = metric(y, y_pred)
    return res

# %%
# SAMPLE
print("Importando ABT...")
engine = sqlalchemy.create_engine("sqlite:///../../../data/gc.db")
conn = sqlite3.connect("../../../data/gc.db")
query = "SELECT * FROM tb_abt_sub"
print("OK.")

df = pd.read_sql_query(query, conn)
# %%
print("Separando em treinamento e back test...")

#Out of time - Back Test
dt_oot = df[df['dtRef'].isin(['2022-01-01','2021-12-31'])].copy()
# %%

df_train = df[~df['dtRef'].isin(['2022-01-01','2021-12-31'])].copy()
print("OK.")
# %%

features = df_train.columns.to_list()[2:-1]
target = 'flagSub'

df_train
# %%
print("Separando em treinamento e teste....")
X_train, X_test, y_train, y_test = model_selection.train_test_split(df_train[features], df_train[target], test_size=0.2, random_state=42)
print("Ok")
# %%
#EXPLORE

cat_features = X_train.dtypes[X_train.dtypes=='object'].index.tolist()
num_features = list(set(X_train.columns) - set(cat_features))
# %%.
# Missing numericos
print("Estatistica de missing...")
is_na = X_train[num_features].isna().sum()
is_na[is_na > 0]
print("OK.")
# %%

missing_0 = ['avgKDA']
missing_1 = ['winRateDust2', 'vlIdade', 'winRateOverpass', 'winRateInferno', 'winRateNuke', 'winRateVertigo', 'winRateAncient', 'winRateMirage', 'winRateTrain']

# %%
# Missing categoricos
print("Estatistica de missing...")
is_na = X_train[cat_features].isna().sum()
is_na
print("OK.")
# %%
# MODIFY
# Input missing_1
print("Construindo pipeline de ML...")
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
print("OK.")
# %%
# PIPELINE COM GRIDSEARCH

param_grid = {'n_estimators':[200, 250],
                                    'min_samples_leaf':[5, 10]}

grid_search = model_selection.GridSearchCV(rf_clf, 
                             param_grid, 
                             n_jobs=1, 
                             cv=4, 
                             scoring='roc_auc',
                             verbose=0,
                             refit=True)

pipeline_rf = pipeline.Pipeline(steps=[("imput 0", imput_0),
                                      ("imput -1", imput_1),
                                      ("onehot", onehot),
                                      ("Modelo", grid_search)])

# %%
print("Encontrando o melhor modelo com grid...")
pipeline_rf.fit(X_train, y_train)
print("OK")

# %%


# %%
print("Analisando metricas...")
auc_train = report_model(X_train, y_train, pipeline_rf, metrics.roc_auc_score)
auc_test = report_model(X_test, y_test, pipeline_rf, metrics.roc_auc_score)
auc_oot = report_model(dt_oot[features], dt_oot[target], pipeline_rf, metrics.roc_auc_score)

print("auc_train:",auc_train)
print("auc_test:",auc_test)
print("auc_oot:",auc_oot)


print("OK.") 
# %%
print("Ajustar o modelo para a base toda...")

pipeline_model = pipeline.Pipeline(steps=[("imput 0", imput_0),
                                      ("imput -1", imput_1),
                                      ("onehot", onehot),
                                      ("Modelo", grid_search.best_estimator_)])
pipeline_model.fit(df[features], df[target])
print("OK")
# %%
print("Feature importance by model...")
features_transformed = pipeline_model[:-1].transform(df[features]).columns.tolist()
feature_importance = pd.DataFrame(pipeline_model[-1].feature_importances_, index=features_transformed).sort_values(ascending=False, by=0)
feature_importance
print("Ok.")

# %%
series_model = pd.Series(
    {'model': pipeline_model,
    'feature':features,
    "auc_train":auc_train,
    "auc_test":auc_test,
    "auc_oot": auc_oot}
)

series_model.to_pickle("../../../models/modelo_sub.pkl")