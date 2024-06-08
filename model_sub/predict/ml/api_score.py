import pandas as pd
import sqlalchemy
import sqlite3

with open("../etl/query_id.sql", "r") as open_file:
    query = open_file.read()

engine = sqlalchemy.create_engine("sqlite:///../../../data/gc.db")
conn = sqlite3.connect("../../../data/gc.db")

model = pd.read_pickle("../../../models/modelo_sub.pkl")

def score(id_Player):
    df = pd.read_sql(query.format(id_Player=id_Player), conn)
    score = model['model'].predict_proba(df[model['feature']])[:,1][0]
    return score