# %%
import argparse
import pandas as pd
import sqlalchemy
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("--date", '-d', default="max")
args = parser.parse_args()

print("Importando modelo")
model = pd.read_pickle("../../../models/modelo_sub.pkl")
print("OK")

print("Importando query...")
with open("../etl/query.sql", "r") as open_file:
    query = open_file.read()
print("OK")

print("Obtendo data para escoragem...")
engine = sqlalchemy.create_engine("sqlite:///../../../data/gc.db")
conn = sqlite3.connect("../../../data/gc.db")

if args.date == "max":
    date = pd.read_sql("SELECT MAX(dtRef) as date FROM tb_book_players", conn)["date"][0]
else:
    date = args.date
    
query = query.format(date=date)
print("OK")

print("Importando dados...")
df = pd.read_sql_query(query, conn)
print("OK.")

print("Realizando o score dos dados")
df_score = df[["dtRef", "idPlayer"]].copy()
df_score["score"] = model["model"].predict_proba(df[model["feature"]])[:,1]
df_score["descModel"] = "Model Subscription"
print("OK")

print("Enviando dados para o DB...")
conn.execute(f"DELETE FROM tb_model_score WHERE dtRef='{date}'")
df_score.to_sql("tb_model_score", conn, if_exists="append", index=False)
print("OK.")
