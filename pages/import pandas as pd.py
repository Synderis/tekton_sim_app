import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, MetaData, engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL

connection_url = engine.URL.create(
    r"mssql+pyodbc",
    host=r"DESKTOP-J8L86O2",
    database=r"tekton_sim_data",
    query={
        r"driver": "ODBC Driver 17 for SQL Server",
        r"LongAsMax": "Yes",
    },
)

db_engine = create_engine(connection_url, echo=False)

connection_string = r"Driver={ODBC Driver 17 for SQL Server}; Server=DESKTOP-J8L86O2; Database=tekton_sim_data; Trusted_Connection=yes;"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
engine = create_engine(connection_url)
conn = engine.connect()
data = conn.execute(
    text("""select tick_times, hp_after_pre_anvil from tekton_results""")
)
