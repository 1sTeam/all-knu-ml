import pandas as pd

df_csv = pd.read_csv("dataframe.csv",encoding='utf-8')

print(df_csv.keys())