import pandas as pd

df_csv = pd.read_csv("dataframe.csv",encoding='utf-8')

#머신러닝 학습을 위한 공지사항 data 추출
dataset = df_csv[['Binary','Title','Text']]

#한 행씩 추출
for index, row in dataset.iterrows():
    print(row.Text)



#print(dataset.head())