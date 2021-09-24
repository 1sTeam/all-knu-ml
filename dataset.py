import pandas as pd
from pykospacing import Spacing

#머신러닝 학습을 위한 공지사항 data 추출
df_csv = pd.read_csv("dataframe.csv",encoding='utf-8')
dataset = df_csv[['Binary','Title','Text']]

#테스트 데이터
test_dataset = dataset.head(5)

#텍스트 띄어쓰기 전처리
def text_spacing(dataset):
    spacing = Spacing()
    for index, row in dataset.iterrows():
        if(type(row.Text) == type("str")):
            dataset.iloc[index,2] = spacing(row.Text)
    return dataset
    


test_dataset = text_spacing(test_dataset)


