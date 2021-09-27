import pandas as pd
from pykospacing import Spacing
from konlpy.tag import Okt
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.normalizer import *


#머신러닝 학습을 위한 공지사항 data 추출
df_csv = pd.read_csv("dataframe.csv",encoding='utf-8')
dataset = df_csv[['Binary','Title','Text']]

#데이터 셋
dataset = dataset.head(1000)

#텍스트 띄어쓰기 전처리
def text_spacing(dataset):
    spacing = Spacing()
    for index, row in dataset.iterrows():
        if(type(row.Text) == type("str")):
            dataset.at[index,'Text'] = spacing(row.Text)
            dataset.at[index,'Title'] = spacing(row.Title)
    return dataset
    
def text_tokenazation(dataset):
    tokenizer = Okt()

    #토큰화 학습 (시간이 오래걸리나 안할시 성능 구려짐)
    
    urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")
    corpus = DoublespaceLineCorpus("2016-10-20.txt")
    i = 0
    for document in corpus:
        if len(document) > 0:
            print(document)
            i = i+1
        if i == 3:
            break

    word_extractor = WordExtractor()
    word_extractor.train(corpus)
    word_score_table = word_extractor.extract()
    
    for index, row in dataset.iterrows():
        if(type(row.Text) == type("str")):
            dataset.at[index, 'Text'] = tokenizer.morphs(row.Text)
            dataset.at[index, 'Title'] = tokenizer.morphs(row.Title)

    return dataset

def text_normalization(dataset):
    for index, row in dataset.iterrows():
        if(type(row.Text) == type("str")):
            dataset.at[index, 'Text'] = emoticon_normalize(row.Text, num_repeats=2)
            dataset.at[index, 'Title'] = repeat_normalize(row.Title, num_repeats=2)
    return dataset

dataset = text_spacing(dataset)
dataset = text_normalization(dataset)
dataset = text_tokenazation(dataset)

dataset.to_csv("dataset.csv", mode='a', header=False, encoding='utf-8-sig')

