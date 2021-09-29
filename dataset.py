import pandas as pd
import numpy as np
from pykospacing import Spacing
from konlpy.tag import Komoran
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.normalizer import *

#머신러닝 학습을 위한 공지사항 data 추출
df_csv = pd.read_csv("dataframe.csv",encoding='utf-8')
dataset = df_csv[['Binary','Title']]

#데이터 셋
dataset = dataset.head(1000)

def text_normalization(dataset):
    dataset = dataset.dropna(how = 'any')
    dataset['Title'] = dataset['Title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","")
    dataset['Title'].replace('', np.nan, inplace=True)

    return dataset

def text_tokenization(dataset):
    komoran = Komoran(userdic = 'user_dictionary.txt')
    print(komoran.morphs(dataset[1,1]))
    return dataset

dataset = text_normalization(dataset)
dataset = text_tokenization(dataset)
# dataset.to_csv("dataset.csv", mode='a', header=False, encoding='utf-8-sig')

