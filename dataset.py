import pandas as pd
import numpy as np
from pykospacing import Spacing
from ckonlpy.tag import Twitter
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
    print(dataset.isnull().sum())

    return dataset
def text_tokenization(dataset):
    twitter = Twitter()
    twitter.add_dictionary('비교과프로그램','Noun')
    twitter.add_dictionary('집중학습지원','Noun')
    twitter.add_dictionary('취창업지원센터','Noun')
    twitter.add_dictionary('진로취창업센터','Noun')
    twitter.add_dictionary('마음나눔센터','Noun')
    twitter.add_dictionary('CTL','Noun')
    twitter.add_dictionary('참인재','Noun')
    twitter.add_dictionary('취업동아리','Noun')
text_normalization(dataset)

# dataset.to_csv("dataset.csv", mode='a', header=False, encoding='utf-8-sig')

