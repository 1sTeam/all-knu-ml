import pandas as pd
import numpy as np
from pykospacing import Spacing
from konlpy.tag import Komoran
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.normalizer import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class text_processing():
    def __init__(self,dataset):
        self.dataset = dataset
        self.token_size = 0

    def text_normalization(self):
        self.dataset = self.dataset.dropna(how = 'any')
        self.dataset['Title'] = self.dataset['Title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","")
        self.dataset['Title'].replace('', np.nan, inplace=True)
        return self.dataset

    def text_tokenization(self):
        komoran = Komoran(userdic= 'dic.txt')
        temp = []
        for sentence in self.dataset['Title']:
            temp.append(komoran.nouns(sentence))
        self.dataset['Title'] = temp
        return self.dataset

    def text_integer(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.dataset['Title'])
        token_size = len(tokenizer.word_index)

        tokenizer = Tokenizer(token_size)
        tokenizer.fit_on_texts(self.dataset['Title'])

        self.dataset['Title'] = tokenizer.texts_to_sequences(self.dataset['Title'])
        self.token_size = token_size
        return self.dataset

# # 머신러닝 학습을 위한 공지사항 data 추출
# df_csv = pd.read_csv("dataframe.csv",encoding='utf-8')
# dataset = df_csv[['Binary','Title']]

# #데이터 셋
# dataset = dataset.head(2000)
# dataset = text_processing(dataset)
# dataset.text_normalization()
# dataset.text_tokenization()
# dataset.text_integer()

# # 데이터 셋 저장
# dataset.dataset.to_csv("dataset.csv")

