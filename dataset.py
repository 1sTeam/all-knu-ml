import pandas as pd
from pykospacing import Spacing
from konlpy.tag import Okt
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

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
    
def text_tokenazation(dataset):
    tokenizer = Okt()

    #토큰화 학습 (시간이 오래걸리나 안할시 성능 구려짐)
    '''
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
    '''
    for index, row in dataset.iterrows():
        if(type(row.Text) == type("str")):
            dataset.at[index, 'Text'] = tokenizer.morphs(row.Text)

    return dataset

test_dataset = text_spacing(test_dataset)
test_dataset = text_tokenazation(test_dataset)
print(test_dataset.Text)

