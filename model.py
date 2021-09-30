import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import dataset as dt
import crawling as cl
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def create_predict(dataset):
  dataset = pd.read_csv('dataset.csv', low_memory=False)
  y_train = np.array(dataset['Binary'])
  x_train = []
  for index in dataset['Title']:
    x_train.append(np.fromstring(index, dtype=int, sep=','))

  x_train = pad_sequences(x_train, maxlen = 20)

  x_train, x_test, y_train, y_test = train_test_split(
      x_train, y_train, test_size=0.2, random_state=1)

  model = Sequential()
  model.add(Embedding(1024, 100))
  model.add(LSTM(128))
  model.add(Dense(1, activation='sigmoid'))

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
  mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
  history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

  loaded_model = load_model('best_model.h5')

  print("\n 테스트 정확도: %.4f" % (load_model.evaluate(x_test, y_test)[1]))
  return load_model


def sentiment_text_processing():
  m_dataset = cl.single_page_crawling_for_modeling()
  m_dataset = dt.text_normalization(m_dataset)
  m_dataset = dt.text_tokenization(m_dataset)
  m_dataset = dt.text_integer(m_dataset)

  return m_dataset
  
def sentiment_predict(m_dataset, model):
  x_train = pad_sequences(m_dataset['Title'], maxlen = 20)

  for sentence in x_train:
    score = max(model.predict(sentence)) # 예측
    score = float(score)
    if(score > 0.5):
      print("{:.2f}% 확률로 비교과프로그램입니다.\n".format(score * 100))
    else:
      print("{:.2f}% 확률로 비교과 프로그램이 아닙니다.\n".format((1 - score) * 100))


dataset = pd.read_csv('dataset.csv', low_memory=False)
model = create_predict(dataset)

tp = dt.text_processing(dataset)


sentiment_predict(tp.dataset, model)