import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import dataset as dt
import crawling as cl
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM,Bidirectional
from tensorflow.keras.models import Sequential
from keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflowjs as tfjs
from pandas.plotting import scatter_matrix

def create_predict():
  dataset = read_csv("dataset.csv")
  y_train = np.array(dataset['Binary'])
  x_train = []
  for index in dataset['Title']:
    x_train.append(np.fromstring(index, dtype=np.uint8, sep=','))

  x_train = pad_sequences(x_train, maxlen = 12)
  print(x_train.shape)
  x_train, x_test, y_train, y_test = train_test_split(
      x_train, y_train, test_size=0.2, random_state=1)

  model = Sequential()

  model.add(Embedding(1152, 100))
  model.add(LSTM(128))
  # model.add(Dense(1, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  # model.add(layers.Dense(16, activation='relu', input_shape=(len(dataset),)))
  # model.add(layers.Dense(1, activation='sigmoid'))


  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
  mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
  history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_split=0.2)
  # history = model.fit(x_train, y_train, epochs=15, batch_size=60, validation_split=0.2)
  loaded_model = load_model('best_model.h5')
  
  print("model 검증 정확도 : %.4f" %(loaded_model.evaluate(x_test,y_test)[1]))
  return loaded_model


def sentiment_text_processing():
  m_dataset = cl.single_page_crawling_for_modeling()
  print(m_dataset['Title'])
  m_dataset = dt.text_processing(m_dataset)
  m_dataset.text_normalization()
  m_dataset.text_tokenization()
  m_dataset.text_integer()

  return m_dataset.dataset
  
def sentiment_predict(m_dataset, model):
  x_train = pad_sequences(m_dataset['Title'], maxlen = 12)
  for sentence in x_train:
    sentence = sentence.reshape(1,-1)
    score = float(model.predict(sentence))
    if(score > 0.5):
      print("{:.2f}% 확률로 비교과프로그램입니다.\n".format(score * 100))
    else:
      print("{:.2f}% 확률로 비교과 프로그램이 아닙니다.\n".format((1 - score) * 100))

# # visualize(dataset['Title'], dataset['Binary'])
# #모델 생성
model = create_predict()

#모델 불러오기
# model = load_model('best_model.h5')

# 모델 저장
# model.save(filepath="m_model")
# tfjs.converters.save_keras_model(model, "m_model_tfjs")

s_dataset = sentiment_text_processing()
sentiment_predict(s_dataset, model)
