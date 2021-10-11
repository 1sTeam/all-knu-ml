import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def create_predict(dataset):
  dataset = pd.read_csv('dataset.csv', low_memory=False)
  y_train = np.array(dataset['Binary'])
  x_train = []
  for index in dataset['Title']:
    x_train.append(np.fromstring(index, dtype=int, sep=','))

  x_train = pad_sequences(x_train, maxlen = 12)

  # x_train, x_test, y_train, y_test = train_test_split(
  #     x_train, y_train, test_size=0.2, random_state=1)

  model = Sequential()
  # model.add(Embedding(1152, 100))
  # model.add(Bidirectional(LSTM(100)))
  # model.add(Dense(1, activation='relu'))
  model.add(layers.Dense(16, activation='relu', input_shape=(999,)))
  model.add(layers.Dense(16, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))


  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
  mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
  # history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_split=0.2)
  history = model.fit(x_train, y_train, epochs=15, batch_size=30, validation_split=0.2)

  # loaded_model = load_model('best_model.h5')
  return model


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

  # for sentence in x_train:
  score = model.predict(x_train)

  print(score)

  for s in score:
    s = float(s)
    if(s > 0.5):
      print("{:.2f}% 확률로 비교과프로그램입니다.\n".format(s * 100))
    else:
      print("{:.2f}% 확률로 비교과 프로그램이 아닙니다.\n".format((1 - s) * 100))

def visualize(m_dataset,m_target):
    
    x_train = []
    for index in dataset['Title']:
        x_train.append(np.fromstring(index, dtype=int, sep=','))
        

#전처리 된 데이터셋 불러오기
dataset = pd.read_csv('dataset.csv', low_memory=False)


# visualize(dataset['Title'], dataset['Binary'])
#모델 생성
model = create_predict(dataset)

#모델 불러오기
# model = load_model('best_model.h5')

s_dataset = sentiment_text_processing()
sentiment_predict(s_dataset, model)