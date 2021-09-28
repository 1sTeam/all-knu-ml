import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('dataset.csv', low_memory=False)
# dataset['Text'] = dataset['Text'].fillna('')

# m_dataset = np.array([dataset['Title'].to_numpy,dataset['Text'].to_numpy])
m_dataset = np.array(dataset['Title'])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(m_dataset)
m_dataset = tokenizer.texts_to_sequences(m_dataset)

x_train = []
for i in m_dataset:
    x_train.append(np.array(i))

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 30
below_threshold_len(max_len, x_train)

y_train = np.array(dataset['Binary'])
x_train = pad_sequences(x_train, maxlen = max_len)

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.2, random_state=1)

model = Sequential()
# model.add(Embedding(len(tokenizer.word_index), 100))
# model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))