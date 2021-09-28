#%%
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('dataset.csv', low_memory=False)
# dataset['Text'] = dataset['Text'].fillna('')

m_target = dataset['Binary'].to_numpy
# m_dataset = np.array([dataset['Title'].to_numpy,dataset['Text'].to_numpy])
m_dataset = dataset['Title'].to_numpy

dataset['Binary'].value_counts().plot(kind='bar')
