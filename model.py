from numpy import dot
from numpy.core.numeric import False_
from numpy.linalg import norm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

dataset = pd.read_csv('dataset.csv', low_memory=False)
print(dataset.head(2))
