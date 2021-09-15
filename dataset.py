import pandas as pd

csv = pd.read_csv("dataframe.csv", sep='delimiter', header=None)

print(csv.shape)