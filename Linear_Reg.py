import numpy as np
import pandas as pd
import matplotlib as plt

df = pd.read_csv("Simple.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:, 1].values

#splitting data

from sklearn.model_selection import train_test_split
trainx, trainy, testx, testy = train_test_split(x,y, test_size = .3, random_state = 0)


