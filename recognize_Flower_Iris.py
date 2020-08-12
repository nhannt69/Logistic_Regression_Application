#nhan dang hoa bang model logistic regression su dung thu vien
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, datasets

data = pd.read_csv('iris_Flower.csv').values
X = data[:, 0:4].reshape(-1, 4)
Y = data[:, 4].reshape(-1, 1)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

lgtre = linear_model.LogisticRegression()
lgtre.fit(Xbar, Y)

(w0, w1, w2, w3, w4) = (lgtre.intercept_, lgtre.coef_[0][1], lgtre.coef_[0][2], lgtre.coef_[0][3], lgtre.coef_[0][4])
print(w0, w1, w2, w3, w4)
x1 = (float)(input())
x2 = (float)(input())
x3 = (float)(input())
x4 = (float)(input())
print(1/(1 + np.exp(-(w0+w1*x1+w2*x2+w3*x3+w4*x4))))