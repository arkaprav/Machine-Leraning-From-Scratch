import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
class LR:
    def fit(self,X,y,epochs = 10, l=0.01):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.l = l
        self.m, self.b, self.loss = self.process()
        print(str(self.m) + "," + str(self.b) + "," + str(self.loss))
    def mse(self, m, b):
        loss = 0
        for i in range(self.y.size):
            loss += (self.y[i] - (m * self.X[i][0] + b)) ** 2
        return loss/float(self.y.size)
    def gradient_descent(self, m, b):
        n = self.y.size
        m_gradient = 0
        b_gradient = 0
        for i in range(n):
            m_gradient += - (2/n) * self.X[i][0] * (self.y[i] - (m * self.X[i][0] + b))
            b_gradient += - (2/n) * (self.y[i] - (m * self.X[i][0] + b))
        m -= self.l*m_gradient
        b -= self.l*b_gradient
        return m, b
    def process(self):
        m = 1
        b = 1
        minloss = sys.maxsize
        for i in range(self.epochs):
            if(i%10000==0):
                print("epoch completed:" + str(i))
            loss = self.mse(m,b)
            m,b = self.gradient_descent(m,b)
            if(minloss>loss):
                minloss = loss
                mp = m
                bp = b
        return mp, bp, minloss
    def predict(self, A):
        y = []
        for i in A:
            y[i] = self.m * i + self.b
        return y
        

df = pd.read_csv("C://data.csv")

X = np.asarray(df["SAT"]).reshape(-1,1)
y = df["GPA"]
print(X[0])
lr = LinearRegression()
lr.fit(X,y)
y_hat = lr.predict(X)
print(mean_squared_error(y,y_hat))
print(lr.coef_,lr.intercept_,lr.get_params())

lrp = LR()
lrp.fit(X,y,50000, 0.0000001)

# l = [[i for i in range(1,20,2)]]
# a =np.asarray(l)
# l = [[0, 5, 9, 13, 24, 25, 29, 33, 40]]
# b =np.asarray(l)
# print(b)
# lr = LR()
# lr.fit(a,b,18500,0.0001)
# lrp = LinearRegression()
# lrp.fit(a,b)
# l = [[3,2,5,6,6,7,8,9,11,10]]
# print(lrp.predict(np.asarray(l)))