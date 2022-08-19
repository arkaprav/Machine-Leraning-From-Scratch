import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
class MLR:
    def fit(self,features, target, epochs = 2147483647, lr = 0.0000001):
        self.X = features
        self.y = target
        self.epochs = epochs
        self.lr = lr
        m, b, loss = self.process()
        self.weights = m
        self.bias = b 
        return m, b,loss
    def mse(self, m, b):
        n = len(self.y)
        loss = ((self.y - (np.dot(self.X,m)+b*np.ones((self.X.shape[0],1))))**2)/n
        return np.sum(loss)
    def gradient_descend(self,m,b):
        n = len(self.y)
        l = (np.dot(self.X,m)+b*np.ones((self.X.shape[0],1))) - self.y
        d_m = (2/n)*np.dot(np.transpose(self.X),l)
        d_b = (2/n)*np.sum(l)
        m -= self.lr*d_m
        b -= self.lr*d_b
        return m, b
    def process(self):
        m = np.ones((self.X.shape[1],1))
        b = 1
        minloss = sys.maxsize
        for i in range(self.epochs):
            loss = self.mse(m,b)
            if(i%2000000==0):
                print("epoch completed: "+str(i)+" -- loss: "+str(loss))
            m, b = self.gradient_descend(m,b)
            if(minloss>loss):
                minloss = loss
                mp = m
                bp = b
        return mp, bp, minloss
    def predict(self, A):
        ans = np.dot(A,self.weights)+b*np.ones((self.X.shape[0],1))
        return ans

model = MLR()
l = [[1,2,3],[3,4,5],[5,6,7]]
X = np.asarray(l)
l = [[24],[43],[62]]
y = np.asarray(l)
lr = LinearRegression()
lr.fit(X,y)
print(lr.coef_,lr.intercept_,lr.get_params())
m, b, k = model.fit(X,y)
# loss = model.mse(m,b)
# print(loss)
# print(X)
# print(np.dot(X,m)+b)
# print(y)
# print(y - (np.dot(X,m)+b))
# nw = (np.dot(X,m)+b) - y
# n = len(y)
# m-= 0.0000001*(2/n)*np.dot(np.transpose(X),nw)
print(m)
# b -= 0.0000001*(2/n)*nw
print(b)
print(k)