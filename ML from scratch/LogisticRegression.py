import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as ll
class LGR:
    def fit(self, features, target, epochs = 2147483647, alpha = 0.3):
        self.X = features
        self.y = target
        self.epochs = epochs
        self.alpha = alpha
        theta, b, minloss = self.process()
        self.weights = theta 
        self.bias = b
        return theta, b, minloss

    def sigmoid(self,a):
        return 1/(1+np.exp(-a))

    def log_loss(self,theta,b):
        h = np.dot(self.X,theta)+b*np.ones((self.X.shape[0],1))
        y_pred = self.sigmoid(h)
        a = self.y.reshape(self.y.shape[0], 1)
        return np.sum(a*np.log(y_pred)+(1-a)*np.log(1-y_pred))/(-len(self.y)/2)

    def gradient_descend(self,theta,b):
        h = np.dot(self.X,theta)+b*np.ones((self.X.shape[0],1))
        y_pred = self.sigmoid(h)
        d_theta = (self.alpha/len(self.y))*np.dot(self.X.T,(y_pred - self.y))
        d_b = (self.alpha/len(self.y))*np.sum(y_pred - self.y)
        theta = theta - d_theta
        b = b - d_b
        return theta, b

    def process(self):
        l = [[1],[1],[1]]
        theta = np.asarray(l)
        b = 1
        minloss = sys.maxsize
        for i in range(self.epochs):
            loss = self.log_loss(theta,b)
            theta, b = self.gradient_descend(theta,b)
            if(i%1000000 == 0):
                print("Epoch completed: "+str(i)+" --- Loss: "+str(loss))
            if(minloss>loss):
                minloss = loss
                theta_p = theta
                b_p = b
        return theta_p, b_p, minloss
    
    def predict(self,A):
        h = np.dot(A,self.weights)+self.bias*np.ones((A.shape[0],1))
        print(h)
        y_hat = self.sigmoid(h)
        return y_hat


l = [[1,2,5],[-25,-8,-108],[-25,-9,-109]]
X = np.asarray(l)
l = [[1],[0],[0]]
y = np.asarray(l)
lr = LGR()
theta, b, minloss = lr.fit(X,y,10000000)
print(theta)
print(b)
print(minloss)
l = LogisticRegression()
l.fit(X,y)
y_hat = l.predict(X)
print(l.coef_,l.intercept_,ll(y,y_hat),sep="\n")

