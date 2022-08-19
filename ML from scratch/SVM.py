import numpy as np
import sys
class SVM():
    def fit(self, features, target, epochs = 2147483647, lr = 0.000000001, l = 0.3):
        self.X = features
        self.y = target
        self.epochs = epochs
        self.lr = lr
        self.lam = l
        w = np.ones((self.X.shape[1],1))
        b = 0
        print(self.loss(w,b))
        # w, b = self.gradient_descend(w,b)
        # print(self.loss(w,b))
        # return w, b
        w,b,minloss = self.process()
        self.weights = w
        self.bias = b
        return w, b, minloss
    def hinge_loss(self,w,b):
        hl = np.dot(self.X,w)+b*np.ones((self.X.shape[0],1))
        # print(np.dot(self.X,w))
        p = 1-self.y*hl
        l = []
        for i in range(len(self.y)):
            l.append(max(0,p[i][0]))
        l = np.asarray(l).reshape(-1,1)
        return l
    def loss(self,w,b):
        j = self.lam*np.sum(np.dot(w.T,w))+((1/len(self.y))*np.sum(self.hinge_loss(w,b)))
        return j
    def gradient_descend(self,w,b):
        l = self.hinge_loss(w,b)
        dw = np.zeros(w.shape)
        db = 0
        for i in range(len(l)):
            if(l[i][0]!=0):
                for j in range(len(w)):
                    dw[j]+=2*self.lam*w[j]
                db += 0
            else:
                for j in range(len(w)):
                    # print(self.y[i])
                    # print(self.X[i][j])
                    dw[j]+=2*self.lam*w[j]+self.y[i]*self.X[i][j]
                db+=self.y[i][0]
        # print(dw)
        # print(db)
        w -= self.lr*dw/(len(self.y))
        b -= self.lr*db/(len(self.y))
        return w, b
    def process(self):
        w = np.ones((self.X.shape[1],1))
        b = 0
        minloss = sys.maxsize
        for i in range(self.epochs):
            loss = self.loss(w,b)
            w, b = self.gradient_descend(w,b)
            if(i%10000 == 0):
                print("Epoch completed: "+str(i)+" --- Loss: "+str(loss))
            if(minloss>loss):
                minloss = loss
                w_p = w
                b_p = b
        return w_p, b_p, minloss
    def predict(self, A):
        z = np.dot(A,self.weights)+self.bias*np.ones((self.X.shape[0],1))
        return z


svm = SVM()
l = [[1,2,3],[3,4,5],[5,6,7]]
X = np.asarray(l)
l = [[24],[43],[62]]
y = np.asarray(l)
h, l, loss = svm.fit(X,y,20000000)
print(h)
print(l)
print(loss)
