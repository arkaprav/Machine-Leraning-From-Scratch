import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        self.feature_index  = feature_index
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.var_red = var_red
        self.value = value
class Decision_Tree_Regressor:
    def __init__(self,min_sample_split = 2, max_depth = 3):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.root = None
    def build_tree(self,dataset,curr_depth = 0):
        X = dataset[:,:-1]
        y = dataset[:,-1]
        n_samples = np.shape(X)[0]
        if(n_samples>=self.min_sample_split) and (curr_depth<=self.max_depth):
            best_split = self.get_best_split(dataset)
            if best_split["var_red"] > 0:
                left_sub_tree = self.build_tree(best_split['left_dataset'],curr_depth+1)
                right_sub_tree = self.build_tree(best_split['right_dataset'],curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left_sub_tree, right_sub_tree, best_split["var_red"])
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)
    
    def calculate_leaf_value(self, Y):
        val = np.mean(Y)
        return val

    def get_best_split(self,dataset):
        X = dataset[:,:-1]
        y = dataset[:,-1]
        n_features = np.shape(X)[1]
        max_var_red = - float("inf")
        best_split = {}
        for feature_index in range(n_features):
            feature_values = dataset[:,feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                l_dataset , r_dataset = self.split(dataset,feature_index,threshold)
                if(len(l_dataset)>0 and len(r_dataset)>0):
                    l_y , r_y = l_dataset[:,-1], r_dataset[:,-1]
                    curr_var_red = self.var_red(y,l_y,r_y)
                    if(curr_var_red>max_var_red):
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['left_dataset'] = l_dataset
                        best_split['right_dataset'] = r_dataset
                        best_split['var_red'] = curr_var_red
                        max_var_red = curr_var_red
        return best_split

    def var_red(self,parent, left_child, right_child):
        weight_l = len(left_child)/len(parent)
        weight_r = len(right_child)/len(parent)
        var_red = np.var(parent) - (weight_l*np.var(left_child)+weight_r*np.var(right_child))
        return var_red

    def split(self,dataset,feature_index,threshold):
        l_dataset = np.array([row for row in dataset if row[feature_index]<=threshold])
        r_dataset = np.array([row for row in dataset if row[feature_index]>threshold])
        return l_dataset, r_dataset
    
    def fit(self,X,y):
        dataset = np.concatenate((X,y),axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self,X):
        predictions = np.asarray([self.make_prediction(x, self.root) for x in X]).reshape(-1,1)
        return predictions
    
    def make_prediction(self, x, node):
        if node.value != None: return node.value
        feature_val = x[node.feature_index]
        if feature_val<=node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)

data = pd.read_csv("C:\\new_data.csv")
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
dt = Decision_Tree_Regressor()
dt.fit(X_train,Y_train)
Y_pred = dt.predict(X_test)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(Y_pred,Y_test)))