import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index  = feature_index
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.info_gain = info_gain
        self.value = value

class Decision_Tree_Classifier:
    def __init__(self,min_sample = 2, max_depth = 3, mode = "entropy"):
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.mode = mode
        self.root = None
    
    def build_tree(self,dataset,curr_depth = 0):
        X = dataset[:,:-1]
        y = dataset[:,-1]
        # print(y)
        n_samples = np.shape(X)[0]
        if(n_samples>=self.min_sample) and (curr_depth<=self.max_depth):
            best_split = self.get_best_split(dataset)
            if best_split["info_gain"] > 0:
                left_sub_tree = self.build_tree(best_split['left_dataset'],curr_depth+1)
                right_sub_tree = self.build_tree(best_split['right_dataset'],curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left_sub_tree, right_sub_tree, best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)

    def get_best_split(self,dataset):
        X = dataset[:,:-1]
        y = dataset[:,-1]
        n_features = np.shape(X)[1]
        max_info_gain = - float("inf")
        best_split = {}
        for feature_index in range(n_features):
            feature_values = dataset[:,feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                l_dataset , r_dataset = self.split(dataset,feature_index,threshold)
                if(len(l_dataset)>0 and len(r_dataset)>0):
                    l_y , r_y = l_dataset[:,-1], r_dataset[:,-1]
                    curr_info_gain = self.info_gain(y,l_y,r_y)
                    if(curr_info_gain>max_info_gain):
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['left_dataset'] = l_dataset
                        best_split['right_dataset'] = r_dataset
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def split(self,dataset,feature_index,threshold):
        l_dataset = np.array([row for row in dataset if row[feature_index]<=threshold])
        r_dataset = np.array([row for row in dataset if row[feature_index]>threshold])
        return l_dataset, r_dataset

    def info_gain(self,parent, left_child, right_child):
        weight_l = len(left_child)/len(parent)
        weight_r = len(right_child)/len(parent)
        if self.mode == "gini":
            info_gain = self.gini(parent) - (weight_l*self.gini(left_child)+weight_r*self.gini(right_child))
        else:
            info_gain = self.entropy(parent) - (weight_l*self.entropy(left_child)+weight_r*self.entropy(right_child))
        return info_gain

    def gini(self,node):
        class_labels = np.unique(node)
        gini = 0
        for cls in class_labels:
            p_cls = len(node[node == cls]) / len(node)
            gini += p_cls**2
        return 1 - gini

    def entropy(self, node):
        class_labels = np.unique(node)
        entropy = 0
        for cls in class_labels:
            p_cls = len(node[node == cls]) / len(node)
            entropy += -p_cls*np.log2(p_cls)
        return entropy
    
    def fit(self,X,y):
        dataset = np.concatenate((X,y),axis=1)
        self.root = self.build_tree(dataset)
        return self.root
    
    def predict(self,X, root = None):
        if root  == None:
            r = self.root
        else:
            r = root
        predictions = np.asarray([self.make_prediction(x, r) for x in X]).reshape(-1,1)
        return predictions
    
    def make_prediction(self, x, node):
        if node.value != None: return node.value
        feature_val = x[node.feature_index]
        if feature_val<=node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)

    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
dt = Decision_Tree_Classifier(max_depth=3)
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("C:\\iris.csv", skiprows=1, header=None, names=col_names)
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
a = dt.fit(X_train,Y_train)
Y_pred = dt.predict(X_test,a) 
# print(Y_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))