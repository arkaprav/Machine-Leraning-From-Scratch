import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
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
        n_samples = np.shape(X)[0]
        if(n_samples>=self.min_sample) and (curr_depth<self.max_depth-1):
            best_split = self.get_best_split(dataset)
            if best_split['info_gain'] > 0:
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

class Random_Forest_Classier:
    def __init__(self,n_trees = 100,min_sample = 2, max_depth = 3, n_feats = 2, mode = 'entropy'):
        self.n_trees = n_trees
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.mode = mode
        self.trees = []
    def bootstrap(self, dataset, n_bootstrap):
        b_i = np.random.randint(0,len(dataset),size=n_bootstrap)
        df_bootstrap = dataset.iloc[b_i]
        return df_bootstrap
    def sub_samples(self,dataset):
        columns = len(dataset.columns)-1
        if self.n_feats!=None and self.n_feats <= columns:
            b_i = np.random.randint(0,columns,size=self.n_feats)
            return dataset.iloc[:,b_i], b_i
        else:
            print(self.n_feats,columns)
            return None
    def fit(self,X,y):
        y = pd.DataFrame(y)
        l = y.columns
        self.pred_column_name = list(l)
        dataset = pd.concat([X,y],1)
        forest = []
        for i in range(self.n_trees):
            df_bootstrap = self.bootstrap(dataset,int(dataset.shape[0]//1.5))
            samples, b_i =  self.sub_samples(df_bootstrap)
            go_y = df_bootstrap.iloc[:,-1]
            if samples.empty == True:
                print("n_feats out of bound")
                exit()
            dt = Decision_Tree_Classifier(self.min_sample,self.max_depth,self.mode)
            tree = dt.fit(samples.values,go_y.values.reshape(-1,1))
            forest.append((tree,b_i))
        self.forest = forest
        return forest
    def predict(self,A):
        predictions = []
        l = []
        for tree in self.forest:
            dt = Decision_Tree_Classifier(self.min_sample,self.max_depth,self.mode)
            predi = dt.predict(A.iloc[:,tree[1]].values,tree[0])
            predictions.append(predi)
        ans = []
        for i in range(len(predictions[0])):
            l = []
            for j in range(len(self.forest)):
                l.append(predictions[j][i][0])
            a = np.unique(l)
            d ={}
            for k in a:
                d[k] = 0
            for g in l:
                d[g]+=1
            b = max(d.values())
            for h in d.keys():
                if d[h] == b:
                    ans.append(h)
        return pd.DataFrame(np.asarray(ans).reshape(-1,1),index = A.index, columns = self.pred_column_name)
dt = Random_Forest_Classier()
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("C:\\iris.csv", skiprows=1, header=None, names=col_names)
X = data.iloc[:,:-1]
Y = data.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
dt.fit(X_train,Y_train)
Y_pred = dt.predict(X_test)
Y_pred = Y_pred.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1,1)
from sklearn.metrics import accuracy_score
print('Accuracy: '+ str(accuracy_score(Y_test, Y_pred)))
    