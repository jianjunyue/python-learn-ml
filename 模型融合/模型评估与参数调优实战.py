# 《python机器学习》第6章 《模型评估与参数调优实战》

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=None)
# print(df[1].value_counts())
class_mapping={ label:idx for idx,label in enumerate(np.unique(df[1])) }
# print(class_mapping)
df[1]=df[1].map(class_mapping)
print(df[1].value_counts())
X=df.loc[:,2:].values
Y=df.loc[:,1].values
# print(Y)

print("-------------------------------train_test_split-----------------------------------------")
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
# print(X_train)
stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.transform(X_test)
print("------------------------------------------------------------------------")
# print(X_train_std)
print("------------------------------------------------------------------------")
# print(X_test_std)

pca=PCA(n_components=5)
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
print("----------------------X_train_pca--------------------------------------------------")
# print(X_train_pca)
print("----------------------X_test_pca--------------------------------------------------")
# print(y_train)
# X_train_pca=X_train
# X_test_pca=X_test

lr=LogisticRegression()
lr.fit(X_train_pca,y_train)
score=lr.score(X_test_pca,y_test)
# print(score)

size=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
for i in size:
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    lr = LogisticRegression()
    lr.fit(X_train_pca, y_train)
    score = lr.score(X_test_pca, y_test)
    # print(i)
    # print(score)
    # print("------------------------------------------------------------------------")

print("-------------------------------StratifiedKFold-----------------------------------------")
kfold=StratifiedKFold(n_splits=10,random_state=1)#分层划分
scores=[]
for train_index, test_index in kfold.split(X,Y):
    # print("Train Index:", train_index, ",Test Index:", test_index)
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=Y[train_index],Y[test_index]
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    score = lr.score(X_test,y_test)
    scores.append(score)
    print(score)


#cross_val_score 等同于StratifiedKFold，支持并行
print("-------------------------------cross_val_score-----------------------------------------")
lr = LogisticRegression()
scores=cross_val_score(estimator=lr,X=X,y=Y,cv=10,n_jobs=2)
print(scores)

