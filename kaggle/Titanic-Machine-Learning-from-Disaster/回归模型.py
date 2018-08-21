import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train.csv"
data=pd.read_csv(path)
# print(data.head())
# print(data.describe())

# print(data["Age"].describe())
# print(data["Age"].unique())
#用均值填充
# data["Age"]=data["Age"].fillna(data["Age"].median)
data["Age"]=data["Age"].fillna(25)
print("--------------Sex---------------")
print(data["Sex"].describe())
print(data["Sex"].unique())
print(data["Sex"].value_counts())
data.loc[data["Sex"]=="male","Sex"]=0
data.loc[data["Sex"]=="female","Sex"]=1
print(data["Sex"].describe())
print(data["Sex"].unique())
print("--------------Embarked---------------")
print(data["Embarked"].describe())
print(data["Embarked"].unique())
print(data["Embarked"].value_counts())
data.loc[data["Embarked"]=="S","Embarked"]=0
data.loc[data["Embarked"]=="C","Embarked"]=1
data.loc[data["Embarked"]=="Q","Embarked"]=2
data["Embarked"]=data["Embarked"].fillna(3)
# data.loc[data["Embarked"]==None,"Embarked"]=3
print(data["Embarked"].describe())
print(data["Embarked"].unique())
print(data["Embarked"].value_counts())

print("--------------LogisticRegression---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
train=data[predictors].values
y=data["Survived"]
print(train)
# print(train.describe())
kfold=StratifiedKFold(n_splits=3,random_state=1)#分层划分
scores=[]
# for train_index, test_index in kfold.split(train,y):
#     # print("Train Index:", train_index, ",Test Index:", test_index)
#     X_train,X_test=train[train_index],train[test_index]
#     y_train,y_test=y[train_index],y[test_index]
#     lr = LogisticRegression()
#     lr.fit(X_train,y_train)
#     score = lr.score(X_test,y_test)
#     scores.append(score)
#     print(score)

print(data["Age"])
print(pd.qcut(data['Age'],5))