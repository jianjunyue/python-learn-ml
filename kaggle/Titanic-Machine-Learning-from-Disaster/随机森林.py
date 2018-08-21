import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import  SelectKBest,f_classif
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train.csv"
data=pd.read_csv(path)
path_test="/Users/jianjun.yue/PycharmGItHub/data/titanic/test.csv"
data_test=pd.read_csv(path_test)
# print(data.head())
# print(data.describe())

# print(data["Age"].describe())
# print(data["Age"].unique())
# print(data["Age"].value_counts())
#用均值填充
data["Age"]=data["Age"].fillna(24)
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

print("--------------追加特征---------------")
print(data["Ticket"].describe())
print(data["Ticket"].unique())
print(data["Ticket"].value_counts())

print("--------------Fare 归一化---------------")
data_scaler = MinMaxScaler(feature_range=(0, 1))
data_Fare= np.array(data["Fare"].values)
lenInt=len(data_Fare)
arr = []
for i in range(0,lenInt):
    temp=[]
    temp.append(data_Fare[i])
    arr.append(temp)
data_rescaledX = data_scaler.fit_transform(arr)
data["Fare_scaler"]=data_rescaledX

print("--------------Name 特征处理---------------")
data["NameLength"]=data["Name"].apply(lambda x:len(x))
def getTitle(name):
    title_search=re.search("([A-Za-z]+)\.",name)
    if title_search:
        return title_search.group(1)
    return ""
titles=data["Name"].apply(getTitle)
title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Major":7,"Col":8,"Mlle":9,"Sir":10,"Lady":11,"Ms":12,"Lady":13,"Jonkheer":14,"Mme":15,"Capt":16,"Don":17,"Countess":18}
for k,v in title_mapping.items():
    titles[titles==k]=v
data["Title"]=titles


# data.to_csv('/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv' ,index = False)

print("--------------RandomForestClassifier---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare_scaler","Embarked","NameLength","Title"]
train=data[predictors]
y=data["Survived"]
# print(train)
# print(train.describe())
kfold=StratifiedKFold(n_splits=3,random_state=1, shuffle=True)#分层划分
alg=RandomForestClassifier()
scores=cross_val_score(alg,train,y,cv=3)
# scores = np.sqrt(-cross_val_score(alg,train,y,cv=3, scoring='neg_mean_squared_error'))
print(scores)
# scores =cross_val_score(alg,train,y,cv=3, scoring='neg_mean_squared_error')
# print(scores)

print("------------特征重要性可视化显示-----------")
selector=SelectKBest(f_classif,k=5)
selector.fit(data[predictors],data["Survived"])
scores= -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation="vertical")
plt.show()
