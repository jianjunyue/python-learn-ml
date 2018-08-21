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
data_test["Age"]=data_test["Age"].fillna(24)
print("--------------Sex---------------")
print(data["Sex"].describe())
print(data["Sex"].unique())
print(data["Sex"].value_counts())
data.loc[data["Sex"]=="male","Sex"]=0
data.loc[data["Sex"]=="female","Sex"]=1

data_test.loc[data_test["Sex"]=="male","Sex"]=0
data_test.loc[data_test["Sex"]=="female","Sex"]=1


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

data_test.loc[data_test["Embarked"]=="S","Embarked"]=0
data_test.loc[data_test["Embarked"]=="C","Embarked"]=1
data_test.loc[data_test["Embarked"]=="Q","Embarked"]=2
data_test["Embarked"]=data_test["Embarked"].fillna(3)

# data.loc[data["Embarked"]==None,"Embarked"]=3
print(data["Embarked"].describe())
print(data["Embarked"].unique())
print(data["Embarked"].value_counts())

print("--------------追加特征---------------")
print(data["Ticket"].describe())
print(data["Ticket"].unique())
print(data["Ticket"].value_counts())

print("--------------Fare 归一化---------------")
print(data_test["Fare"].describe())
print(data_test["Fare"].unique())
print(data_test["Fare"].value_counts())
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

data_scaler = MinMaxScaler(feature_range=(0, 1))
data_test["Fare"]=data_test["Fare"].fillna(10)
data_Fare= np.array(data_test["Fare"].values)
lenInt=len(data_Fare)
arr = []
for i in range(0,lenInt):
    temp=[]
    temp.append(data_Fare[i])
    arr.append(temp)
data_rescaledX = data_scaler.fit_transform(arr)
data_test["Fare_scaler"]=data_rescaledX

print("--------------Name 特征处理---------------")
data["NameLength"]=data["Name"].apply(lambda x:len(x))

data_test["NameLength"]=data_test["Name"].apply(lambda x:len(x))
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

titles_test=data_test["Name"].apply(getTitle)
title_mapping_test={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Major":7,"Col":8,"Mlle":9,"Sir":10,"Lady":11,"Ms":12,"Lady":13,"Jonkheer":14,"Mme":15,"Capt":16,"Don":17,"Countess":18}
for k,v in title_mapping_test.items():
    titles_test[titles_test==k]=v
data_test["Title"]=titles_test


data.to_csv('/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv' ,index = False)
data_test.to_csv('/Users/jianjun.yue/PycharmGItHub/data/titanic/test_pre.csv' ,index = False)
