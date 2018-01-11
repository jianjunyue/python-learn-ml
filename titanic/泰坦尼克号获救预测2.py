import pandas as pd
import numpy as np

titanic=pd.read_csv("/Users/yuejianjun/PycharmProjects/PythonMLCSDNProject/file/train.csv")
# print(titanic.head())
# print(titanic.describe())
#数据预处理
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median)
# print(titanic.describe())
# print(titanic.describe())

# print(titanic["Sex"].unique())

titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1

# print(titanic["Sex"].unique())

# print(titanic["Embarked"].unique())
titanic["Embarked"]=titanic["Embarked"].fillna("S")
# print(titanic["Embarked"].unique())
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2
# print(titanic["Embarked"].unique())
# titanic=titanic[1:]
# print(titanic.head(3))

predictors=["Pclass","Sex" ,"SibSp","Parch","Fare","Embarked"]
X=titanic[predictors]
y=titanic["Survived"]
from sklearn.cross_validation import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
# print(X_train.head(3))

from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer()
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))
print(len(vec.feature_names_))
# print(vec)

#使用决策树模型预测和评估
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
score=dt.score(X_test,y_test)
print(score)


#筛选前20%特征决策树模型预测和评估
from sklearn import feature_selection
fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)
X_train_fs=fs.fit_transform(X_train,y_train)

dt.fit(X_train_fs,y_train)
X_train_fs=fs.transform(X_test)
score=dt.score(X_train_fs,y_test)
print(score)

#交叉验证
from sklearn.cross_validation import  cross_val_score
import numpy as np
print("--------交叉验证-------")
percentiles=range(1,100,2)
results=[]
for i in percentiles:
    fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    X_train_fs=fs.fit_transform(X_train,y_train)
    scores=cross_val_score(dt,X_train_fs,y_train,cv=5)
    results=np.append(results,scores.mean())
print(results)

opt=np.where(results==results.max())[0]
print("最优体征筛选的百分比：")
# print(opt)
# print(percentiles[opt])

import pylab as pl
pl.plot(percentiles,results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
# pl.show()

#最佳体征，训练模型
from sklearn import feature_selection
fs=feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
X_train_fs=fs.fit_transform(X_train,y_train)
dt.fit(X_train_fs,y_train)
X_test_fs=fs.transform(X_test)
# print(dt.score(X_train_fs,y_test))

from sklearn.ensemble import RandomForestClassifier
print("---- 随机森林模型 -----")
rfc=RandomForestClassifier()
score=cross_val_score(rfc,X_train,y_train,cv=5).mean()
print(score)

from xgboost import XGBClassifier
print("---- xgboost随机森林模型 -----")
xgbc=XGBClassifier()

from sklearn.cross_validation import cross_val_score
score=cross_val_score(xgbc,X_train,y_train,cv=5).mean()
print(score)

