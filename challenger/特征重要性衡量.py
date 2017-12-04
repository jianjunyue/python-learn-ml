#特征重要性衡量
from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pandas import Series, DataFrame
# titanic=pd.read_csv("data/titqnic.csv")
#
# new_predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]
# selector=SelectKBest(f_classif,k=5)
# selector.fit(titanic[new_predictors],titanic["Survived"])
# scores=-np.log10(selector.pvalues_)
# print(scores)
# plt.bar(range(len(new_predictors)),scores)
# plt.xticks(range(len(new_predictors)),new_predictors,rotation='vertical')
# plt.show()



df_train=pd.read_csv("/Users/jianjun.yue/PycharmProjects/data/stock_train_data_20170901.csv")
df_train=df_train[df_train["era"]==20]

print("-------------era---------------------------------------------------------------------------------------------")
print(df_train["era"].value_counts())

delcolumns=["id","label","weight","era"]
# new_names=filter(lambda name:name not in delcolumns,  stock_data.columns)
predictors= [name for name in df_train.columns if name not in delcolumns]
# print(new_names)
# df_train=stock_data[stock_data.columns]
y=df_train["label"]

print(type(df_train))

X=df_train[predictors]
selector=SelectKBest(f_classif,k=5)
selector.fit(X,y)
scores=-np.log10(selector.pvalues_)
dt=DataFrame()
dt["predictors"]=predictors
dt["scores"]=scores
dt=dt.sort_values(by = 'scores',axis = 0,ascending = False)
print(dt["scores"])
plt.bar(range(len(predictors)),dt["scores"])
plt.xticks(range(len(predictors)),dt["predictors"],rotation='vertical')
plt.show()
#少样本的情况情况下绘出学习曲线

from common.FeatureWeightValues import FeatureValues
fv=FeatureValues()
fv.sort_values(X,y,predictors)

