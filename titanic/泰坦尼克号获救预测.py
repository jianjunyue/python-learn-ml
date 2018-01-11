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

print(titanic["Sex"].unique())

# print(titanic["Embarked"].unique())
titanic["Embarked"]=titanic["Embarked"].fillna("S")
# print(titanic["Embarked"].unique())
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2
# print(titanic["Embarked"].unique())
# titanic=titanic[1:]
# print(titanic.head(3))

#线性回归模型预测
from sklearn.linear_model import  LinearRegression
from sklearn.cross_validation import  KFold
#"Pclass"-船窗等级,"Sex"-性别,"Age"-年龄,"SibSp"-兄弟姐妹数,"Parch"-老人小孩数,"Fare"-船票,"Embarked"-登船地点
# predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
predictors=["Pclass","Sex" ,"SibSp","Parch","Fare","Embarked"]
alg=LinearRegression()

t=[[0, 0], [1, 1], [2, 2]]
tt=[0, 1, 2]
alg.fit(t,tt)
print("-----------dddd---")
newtitanic=titanic[predictors]
# print(newtitanic.describe())
newSurvived=titanic["Survived"]
# alg.fit(newtitanic,newSurvived)
print("-----------dddd---")

kf=KFold(titanic.shape[0],3,random_state=1)

print("-------线性回归模型预测-------")
predictions=[]
for train,test in kf:
    train_predictors=(titanic[predictors].iloc[train,:])
    train_target=titanic["Survived"].iloc[train]
    print("--------------")
    # print(train_predictors)

    alg.fit(train_predictors,train_target)
    test_predictions=alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
# predictions[predictions >.5]=1
# predictions[predictions <=.5]=0
# accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)
# print(accuracy)

#逻辑回归模型
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression(random_state=1)
print(titanic[predictors].head())
print("-------逻辑回归模型-------")
print(titanic["Survived"].head())
scores=cross_validation.cross_val_score(lgr,titanic[predictors],titanic["Survived"],cv=3)
print(scores.mean())

#随机森林模型
from sklearn.ensemble import RandomForestClassifier
print("-------随机森林模型-------")
rfc=RandomForestClassifier(random_state=1,n_estimators=10,min_samples_split=6,min_samples_leaf=3)
kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
scores=cross_validation.cross_val_score(rfc,titanic[predictors],titanic["Survived"],cv=kf)
print(scores.mean())

#特征工程
titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]
titanic["NameLength"]=titanic["Name"].apply(lambda x:len(x))
import re
def getTitle(name):
    title_search=re.search("([A-Za-z]+)\.",name)
    if title_search:
        return title_search.group(1)
    return ""
# print(getTitle("Braund, Mr. Owen Harris"))
titles=titanic["Name"].apply(getTitle)
# print(pd.value_counts(titles))

title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Major":7,"Col":8,"Mlle":9,"Sir":10,"Lady":11,"Ms":12,"Lady":13,"Jonkheer":14,"Mme":15,"Capt":16,"Don":17,"Countess":18}
for k,v in title_mapping.items():
    titles[titles==k]=v
# print(pd.value_counts(titles))
titanic["Title"]=titles
# print(titanic)

#特征重要性衡量
predictors=["Pclass","Sex" ,"SibSp","Parch","Fare","Embarked","FamilySize","NameLength","Title"]
print("------------增加新特征-----------")
rfc=RandomForestClassifier(random_state=1,n_estimators=100,min_samples_split=8,min_samples_leaf=4)
kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
scores=cross_validation.cross_val_score(rfc,titanic[predictors],titanic["Survived"],cv=kf)
print(scores.mean())

from sklearn.feature_selection import  SelectKBest,f_classif
import matplotlib.pyplot as plt
print("------------特征重要性可视化显示-----------")
selector=SelectKBest(f_classif,k=5)
selector.fit(titanic[predictors],titanic["Survived"])
scores= -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation="vertical")
# plt.show()

print("------------重要特征训练模型-----------")
predictors=["Pclass","Sex","Fare","Title","NameLength"]
rfc=RandomForestClassifier(random_state=1,n_estimators=100,min_samples_split=8,min_samples_leaf=4)
kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
scores=cross_validation.cross_val_score(rfc,titanic[predictors],titanic["Survived"],cv=kf)
print(scores.mean())

#多种模型集成混合
from sklearn.ensemble import GradientBoostingClassifier
print("------------集成模型-----------")
algorithms=[
    [GradientBoostingClassifier(random_state=1,n_estimators=25,max_depth=3),predictors],
    [LogisticRegression(random_state=1),predictors]
]
kf=KFold(titanic.shape[0],n_folds=3,random_state=1)
predictions=[]
for train,test in kf:
    train_target=titanic["Survived"].iloc[train]
    full_test_predictions=[]
    for alg,predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train,:],train_target)
        test_predictions=alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions=(full_test_predictions[0]+full_test_predictions[1])/2
    test_predictions[test_predictions<=.5]=0
    test_predictions[test_predictions>.5]=1
    predictions.append(test_predictions)

predictions=np.concatenate(predictions,axis=0)
accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)
print(accuracy)

print("------------集成加权模型-----------")
algorithms=[
    [GradientBoostingClassifier(random_state=1,n_estimators=25,max_depth=3),predictors],
    [LogisticRegression(random_state=1),predictors]
]
full_predictions=[]
for alg,predictors in algorithms:
        alg.fit(titanic[predictors],titanic["Survived"])
        predictions=alg.predict_proba(titanic.astype(float))[:,1]
        full_predictions.append(predictions)

predictions=(full_test_predictions[0]*3+full_test_predictions[1])/4


predictions=np.concatenate(predictions,axis=0)
accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)
print(accuracy)

























