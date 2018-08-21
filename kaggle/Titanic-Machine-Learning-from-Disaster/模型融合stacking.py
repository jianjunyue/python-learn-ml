import pandas as pd
import numpy as np
from mlens.ensemble import SuperLearner
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from sklearn.ensemble import VotingClassifier
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import  SelectKBest,f_classif
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import re
from scipy.stats import randint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

#模型融合(stacking&blending)
#https://blog.csdn.net/qq_36330643/article/details/78576503

path_test="/Users/jianjun.yue/PycharmGItHub/data/titanic/test_pre.csv"
path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data=pd.read_csv(path)
data_test=pd.read_csv(path_test)
print("--------------RandomForestClassifier---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare_scaler","Embarked","NameLength"]
train=data[predictors]
X=train
y=data["Survived"]
X_submission=data_test[predictors]
print(X_submission.head())
print(X_submission.describe())

print("---------------------------")
print(train.head())
print(train.describe())

seed = 2017

def getRange(start,stop,step):
    listTemp = [start]
    for i in range(start+step, stop, step):
        listTemp.append(i)
    return listTemp

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=1)

print("------------------------RandomForestClassifier-------------------------------")
rfc=RandomForestClassifier(random_state=seed)
rfc.fit(X_train, y_train)
preds =rfc.predict(X_test)
print("RandomForestClassifier ROC AUC:%.3f" % roc_auc_score(y_true=y_test,y_score=preds))
print("RandomForestClassifier accuracy_scorer:%.3f" % accuracy_score(y_true=y_test,y_pred=preds))

print("------------------------GradientBoostingClassifier-------------------------------")
gbc=GradientBoostingClassifier(random_state=seed)
gbc.fit(X_train, y_train)
preds =gbc.predict(X_test)
print("GradientBoostingClassifier ROC AUC:%.3f" % roc_auc_score(y_true=y_test,y_score=preds))
print("GradientBoostingClassifier accuracy_scorer:%.3f" % accuracy_score(y_true=y_test,y_pred=preds))


print("------------------------LogisticRegression-------------------------------")
lgr=LogisticRegression(random_state=seed)
lgr.fit(X_train, y_train)
preds =lgr.predict(X_test)
print("LogisticRegression ROC AUC:%.3f" % roc_auc_score(y_true=y_test,y_score=preds))
print("LogisticRegression accuracy_scorer:%.3f" % accuracy_score(y_true=y_test,y_pred=preds))


print("------------------------集成模型-------------------------------")
clf_labels=["RandomForestClassifier","GradientBoostingClassifier","LogisticRegression"]
for clf,label in zip([rfc,gbc,lgr],clf_labels):
    scores=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring="roc_auc")
    print("ROC AUC: %0.2F (+/- %0.2f) [%s]" % (scores.mean(),scores.std(),label))


print("------------------------VotingClassifier集成模型-------------------------------")
ensemble_clf = VotingClassifier(estimators=[('RandomForestClassifier', rfc), ('GradientBoostingClassifier', gbc), ('LogisticRegression', lgr)],voting='soft', weights=[1,1,1],flatten_transform=True)

clf_labels=["RandomForestClassifier","GradientBoostingClassifier","LogisticRegression","VotingClassifier"]
for clf,label in zip([rfc,gbc,lgr,ensemble_clf],clf_labels):
    scores=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring="roc_auc")
    print("ROC AUC: %0.2F (+/- %0.2f) [%s]" % (scores.mean(),scores.std(),label))

print("------------------------blending集成模型-------------------------------")
clfs=[rfc,gbc,lgr,ensemble_clf]

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

skf= StratifiedKFold(n_splits=3,shuffle=True)
train_indexs=[]
test_indexs=[]
for train_index, test_index in skf.split(X,y):
    train_indexs.append(train_index)
    test_indexs.append(test_index)
dataset_blend_train = np.zeros((len(y), len(clfs)))
for j, clf in enumerate(clfs):
    # print("j:%s" % j)
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(train_indexs)))# 每个分类器的单次fold预测结果
    for i, (train_index, test_index) in enumerate(zip(train_indexs,test_indexs)):
        # print("i:%s" % i)
        b_X_train = X.values[train_index]
        b_y_train = y[train_index]
        b_X_test = X.values[test_index]
        b_y_test = y[test_index]
        clf.fit(b_X_train, b_y_train)
        y_submission = clf.predict_proba(b_X_test)[:, 1]
        dataset_blend_train[test_index, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_submission.values)[:, 1]
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)
# y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
y_submission = clf.predict(dataset_blend_test)
print("Blending Linear stretch of predictions to [0,1]")
# y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
y_pred=pd.DataFrame()
y_pred["PassengerId"]=data_test["PassengerId"]
y_pred["Survived"]=y_submission
y_pred["Survived"]=y_pred["Survived"].apply(lambda x: int(x))
y_pred.to_csv('/Users/jianjun.yue/PycharmGItHub/data/titanic/submission.csv',index=None)