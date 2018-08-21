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

path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train.csv"
path_test="/Users/jianjun.yue/PycharmGItHub/data/titanic/test_pre.csv"
path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data=pd.read_csv(path)
data_test=pd.read_csv(path_test)
print("--------------RandomForestClassifier---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare_scaler","Embarked","NameLength"]
train=data[predictors]
y=data["Survived"]
X_submission=data_test[predictors]
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

ensemble_clf.fit(X_train, y_train)
preds =ensemble_clf.predict(X_test)
print("VotingClassifier ROC AUC:%.3f" % roc_auc_score(y_true=y_test,y_score=preds))
print("VotingClassifier accuracy_scorer:%.3f" % accuracy_score(y_true=y_test,y_pred=preds))

clf_labels=["RandomForestClassifier","GradientBoostingClassifier","LogisticRegression","VotingClassifier"]
for clf,label in zip([rfc,gbc,lgr,ensemble_clf],clf_labels):
    scores=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring="roc_auc")
    print("ROC AUC: %0.2F (+/- %0.2f) [%s]" % (scores.mean(),scores.std(),label))

es=ensemble_clf.fit(train, y)
y_submission =es.predict(X_submission)
y_pred=pd.DataFrame()
y_pred["PassengerId"]=data_test["PassengerId"]
y_pred["Survived"]=y_submission
# y_pred["Survived"]=y_pred["Survived"].apply(lambda x: int(x))
y_pred.to_csv('/Users/jianjun.yue/PycharmGItHub/data/titanic/submission.csv',index=None)

print("------------------------VotingClassifier集成模型评估-------------------------------")
# colors=["black","orange","blue","green"]
# linestyles=[":","-","--","-."]
# clf_labels=["RandomForestClassifier","GradientBoostingClassifier","LogisticRegression","VotingClassifier"]
# for clf,label,clr,style in zip([rfc,gbc,lgr,ensemble_clf],clf_labels,colors,linestyles):
#     y_pred=clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
#     fpr,tpr,thresholds=roc_curve(y_true=y_test,y_score=y_pred)
#     roc_auc=auc(x=fpr,y=tpr)
#     print("fpr:%s , tpr:%s , roc_auc:%s , label:%s" % (fpr,tpr,roc_auc,label))
#     plt.plot(fpr,tpr,color=clr,linestyle=style,label="%s (auc=%0.2f)" % (label,roc_auc))
#
# plt.legend(loc="lower right")
# plt.plot([0,1],[0,1],linestyle="--",color="gray",linewidth=2)
# plt.xlim([-0.1,1.1])
# plt.ylim([-0.1,1.1])
# plt.grid()
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.show()








