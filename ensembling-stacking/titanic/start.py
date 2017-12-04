from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from ModelEnsemble import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC
# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

clfs = {'rf': RandomForestClassifier(**rf_params),
    'et': ExtraTreesClassifier(**et_params),
    'ada': AdaBoostClassifier(**ada_params),
    'gb': GradientBoostingClassifier(**gb_params),
    'svc': SVC(**svc_params)
    }

train=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/titqnic.csv")
test =pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/titqnic.csv")
y_train_df=train["Survived"]
X_train_df =train.drop(["id","Survived"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.33,random_state=3)
# print("-----------X_train-------------")
# print(X_train)
# print("-----------y_train-------------")
# print(y_train)
# print("-----------X_test-------------")
# print(X_test)
models=ModelEnsemble()
predicts=models.predict(clfs,X_train,y_train,X_test)
print(predicts)


predicts["Ensemble"]=predicts["RandomForestClassifier"]+predicts["ExtraTreesClassifier"]+predicts["AdaBoostClassifier"]+predicts["GradientBoostingClassifier"]+predicts["SVC"]

predicts.loc[predicts["Ensemble"] <3, "Ensemble"] = 0
predicts.loc[predicts["Ensemble"] >=3, "Ensemble"] = 1
# print(predicts)

for column in predicts.columns:
    accuracy = accuracy_score(y_test, predicts[column])
    print(column)
    print(accuracy)