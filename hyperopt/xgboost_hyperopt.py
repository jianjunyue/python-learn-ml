#coding:utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

from sklearn.preprocessing import Imputer
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import cross_val_score
import pickle
import time
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

def loadFile():
    data = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx', header=0, encoding='utf-8')
    # data = data.values
    return data

data = loadFile()
y_train=data["Y"]
train_df=data.drop(["ID","Y"], axis=1)
quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
print(len(quantity))
train_df=train_df[quantity]
X_train = Imputer().fit_transform(train_df)

def GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 5 + 50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]+1
    print("max_depth:" + str(max_depth))
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))
    print("subsample:" + str(subsample))
    print("min_child_weight:" + str(min_child_weight))

    gbm = xgb.XGBRegressor(nthread=4,    #进程数
                            max_depth=max_depth,  #最大深度
                            n_estimators=n_estimators,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            max_delta_step = 10,  #10步不降则停止
                            objective="reg:linear")

    metric = cross_val_score(gbm,X_train,y_train,cv=5,scoring="roc_auc").mean()
    print(metric)
    return -metric

space = {"max_depth":hp.randint("max_depth",15),
         "n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.05,0.06
         "subsample":hp.randint("subsample",4),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM,space,algo=algo,max_evals=4)

print(best)
print(GBM(best))