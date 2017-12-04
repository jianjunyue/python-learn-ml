import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import datetime
import matplotlib.pyplot as plt

df_train=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/stock_train_data_20170901.csv")

print(type(df_train))
df_train=df_train[df_train["era"]==20]

print("-------------era---------------------------------------------------------------------------------------------")
print(df_train["era"].value_counts())

delcolumns=["id","label","weight","era"]
# new_names=filter(lambda name:name not in delcolumns,  stock_data.columns)
predictors= [name for name in df_train.columns if name not in delcolumns]
# print(new_names)
# df_train=stock_data[stock_data.columns]
ytrain=df_train["label"]
#
# #初始参数如下：
# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# # modelfit(xgb1, df_train[predictors].values,ytrain)
#
def getRange(start,stop,step):
    listTemp = [start]
    for i in range(start+step, stop, step):
        listTemp.append(i)
    return listTemp
#
#
# #第二步：max_depth([默认6],典型值：3-10) 和 min_child_weight([默认1],典型值：3-10) 参数调优
param_test1 = {
 'max_depth':getRange(3,10,2),
 'min_child_weight':getRange(2,8,2)
}
# gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=50, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27),param_grid = param_test1,scoring='roc_auc',n_jobs=2,iid=False,cv=5)
# gsearch1.fit(df_train[predictors],ytrain)
# print("------------------------")
# print(gsearch1.grid_scores_)
# print("------------------------")
# print(gsearch1.best_params_)
# print("------------------------")
# print(gsearch1.best_score_)
#
#
# #第三步：gamma([默认0]，典型值：0-0.2)参数调优
# param_test3 = {
#  'gamma':[i/100.0 for i in range(0,5)]
# }
# # gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=2, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# # gsearch3.fit(df_train[predictors],ytrain)
# # print("------------------------")
# # print(gsearch3.grid_scores_)
# # print("------------------------")
# # print(gsearch3.best_params_)
# # print("------------------------")
# # print(gsearch3.best_score_)
#
# #第四步：调整subsample([默认1],典型值：0.5-0.9) 和 colsample_bytree([默认1],典型值：0.5-0.9) 参数
# param_test4 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }
# # gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=2, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# # gsearch4.fit(df_train[predictors],ytrain)
# # print("------------------------")
# # print(gsearch4.grid_scores_)
# # print("------------------------")
# # print(gsearch4.best_params_)
# # print("------------------------")
# # print(gsearch4.best_score_)
#
# #第五步：正则化参数调优 lambda->reg_lambda([默认1]) , alpha->reg_alpha[默认1]
# param_test6 = {
#  'reg_alpha':[3.9,4,4.1]
# }
# # gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=2, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.9, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# # gsearch6.fit(df_train[predictors],ytrain)
# # print("------------------------")
# # print(gsearch6.grid_scores_)
# # print("------------------------")
# # print(gsearch6.best_params_)
# # print("------------------------")
# # print(gsearch6.best_score_)
#
# #第六步：降低学习速率
# learning_rate =0.01,
# n_estimators=5000,
# # gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.01, n_estimators=5000, max_depth=2, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.9, objective= 'binary:logistic', nthread=4,reg_alpha=4, scale_pos_weight=1,seed=27),param_grid={}, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# # gsearch6.fit(df_train[predictors],ytrain)
# # print("------------------------")
# # print(gsearch6.grid_scores_)
# # print("------------------------")
# # print(gsearch6.best_params_)
# # print("------------------------")
# # print(gsearch6.best_score_)
#
xgb3 = XGBRegressor(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=6,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=2,
 scale_pos_weight=1,
 seed=27)
xgb3.fit(df_train[predictors],ytrain)
# xgb3.save_model('model/xgb.model20') # 用于存储训练出的模型
# xgb3.predict()

from sklearn.externals import joblib
modelPath="/Users/jianjun.yue/PycharmGItHub/model/xgb.model"+str(datetime.datetime.now().strftime("%Y%m%d%H%M"))+".pkl"
joblib.dump(xgb3,modelPath)

xgbtemp=joblib.load(modelPath)
df_test=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/stock_test_data_20170901.csv")
predictions=xgbtemp.predict(df_test[predictors])
print(type(predictions))
new_predictions=[]
for score in predictions:
    if score>0.98:
        score=0.98
    elif score<0.01:
        score = 0.01
    new_predictions.append(score)

df_test["proba"]=new_predictions
# df_test.loc[df_test["proba"]>0.98] = 0.98
# df_test.loc[df_test["proba"]<0.01] = 0.01
result =df_test[["id","proba"]]
# result["id"]=result["id"].astype("int")
# print(result.head(10))
# print(predictions)
result.to_csv("/Users/jianjun.yue/PycharmGItHub/data/submission"+str(datetime.datetime.now().strftime("%Y%m%d%H%M"))+".csv",index=False)
#
#
# params={
# 'booster':'gbtree',
# 'objective': 'binary:logistic', #多分类的问题
# 'num_class':2, # 类别数，与 multisoftmax 并用
# 'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
# 'max_depth':9, # 构建树的深度，越大越容易过拟合
# # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
# 'subsample':0.8, # 随机采样训练样本
# 'colsample_bytree':0.8, # 生成树时进行的列采样
# 'min_child_weight':3,
# # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
# #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
# #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
# 'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
# 'eta': 0.001, # 如同学习率
# 'seed':1000,
# 'nthread':2,# cpu 线程数
# 'eval_metric': 'auc'
# }
#
# plst = list(params.items())
# num_rounds = 5000 # 迭代次数
# watchlist = [(df_train[predictors], 'train'),(df_train["affairs"], 'val')]
# #训练模型并保存
# # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
# model = xgb.train(plst,df_train[predictors], num_rounds, watchlist,early_stopping_rounds=100)
# model.save_model('model/xgb.model') # 用于存储训练出的模型
# print("best best_ntree_limit",model.best_ntree_limit)
