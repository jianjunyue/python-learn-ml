import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
#matplotlib inline
from matplotlib.pylab import rcParams

#XGBoost参数调优完全指南（附Python代码）
#http://blog.csdn.net/han_xiaoyang/article/details/52665396

rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('train_modified_part.csv')
# test = pd.read_csv('test_modified.csv')
test = pd.read_csv('test_modified_part.csv')
target='Disbursed'
IDcol = 'ID'
print(train['Disbursed'].value_counts())

def getRange(start,stop,step):
    listTemp = [start]
    for i in range(start+step, stop, step):
        listTemp.append(i)
    return listTemp

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("Model Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    #     Predict on testing data:
    # dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    # # results = test_results.merge(dtest[['ID', 'predprob']], on='ID')
    # print('AUC Score (Test): %f' % metrics.roc_auc_score(dtest['Disbursed'], dtest['predprob']))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb1, train, predictors)

#参数调优的一般方法
#1.选择较高的学习速率(learning rate)。一般情况下，学习速率的值为0.1。但是，对于不同的问题，理想的学习速率有时候会在0.05到0.3之间波动。选择对应于此学习速率的理想决策树数量。XGBoost有一个很有用的函数“cv”，这个函数可以在每一次迭代中使用交叉验证，并返回理想的决策树数量。
#2.对于给定的学习速率和决策树数量，进行决策树特定参数调优(max_depth, min_child_weight, gamma, subsample, colsample_bytree)。在确定一棵树的过程中，我们可以选择不同的参数，待会儿我会举例说明。
#3.xgboost的正则化参数的调优。(lambda, alpha)。这些参数可以降低模型的复杂度，从而提高模型的表现
#4.降低学习速率，确定理想参数。（learning_rate，n_estimators）

#第一步：确定学习速率和tree_based 参数调优的估计器数目
#为了确定boosting参数，我们要先给其它参数一个初始值。咱们先按如下方法取值：
#1.max_depth = 5 :这个参数的取值最好在3-10之间。我选的起始值为5，但是你也可以选择其它的值。起始值在4-6之间都是不错的选择。
#2.min_child_weight = 1:在这里选了一个比较小的值，因为这是一个极不平衡的分类问题。因此，某些叶子节点下的值会比较小
#3.gamma = 0: 起始值也可以选其它比较小的值，在0.1到0.2之间就可以。这个参数后继也是要调整的
#4.subsample, colsample_bytree = 0.8: 这个是最常见的初始值了。典型值的范围在0.5-0.9之间
#5.scale_pos_weight = 1: 这个值是因为类别十分不平衡

#初始参数如下：
#learning_rate([默认0.3],典型值为0.01-0.2)
predictors = [x for x in train.columns if x not in [target,IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


#第二步：max_depth([默认6],典型值：3-10) 和 min_child_weight([默认1],典型值：3-10) 参数调优
param_test1 = {
 'max_depth':getRange(3,10,2),
 'min_child_weight':getRange(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27),param_grid = param_test1,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch1.fit(train[predictors],train[target])
# print("------------------------")
# print(gsearch1.grid_scores_)
# print("------------------------")
# print(gsearch1.best_params_)
# print("------------------------")
# print(gsearch1.best_score_)

#第三步：gamma([默认0]，典型值：0-0.2)参数调优
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
# print("------------------------")
# print(gsearch3.grid_scores_)
# print("------------------------")
# print(gsearch3.best_params_)
# print("------------------------")
# print(gsearch3.best_score_)

#第四步：调整subsample([默认1],典型值：0.5-0.9) 和 colsample_bytree([默认1],典型值：0.5-0.9) 参数
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3, min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
print("------------------------")
print(gsearch4.grid_scores_)
print("------------------------")
print(gsearch4.best_params_)
print("------------------------")
print(gsearch4.best_score_)

#第五步：正则化参数调优 lambda->reg_lambda([默认1]) , alpha->reg_alpha[默认1]
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4, min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])

#第六步：降低学习速率
learning_rate =0.01,
n_estimators=5000,