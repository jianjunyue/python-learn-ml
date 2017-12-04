import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
#matplotlib inline
from matplotlib.pylab import rcParams
#调参
class xgb_ParameterTuning(object):

    def __init__(self):
        self.train = pd.read_csv('train_modified_part.csv')
        self.target = 'Disbursed'
        IDcol = 'ID'
        self.predictors = [x for x in self.train.columns if x not in [self.target, IDcol]]
        print("--------end load data train_test_split--------")

    def getRange(self, start, stop, step):
        listTemp = [start]
        for i in range(start + step, stop, step):
            listTemp.append(i)
        return listTemp

    def fit(self):
        # 第二步：max_depth([默认6],典型值：3-10) 和 min_child_weight([默认1],典型值：3-10) 参数调优
        param_test1 = {
            'max_depth': self.getRange(3, 10, 2),
            'min_child_weight': self.getRange(1, 6, 2)
        }
        gsearch1 = GridSearchCV(
            estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, min_child_weight=1, gamma=0,
                                    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
                                    scale_pos_weight=1, seed=27), param_grid=param_test1, scoring='roc_auc', n_jobs=4,
            iid=False, cv=5)
        gsearch1.fit(self.train[self.predictors],self.train[self.target])
        print("------------------------")
        print(gsearch1.grid_scores_)
        print("------------------------")
        print(gsearch1.best_params_)
        print("------------------------")
        print(gsearch1.best_score_)
        print("Best: %f using %s " % (gsearch1.best_score_, gsearch1.best_params_))
        means = gsearch1.cv_results_["mean_test_score"]
        params = gsearch1.cv_results_["params"]

        for mean, param in zip(means, params):
            print("%f with %r" % (mean, param))

    def getRange(start, stop, step):
        listTemp = [start]
        for i in range(start + step, stop, step):
            listTemp.append(i)
        return listTemp


