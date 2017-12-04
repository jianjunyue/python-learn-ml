from FeatureEngineering import *
from ParameterTuning import *
from sklearn.ensemble import RandomForestRegressor
#默认 xgb params 默认
xgb_params = {}
xgb_params['n_estimators'] = 140
xgb_params['max_depth'] = 5
# xgb_params['objective'] = 'binary:logistic'
xgb_params['min_samples_split'] = 2
# xgb_params['nthread'] = 4
# xgb_params['scale_pos_weight'] = 1


def getRange(start, stop, step):
 listTemp = [start]
 for i in range(start + step, stop, step):
  listTemp.append(i)
 return listTemp

fe=FeatureEngineering()
X_train_all, y_train_all=fe.train_test_all_data()
target = 'logerror'
predictors = [x for x in X_train_all.columns ]

param_grid = {
 # 'max_depth':getRange(3,6,2),
 'n_estimators':getRange(400,600,100)
}
pt=ParameterTuning()
clf=RandomForestRegressor(**xgb_params)
print("------start ------")
pt.fit_Regression(clf,param_grid,X_train_all[predictors],y_train_all[target])

print("------end ------")