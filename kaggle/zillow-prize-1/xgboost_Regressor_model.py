from xgb_ParameterTuning import *
from ParameterTuning import *

from xgboost.sklearn import XGBRegressor
#默认 xgb params 默认
xgb_params = {}
xgb_params['learning_rate'] = 0.1
xgb_params['n_estimators'] = 140
xgb_params['max_depth'] = 5
xgb_params['min_child_weight'] = 1
xgb_params['gamma'] = 0
xgb_params['subsample'] = 0.8
xgb_params['colsample_bytree'] = 0.8
xgb_params['objective'] = 'binary:logistic'
xgb_params['seed'] = 27
xgb_params['nthread'] = 4
xgb_params['scale_pos_weight'] = 1


def getRange(start, stop, step):
 listTemp = [start]
 for i in range(start + step, stop, step):
  listTemp.append(i)
 return listTemp

train = pd.read_csv('train_modified_part.csv')
target = 'Disbursed'
IDcol = 'ID'
predictors = [x for x in train.columns if x not in [ target, IDcol]]

param_grid = {
 'max_depth':getRange(3,6,2),
 'min_child_weight':getRange(3,6,2)
}
pt=ParameterTuning()
clf=XGBRegressor(**xgb_params)
pt.fit_Classifier(clf,param_grid,train[predictors],train[target])
print(type(train[target]))
print(train[target])
