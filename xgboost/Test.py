
from xgb_ParameterTuning import *

xbpt=xgb_ParameterTuning()
xbpt.fit()
# def getRange(start,stop,step):
#     listTemp = [start]
#     for i in range(start+step, stop, step):
#         listTemp.append(i)
#     return listTemp
#
# train = pd.read_csv('train_modified_part.csv')
# target = 'Disbursed'
# IDcol = 'ID'
# predictors = [x for x in train.columns if x not in [target, IDcol]]
# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27),param_grid = param_test1,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
# gsearch1.fit(train[predictors],train[target])
#
# print("------------------------")
# print(gsearch1.grid_scores_)
# print("------------------------")
# print(gsearch1.best_params_)
# print("------------------------")
# print(gsearch1.best_score_)