import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
#matplotlib inline
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
train = pd.read_csv('train_modified_part.csv')
print( train['Disbursed'].value_counts())
target='Disbursed'
IDcol = 'ID'

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds,
                                                    scoring='roc_auc')

    # Print model report:
    print("Model Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
        np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance:
    if printFeatureImportance:
        print(alg.feature_importances_)
        print(predictors)
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
# print(predictors)
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train, predictors)

list1=[10]
for i in range(50,100,10):
    list1.append(i)

def getRange(start,stop,step):
    listTemp = [start]
    for i in range(start+step, stop, step):
        listTemp.append(i)
    return listTemp

#Choose all predictors except target & IDcols
# predictors = [x for x in train.columns if x not in [target, IDcol]]
# param_grid_para = {'n_estimators':list1}
# model=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10)
# gsearch1 = GridSearchCV(estimator = model,param_grid=param_grid_para, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(train[predictors],train[target])
# print(gsearch1.grid_scores_)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)

tmp=getRange(5,16,2)
type(tmp)
print(tmp)
#Grid seach on subsample and max_features
param_test2 = {'max_depth':getRange(5,16,2), 'min_samples_split':getRange(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,
                                                max_features='sqrt', subsample=0.8, random_state=10),
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
print("---------------------------")
print(gsearch2.grid_scores_)
print("---------------------------")
print(gsearch2.best_params_)
print("---------------------------")
print(gsearch2.best_score_)