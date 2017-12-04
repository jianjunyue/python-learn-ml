from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from FeatureEngineering import *

class ModelEnsemble(object):
    def __init__(self):
        stack = FeatureEngineering()
        self. X_train, self.X_test, self.y_train_temp, self.y_test_temp = stack.train_test_split()
        self.y_train=self.y_train_temp.values.ravel()
        print(type(self.y_train_temp.values))
        print(type(self.y_train))
        self.y_test=self.y_test_temp.values.ravel()
        print("--------end load data train_test_split--------")

    def run_model(self,clf_key,clf):
        # print('the classifier is :', clf_key)
        # self.try_different_method(clf)
        loss_error=self.predict_ls(clf_key, clf)
        return loss_error

    def run(self):
        clfs = {'xgb': XGBRegressor(**self.xgb_params),
            'rf': RandomForestRegressor(**self.rf_params),
            'et': ExtraTreesRegressor(),
            # 'svr': SVR(kernel='rbf', C=1.0, epsilon=0.05),
            'dt': DecisionTreeRegressor(),
            'ada': AdaBoostRegressor()
            }

        for clf_key in clfs.keys():
            print('the classifier is :', clf_key)
            clf = clfs[clf_key]
            self.try_different_method(clf)
            self.predict_ls(clf_key, clf)

    def try_different_method(self,clf):
        clf.fit(self.X_train, self.y_train)
        score = clf.score(self.X_test, self.y_test)
        print('the score is :', score)
        results = cross_val_score(clf, self.X_test, self.y_test, cv=5, scoring='r2')
        print("clf score: %.4f (%.4f)" % (results.mean(), results.std()))

    def predict_ls(self,clf_key, clf):
        clf.fit(self.X_train, self.y_train)
        y_pre = clf.predict(self.X_test)
        # stacker_pred_ensemble_df[clf_key] = y_pre
        loss_error = self.def_loss_score(y_pre, self.y_test)
        print("loss_error:", loss_error)
        return loss_error

    def def_loss_score(self,y_pre,y):
        loss_error = 0;
        z = zip(y_pre, y)
        for py, ty in z:
            loss_error += abs(py-ty)
        return loss_error / len(y)

    # rf params
    rf_params = {}
    rf_params['n_estimators'] = 50
    rf_params['max_depth'] = 8
    rf_params['min_samples_split'] = 100
    rf_params['min_samples_leaf'] = 30

    # xgb params
    xgb_params = {}
    xgb_params['n_estimators'] = 50
    xgb_params['min_child_weight'] = 12
    xgb_params['learning_rate'] = 0.27
    xgb_params['max_depth'] = 6
    xgb_params['subsample'] = 0.77
    xgb_params['reg_lambda'] = 0.8
    xgb_params['reg_alpha'] = 0.4
    xgb_params['base_score'] = 0
    # xgb_params['seed'] = 400
    xgb_params['silent'] = 1

    # lgb params
    lgb_params = {}
    lgb_params['n_estimators'] = 50
    lgb_params['max_bin'] = 10
    lgb_params['learning_rate'] = 0.321  # shrinkage_rate
    lgb_params['metric'] = 'l1'  # or 'mae'
    lgb_params['sub_feature'] = 0.34
    lgb_params['bagging_fraction'] = 0.85  # sub_row
    lgb_params['bagging_freq'] = 40
    lgb_params['num_leaves'] = 512  # num_leaf
    lgb_params['min_data'] = 500  # min_data_in_leaf
    lgb_params['min_hessian'] = 0.05  # min_sum_hessian_in_leaf
    lgb_params['verbose'] = 0
    lgb_params['feature_fraction_seed'] = 2
    lgb_params['bagging_seed'] = 3