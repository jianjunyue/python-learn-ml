import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.preprocessing import Imputer
# from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

train_df = pd.read_csv('../../../data/house_price_train.csv')
test_df = pd.read_csv('../../../data/house_price_test.csv')

submission_filename = "../../../data/house_price/sample_Submission"

numeric_cols = train_df.columns[train_df.dtypes != 'object']
numeric_cols=numeric_cols.drop("SalePrice")
x_train =train_df[numeric_cols]
# target=train_df["SalePrice"]
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
y_train=np.log1p(train_df.pop('SalePrice'))
mean_cols = x_train.mean()
x_train=x_train.fillna(mean_cols)
numeric_col_means = x_train.loc[:, numeric_cols].mean()
numeric_col_std = x_train.loc[:, numeric_cols].std()
x_train.loc[:, numeric_cols] = (x_train.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

test_df=test_df.fillna(0)
x_test=test_df[numeric_cols]



#模型融合
class Ensemble(object):
    def __init__(self, n_splits, stacker,  base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    #多模型融合结果预测
    def fit_predict(self, X_train, y_train, X_test):
        y_pred_ensemble = np.zeros((X_test.shape[0], len(self.base_models)))
        stacker_pred_ensemble = np.zeros((X_train.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            try:
               clf.fit(X_train, y_train)
               t=clf.predict(X_test)
               y_pred = clf.predict(X_test)[:]
               # clf.save_model('xg.model')
               # y_pred.to_csv("xg.model.csv")
               y_pred_ensemble[:, i] = y_pred

               stacker_y_pred = clf.predict(X_train)[:]
               stacker_pred_ensemble[:, i] = stacker_y_pred
            except Exception as err:
                print(err)

        lr= self.stacker.fit(stacker_pred_ensemble, y_train)
        print(lr.coef_)
        print(lr.intercept_)
        res = lr.predict(y_pred_ensemble)[:]
        return res

# rf params
rf_params = {}
rf_params['n_estimators'] = 32
rf_params['max_depth'] = 8
rf_params['max_features'] = 0.3
rf_params['min_samples_split'] = 100
rf_params['min_samples_leaf'] = 30

# xgb params
xgb_params = {}
# xgb_params['n_estimators'] = 50
xgb_params['min_child_weight'] = 12
xgb_params['learning_rate'] = 0.37
xgb_params['max_depth'] = 6
xgb_params['subsample'] = 0.77
xgb_params['reg_lambda'] = 0.8
xgb_params['reg_alpha'] = 0.4
xgb_params['base_score'] = 0
# xgb_params['seed'] = 400
xgb_params['silent'] = 1

# ridge params
ridge_params = {}
ridge_params['n_estimators'] = 500
ridge_params['max_features'] = 8

# RF model
rf_model = RandomForestRegressor(**rf_params)

# ridge model
ridge_model = RandomForestRegressor(**ridge_params)

# XGB model
xgb_model = XGBRegressor(**xgb_params)

stack = Ensemble(n_splits=3,
                 stacker=LinearRegression(),
                 base_models=(rf_model,ridge_model,xgb_model))

y_pred_ensemble = stack.fit_predict(x_train, y_train, x_test)


# 按照指定的格式生成结果
def create_submission(ids, predictions, filename=submission_filename):
    submission_df = pd.DataFrame(data={'Id' : ids, 'SalePrice': predictions})

    submission_df.to_csv(submission_filename  ,index=False)

pre_df=pd.DataFrame(np.expm1(y_pred_ensemble),columns=["SalePrice"])
pre_df.loc[pre_df["SalePrice"] <0, "SalePrice"] = 0
create_submission(test_df['Id'], pre_df["SalePrice"])