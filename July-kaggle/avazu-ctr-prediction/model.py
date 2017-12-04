#encoding:utf-8 #
import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# Initial setup
train_filename = "../../../data/avazu-ctr-prediction/train_small.csv"
test_filename = "../../../data/avazu-ctr-prediction/test"
submission_filename = "../../../data/avazu-ctr-prediction/sampleSubmission"

train_df = pd.read_csv(train_filename,dtype={'id':pd.np.string_})
test_df = pd.read_csv(test_filename,dtype={'id':pd.np.string_})

tcolumns="year,month,day,hours,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21".split(",")
def get_data(data):
    hour =data["hour"]
    data["hours"]=(hour%100).astype(np.uint32)
    hour=hour//100
    data["day"]=(hour%100).astype(np.uint32)
    hour = hour // 100
    data["month"]=(hour%100).astype(np.uint32)
    hour = hour // 100
    data["year"]=(hour%100).astype(np.uint32)
    for c in tcolumns:
        if data[c].dtype=="object":
            lbl = LabelEncoder()
            lbl.fit(list(data[c].values))
            data[c] = lbl.transform(list(data[c].values))

    return data

train_df= get_data(train_df)
test_df= get_data(test_df)
x_train=train_df[tcolumns]
y_train=train_df[["click"]]
x_test=test_df[tcolumns]

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
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)[:]
            # clf.save_model('xg.model')
            # y_pred.to_csv("xg.model.csv")
            y_pred_ensemble[:, i] = y_pred

            stacker_y_pred = clf.predict(X_train)[:]
            stacker_pred_ensemble[:, i] = stacker_y_pred

        self.stacker.fit(stacker_pred_ensemble, y_train)
        res = self.stacker.predict(y_pred_ensemble)[:]
        return res

# rf params
rf_params = {}
rf_params['n_estimators'] = 32
rf_params['max_depth'] = 8
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

# RF model
rf_model = RandomForestRegressor(**rf_params)

# XGB model
xgb_model = XGBRegressor(**xgb_params)

stack = Ensemble(n_splits=3,
                 stacker=LinearRegression(),
                 base_models=(xgb_model,rf_model))

y_pred_ensemble = stack.fit_predict(x_train, y_train, x_test)


# 按照指定的格式生成结果
def create_submission(ids, predictions, filename=submission_filename):
    submission_df = pd.DataFrame(data={'aid' : ids, 'click': predictions})
    print(submission_df.info())
    submission_df.to_csv(submission_filename  ,index=False)

pre_df=pd.DataFrame(y_pred_ensemble,columns=["click"])
pre_df.loc[pre_df["click"] <0, "click"] = 0
create_submission(test_df['id'].astype(np.uint64), pre_df["click"])