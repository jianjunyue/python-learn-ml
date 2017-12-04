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

train_df = pd.read_csv(train_filename)
test_df = pd.read_csv(test_filename)
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

# print(test_df['id'].astype(np.uint64))

#模型融合
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
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

y_test = stack.fit_predict(x_train, y_train, x_test)

# 按照指定的格式生成结果
def create_submission(ids, predictions, filename=submission_filename):
    # submission_df = pd.DataFrame({"id": ids, "click": predictions})
    submission_df = pd.DataFrame(data={'aid' : ids, 'click': predictions})
    print(submission_df.head())
    # submission_df.to_csv(submission_filename+"_sub", header=['id', 'click'], index=False)
    submission_df.to_csv(submission_filename + "_sub",index=False)

pre_df=pd.DataFrame(y_test,columns=["click"])
create_submission(test_df['id'].astype(np.uint64), pre_df["click"])