import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectKBest,f_classif,f_regression
import matplotlib.pylab as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [2, 5, 8, 11, 0]
intersection = list(set(a).intersection(set(b)))
print(intersection)
# train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
train_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练处理get_dummies.csv',header=0,encoding='utf-8')

predict_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A.xlsx',header=0,encoding='utf-8')
predict_df = pd.get_dummies(predict_df)
quantity_pre_1 = [attr for attr in predict_df.columns if predict_df.dtypes[attr] != 'object']
quantity_1 = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
# print(type(quantity))
quantity = list(set(quantity_pre_1).intersection(set(quantity_1)))
# print(train_df.columns)
# quality = [attr for attr in train_df.columns if train_df.dtypes[attr] == 'object']  # 类型变量集合
# print(train_df[quantity].head(5))
# y_train=train_df["Y"]
y_train=np.log1p(train_df["Y"])
print("---1----")
train_df=train_df.drop(["Y"], axis=1)
print("---11----")
# quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
train_df=train_df[quantity]
# X_train = Imputer().fit_transform(train_df)
X_train=train_df
X_train=X_train.fillna(0)
print(np.isnan(X_train).any())
print("---111----")
xgb1 = XGBRegressor()
xgb1.fit(X_train, y_train)
print("---2----")
feat_imp = pd.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)


def sort_weight_values(X, y, predictors):
    selector = SelectKBest(f_regression, k=5)
    selector.fit(X, y)
    scores = -np.log10(selector.pvalues_)
    dt = pd.DataFrame()
    dt["predictors"] = predictors
    dt["scores"] = scores
    dt = dt.sort_values(by='scores', axis=0, ascending=False)
    dt.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/特征重要性_SelectKBest.csv', header=False, index=False)
    print("------------------------------------------------")
    print("特征重要性衡量:")
    print(dt.head())
    print("------------------------------------------------")
    plt.bar(range(len(predictors)), dt["scores"])
    plt.xticks(range(len(predictors)), dt["predictors"], rotation='vertical')
    plt.show()

sort_weight_values(X_train, y_train,quantity)

# def group_values(value_counts):
#     dict = value_counts.to_dict()
#     df = pd.DataFrame(columns=['keyid', 'count'])
#     listkey = []
#     listcount = []
#     for key in dict:
#         listkey.append(key)
#         listcount.append(dict[key])
#         # df.loc[df.shape[0] + 1] = {'keyid': key, 'count': dict[key]}
#     df['keyid']=listkey
#     df['count']=listcount
#     return df
# df=group_values(feat_imp)
# df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/特征重要性.csv',header=False, index=False)
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')
# plt.show()
print(feat_imp)
# predict_df=predict_df.drop(["ID"],axis=1)
predict_df=predict_df[quantity]
# predict_df = Imputer().fit_transform(predict_df)
print("---3----")
# pred=xgb1.predict(predict_df)
pred=np.expm1(xgb1.predict(predict_df))
pred_df=pd.DataFrame()
pred_df["pred"]=pred

pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv', index=False, float_format='%.9f')
print("---4----")
submission_df=pd.DataFrame()

submission_iddf = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板.csv')
pred_df_TEMP = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_pred.csv')
submission_df["id"]=submission_iddf["id"]
submission_df["pred"]=pred_df_TEMP["pred"]
print(submission_df.head(3))
submission_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub_1229.csv',header=False, index=False, float_format='%.9f')