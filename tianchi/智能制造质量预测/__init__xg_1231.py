import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,StratifiedKFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

from sklearn.learning_curve import learning_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
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
# quantity_pre_1 = [attr for attr in predict_df.columns if predict_df.dtypes[attr] != 'object']
# quantity_1 = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
# # print(type(quantity))
# quantity = list(set(quantity_pre_1).intersection(set(quantity_1)))

df_quantity_temp = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/特征重要性_SelectKBest.csv',header=0,encoding='utf-8')
df_quantity=df_quantity_temp[df_quantity_temp["value"]>1]
quantity=df_quantity["key"]
# print(quantity)
# print(train_df.columns)
# quality = [attr for attr in train_df.columns if train_df.dtypes[attr] == 'object']  # 类型变量集合
# print(train_df[quantity].head(5))
# y_train=train_df["Y"]
y_train=np.log1p(train_df["Y"])
print("---1----")
train_df=train_df.drop(["Y"], axis=1)
print("---11----")
# quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
X_train=train_df[quantity]
# X_train = Imputer().fit_transform(train_df)
X_train=train_df
print("---111----")
xgb1 = XGBRegressor(seed=10)
# kfold = StratifiedKFold(n_folds=5, shuffle=True, random_state=10)
kfold=KFold(n_splits=5,random_state=7)
test_score = np.sqrt(-cross_val_score(xgb1, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error'))
print("------test_score--------")
print(len(quantity))
print(test_score)
print(np.mean(test_score))

params =[500,600,700,800,900,1000,1100,1200]
test_scores = []
for param in params:
    print("------param--------")
    print(param)
    df_quantity_param = df_quantity_temp.head(param)
    quantity = df_quantity_param["key"]
    print(len(quantity))
    X_train = train_df[quantity]
    clf = XGBRegressor(seed=10)
    # kfold = StratifiedKFold(n_folds=5, shuffle=True, random_state=10)
    kfold = KFold(n_splits=5, random_state=7)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
    print(np.mean(test_score))


df_quantity_param = df_quantity_temp[df_quantity_temp["value"] > 2]
quantity = df_quantity_param["key"]
print(len(quantity))
X_train = train_df[quantity]
train_sizes, train_loss, test_loss = learning_curve(XGBRegressor(seed=10),
                                                    X_train, y_train, cv=5, scoring='neg_mean_squared_error',
                                                        train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
# 训练误差均值
train_loss_mean = -np.mean(train_loss, axis = 1)
# 测试误差均值
test_loss_mean = -np.mean(test_loss, axis = 1)

plt.plot(train_sizes, train_loss_mean, 'o-', color = 'r', label = 'Training')
plt.plot(train_sizes, test_loss_mean, 'o-', color = 'g', label = 'Cross-Validation')
# plt.plot(params, test_scores)
plt.title("quantity vs CV Error");
plt.show()

xgb1 = XGBRegressor(seed=10)
df_quantity_param = df_quantity_temp[df_quantity_temp["value"] > 2]
quantity = df_quantity_param["key"]
print(len(quantity))
X_train = train_df[quantity]
xgb1.fit(X_train, y_train)
print("---2----")
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
submission_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A-答案模板_sub_1230.csv',header=False, index=False, float_format='%.9f')