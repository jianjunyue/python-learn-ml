from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from matplotlib import pyplot
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
# from pinyin import PinYin
import seaborn as sns
def rename_columns(pre_name,columns_name):
    name_dict={}
    for name in columns_name:
        name_dict[name]=pre_name+name
    return name_dict
# 导入数据
filename = '/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_train_20180102.csv'
data = read_csv(filename,header=0,encoding='GB2312')

# for column in data.columns:
#     print(column)
#     print(data[column].unique())

# plt.xticks(range(len(cols)),cols)]
# data.plot.scatter(x="红细胞平均体积", y="血糖", ylim=(0,20));
# plt.scatter(data["年龄"],data["血糖"],c = 'r',marker = 'o')
# plt.xticks(data["年龄"].values) #给X轴赋值名称,没有就默认自适应
# plt.yticks(data["血糖"].values) #给X轴赋值名称,没有就默认自适应
# plt.show()

data['year'] = data["体检日期"].apply(lambda x: x.split('/')[2])
data['year'] = data['year'].astype(int)
data['month'] = data["体检日期"].apply(lambda x: x.split('/')[1])
data['month'] = data['month'].astype(int)
data['day'] = data["体检日期"].apply(lambda x: x.split('/')[0])
data['day'] = data['day'].astype(int)
# print(data["性别"].unique())

pclass_dummies_titanic  = pd.get_dummies(data['性别'])
occ_cols = ['性别_' +  columns_name for columns_name in pclass_dummies_titanic.columns]
pclass_dummies_titanic.rename(columns=rename_columns('性别_',pclass_dummies_titanic.columns), inplace = True)
pclass_dummies_titanic = pclass_dummies_titanic.drop(["性别_??"], axis=1)
data = data.join(pclass_dummies_titanic)


#
test = read_csv("/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_test_A_20180102.csv",header=0,encoding='GB2312')
test['year'] = test["体检日期"].apply(lambda x: x.split('/')[2])
test['year'] = test['year'].astype(int)
test['month'] = test["体检日期"].apply(lambda x: x.split('/')[1])
test['month'] = test['month'].astype(int)
test['day'] = test["体检日期"].apply(lambda x: x.split('/')[0])
test['day'] = test['day'].astype(int)
pclass_dummies_titanic  = pd.get_dummies(test['性别'])
occ_cols = ['性别_' +  columns_name for columns_name in pclass_dummies_titanic.columns]
pclass_dummies_titanic.rename(columns=rename_columns('性别_',pclass_dummies_titanic.columns), inplace = True)
test = test.join(pclass_dummies_titanic)

test = test.drop(["id","体检日期","性别","year"], axis=1)
# # print(test.head())
#
# # 将数据分为输入数据和输出结果
X = data.drop(["id","血糖","体检日期","性别","year"], axis=1)
Y = data["血糖"]
Y=np.log1p(Y)
#
clf=XGBRegressor()
clf.fit(X, Y)
#查看重要程度
feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
# print(feat_imp)
# # feat_imp.plot(kind='bar', title='Feature Importances')
# # plt.show()
#print("---111----")
kfold = KFold(n_splits=10, random_state=7)
test_score = np.sqrt(-cross_val_score(clf, X, Y, cv=kfold, scoring='neg_mean_squared_error'))
print("------test_score--------")
print(test_score)
print(np.mean(test_score))
print("---2----")
clf=XGBRegressor()
clf.fit(X, Y)
print("---3----")
pred=np.expm1(clf.predict(test))
pred_df=pd.DataFrame()
pred_df["pred"]=pred

pred_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/sub_0114_性别_get_dummies.csv',header=False, index=False, float_format='%.3f')

