from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from matplotlib import pyplot
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import MinMaxScaler

# 导入数据
filename = '/Users/jianjun.yue/PycharmGItHub/data/人工智能辅助糖尿病遗传风险预测/d_train_20180102.csv'
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename,header=0,encoding='GB2312')
data['year'] = data["体检日期"].apply(lambda x: x.split('/')[2])
data['year'] = data['year'].astype(int)
data['month'] = data["体检日期"].apply(lambda x: x.split('/')[1])
data['month'] = data['month'].astype(int)
data['day'] = data["体检日期"].apply(lambda x: x.split('/')[0])
data['day'] = data['day'].astype(int)
data.loc[data["性别"] == '女', "性别"] = 0
data.loc[data["性别"] == '男', "性别"] = 1
data.loc[data["性别"] == '??', "性别"] = 2
# print(data["性别"].unique())
data['性别'] = data['性别'].astype(int)
data=data.fillna(0)
# print(data.head())
# quantity_pre_1 = [attr for attr in data.columns if data.dtypes[attr] == 'object']
# print(quantity_pre_1)
# 将数据分为输入数据和输出结果
X = data.drop(["id","血糖","体检日期"], axis=1)
Y = data["血糖"]
# data_scaler = MinMaxScaler(feature_range=(0, 1))
# Y = data_scaler.fit_transform(data["血糖"].values.reshape(-1, 1)).ravel()
temp_df = pd.DataFrame()
temp_df["log1pY"]=np.log1p(Y)
print(min(np.log1p(Y)))
print(max(np.log1p(Y)))
temp_df["log1pYlog1pY"]=np.log1p(np.log1p(Y))

print(min(np.log1p(np.log1p(Y))))
print(max(np.log1p(np.log1p(Y))))
temp_df["log"]=np.log(data["血糖"])
temp_df.hist(bins=200)
plt.title("test title")
plt.xlabel('test X');
plt.ylabel('test Y');
plt.show()
Y=np.log1p(np.log1p(Y))
num_folds = 5
seed = 7
models = {}
models['DT'] = DecisionTreeRegressor()
models['EN'] = ElasticNet()
models['KNR'] = KNeighborsRegressor()
models['LS'] = Lasso()
models['LR'] = LinearRegression()
models['RD'] = Ridge()
models['SVR'] = SVR()
models['RF'] = RandomForestRegressor()
models['XG'] = XGBRegressor()

test_scores = []
kfold = KFold(n_splits=num_folds, random_state=seed)
for name in models:
    result = cross_val_score(models[name], X, Y, cv=kfold,scoring='neg_mean_squared_error')
    test_score = np.sqrt(-result)
    test_scores.append(np.mean(test_score))
    msg = '%s: %.3f' % (name,np.mean(test_score))
    print(msg)

# 图表显示

# plt.figure(figsize=(8,6))
# plt.plot(range(len(test_scores)),  test_scores)
# plt.scatter(range(len(test_scores)), test_scores,s=30,c='blue',marker='x',alpha=0.5,label='C2')
# plt.xticks(range(len(test_scores)),models.keys()) #给X轴赋值名称
# plt.legend(loc='upper right')
# plt.title("test title")
# plt.xlabel('test X');
# plt.ylabel('test Y');
# plt.show()