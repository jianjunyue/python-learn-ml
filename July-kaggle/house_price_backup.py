
#房价预测案例

#Step 1: 检视源数据集
import numpy as np
import pandas as pd

train_df = pd.read_csv('../../data/house_price_train.csv', index_col=0)
test_df = pd.read_csv('../../data/house_price_test.csv', index_col=0)
# print(train_df.describe())
# print(train_df.info())
# print(train_df.count())

### 回归问题  - 数据平滑化 log1p, 也就是 log(x+1) ！！！！
### 对于没有大小关系的特征值，转化成字符串类型。all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
# print(prices.hist())

y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pd.concat((train_df, test_df), axis=0)
print(all_df.shape)
print(all_df['MSSubClass'].dtypes)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
print(all_df['MSSubClass'].value_counts())
# print(pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head())
# print(all_df.head())
all_dummy_df = pd.get_dummies(all_df)
# print(all_dummy_df.head())
# print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols) #用平均值来填满这些空缺
numeric_cols = all_df.columns[all_df.dtypes != 'object']


# 数据归一化
#计算标准分布：(X-X')/s
#注意：我们这里也是可以继续使用Log的，我只是给大家展示一下多种“使数据平滑”的办法。
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

#Step 4: 建立模型
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

X_train = dummy_train_df.values
X_test = dummy_test_df.values

alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

print(test_scores)
import matplotlib.pyplot as plt
plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error");
plt.show()

from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error");
plt.show()

#Step 5: Ensemble

ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))

y_final = (y_ridge + y_rf) / 2

#Step 6: 提交结果
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})







