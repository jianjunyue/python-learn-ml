
#房价预测案例

#Step 1: 检视源数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
train_df = pd.read_csv('../../data/house_price_train.csv', index_col=0)
test_df = pd.read_csv('../../data/house_price_test.csv', index_col=0)
# print(train_df.describe())
# print(train_df.info())
# print(train_df.count())

### 回归问题  - 数据平滑化 log1p, 也就是 log(x+1) ！！！！
### 对于没有大小关系的特征值，转化成字符串类型。all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

numeric_cols = train_df.columns[train_df.dtypes != 'object']
numeric_cols=numeric_cols.drop("SalePrice")
# print(train_df[numeric_cols].info())
train =train_df[numeric_cols]
target=train_df["SalePrice"]
# X_scaled = preprocessing.scale(train_df["SalePrice"])
# print(X_scaled)
# plt.bar(train_df["SalePrice"])
# plt.title("Max Features vs CV Error");
# plt.show()
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
# prices.hist()
# plt.show()

target=np.log1p(train_df.pop('SalePrice'))



mean_cols = train.mean()
train=train.fillna(mean_cols)

numeric_col_means = train.loc[:, numeric_cols].mean()
numeric_col_std = train.loc[:, numeric_cols].std()
train.loc[:, numeric_cols] = (train.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
# print(train.head())
alphas = np.logspace(-3, 2, 50)
test_scores = []
scores_std=[]
for alpha in alphas:
    clf = Ridge(alpha)
    print(alpha)
    # this_scores = cross_val_score(clf, train, target,cv=10, scoring='neg_mean_squared_error')
    # test_score = np.sqrt(-this_scores)
    # test_scores.append(np.mean(test_score))
    # scores_std.append(np.std(this_scores))
    test_score = np.sqrt(-cross_val_score(clf, train, target, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error");
plt.show()
# train=train.values
# target=target.values
from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, train, target, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error");
plt.show()
#
# print(train)
# print("------------------------")
# print(target)