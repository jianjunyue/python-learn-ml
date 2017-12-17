from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
#房价预测案例

#Step 1: 检视源数据集
import matplotlib.pyplot as plt
from sklearn import preprocessing
train_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/house_price/train_20171217.csv', index_col=0)
# test_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/house_price/test_20171217.csv', index_col=0)
y_train=train_df["SalePrice"]
X_train=train_df.drop(["SalePrice"], axis=1)


# print(train_df.shape)
# print(test_df.shape)

alphas = np.logspace(-3, 2, 10)
print(alphas)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
alg = Ridge(alpha)
grid_search = GridSearchCV(estimator=alg, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=4,iid=False, cv=5)


plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error");
plt.show()