from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
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

#交叉验证，获取最优参数( 最终使用GridSearch获取最优参数 )
# max_features = [0.2,0.3,0.4,0.45,0.5,0.55]
# test_scores = []
# print("test")
# for max_feat in max_features:
#     clf = RandomForestRegressor(n_estimators = 200,max_features = max_feat)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 5,scoring = 'neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
#     print(test_score)
# plt.plot(max_features,test_scores)
# plt.title('Max Features vs CV Error')
# plt.show()

# Choose some parameter combinations to try
parameters = {
              # 'n_estimators': [40, 60, 90],
             'max_features': ['log2', 'sqrt','auto',0.4]
              # 'max_features': ['log2', 'sqrt','auto'],
              # 'criterion': ['entropy', 'gini'],
              # 'max_depth': [2, 3, 5, 10],
              # 'min_samples_split': [2, 3, 5],
              # 'min_samples_leaf': [1,5,8]
             }
alg = RandomForestRegressor(n_estimators = 200)
grid_search = GridSearchCV(estimator=alg, param_grid=parameters, scoring='neg_mean_squared_error', n_jobs=4, iid=False, cv=5)
grid_result = grid_search.fit(X_train, y_train)
print(grid_result.grid_scores_)
print(grid_result.best_params_)
print(grid_result.best_score_)
# print("Best: %f using %s " % (grid_result.best_score_, grid_result.best_params_))
# train_means = grid_result.cv_results_["mean_train_score"]
# test_means = grid_result.cv_results_["mean_test_score"]
# params = grid_result.cv_results_["params"]
#
# print("------------alg.feature_importances_------------")
# print(alg.feature_importances_)
#
# for train_mean, test_mean, param in zip(train_means, test_means, params):
#     print("%f , %f with %r" % (train_mean, test_mean, param))