import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('../../data/kaggle_bike_competition_train.csv',header = 0)
# 把月、日、和 小时单独拎出来，放到3列中
df_train['month'] = pd.DatetimeIndex(df_train.datetime).month
df_train['dayofweek'] = pd.DatetimeIndex(df_train.datetime).dayofweek
df_train['days_in_month'] = pd.DatetimeIndex(df_train.datetime).days_in_month
df_train['hour'] = pd.DatetimeIndex(df_train.datetime).hour
df_train=df_train.drop(['datetime','casual','registered'], axis = 1)

df_train_data=df_train.drop(['count'],axis = 1).values
df_train_target = df_train['count'].values

print(df_train_data)
print(df_train_target)

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score

# 总得切分一下数据咯（训练集和测试集）
cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=3, test_size=0.2,
                                   random_state=0)

# 各种模型来一圈 -- 模型选择

print("岭回归")
for train, test in cv:
    svc = linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))

print("支持向量回归/SVR(kernel='rbf',C=10,gamma=.001)")
for train, test in cv:
    svc = svm.SVR(kernel='rbf', C=10, gamma=.001).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))

print("随机森林回归/Random Forest(n_estimators = 100)")
for train, test in cv:
    svc = RandomForestRegressor(n_estimators=100).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))


#参数咋调啊？我们有一个工具可以帮忙，叫做GridSearch
X = df_train_data
y = df_train_target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=0)

tuned_parameters = [{'n_estimators': [10, 100, 500]}]

scores = ['r2']

for score in scores:

    print(score)

    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("别！喝！咖！啡！了！最佳参数找到了亲！！：")
    print
    ""
    # best_estimator_ returns the best estimator chosen by the search
    print(clf.best_estimator_)
    print
    ""
    print("得分分别是:")
    print
    ""
    # grid_scores_的返回值:
    #    * a dict of parameter settings
    #    * the mean score over the cross-validation folds
    #    * the list of scores for each fold
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print
    ""

#模型状态是不是，过拟合or欠拟合 依旧是学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves (Random Forest, n_estimators = 100)"
cv = cross_validation.ShuffleSplit(df_train_data.shape[0], n_iter=10,test_size=0.2, random_state=0)
estimator = RandomForestRegressor(n_estimators = 100)
plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=4)

plt.show()