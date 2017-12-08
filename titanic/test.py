from sklearn.learning_curve import learning_curve
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
#matplotlib inline
from matplotlib.pylab import rcParams

#绘制学习曲线，以确定模型的状况
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print(train_sizes)
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    # if ylim:
    #     plt.ylim(ylim)
    plt.title(title)
    plt.show()



train=pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/titqnic.csv")
# test=pd.read_csv("data/test.csv")

def clean_data(titanic):
    # Sex 预处理
    # titanic.loc[titanic["Sex"] == 'male', "Sex"] = 0
    # titanic.loc[titanic["Sex"] == 'female', "Sex"] = 1
    # titanic["Sex"] = titanic["Sex"].fillna(titanic["Sex"].median())
    # titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    # titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
    # titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 1
    # titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 2
    # titanic["Embarked"] = titanic["Embarked"].fillna(0)
    return titanic

train_data=clean_data(train)
# test_data=clean_data(test)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#少样本的情况情况下绘出学习曲线

X=train_data[predictors]
y=train_data["Survived"]
print(np.linspace(.05, 0.2, 5))
xgb=XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=3, min_child_weight=1, gamma=0, subsample=0.8,
              colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
plot_learning_curve(xgb, "LinearSVC(C=10.0)",X, y, ylim=(0, 1.01),
                    train_sizes=np.linspace(.05, 0.2, 5))