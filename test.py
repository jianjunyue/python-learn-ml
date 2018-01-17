import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")
def clean_data(titanic):
    # Sex 预处理
    titanic.loc[titanic["Sex"] == 'male', "Sex"] = 0
    titanic.loc[titanic["Sex"] == 'female', "Sex"] = 1
    titanic["Sex"] = titanic["Sex"].fillna(titanic["Sex"].median())
    # print(titanic["Sex"].value_counts())
    # Age 预处理
    # print(titanic["Age"].value_counts())
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    # SibSp 预处理print(titanic.describe())

    # print(titanic["SibSp"].value_counts())
    # print(titanic["Parch"].value_counts())
    # Embarked 预处理
    # print(titanic["Embarked"].value_counts())
    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 2
    titanic["Embarked"] = titanic["Embarked"].fillna(0)

    return titanic
    # print(titanic["Embarked"].value_counts())
    # print(titanic["Embarked"].unique())
    # print(titanic.describe())
    # print(titanic.head())
    # titanic.info()
    # print (titanic["Embarked"].info())
    # print(titanic.shape)

train_data=clean_data(train)
test_data=clean_data(test)


# "Pclass"-船窗等级,"Sex"-性别,"Age"-年龄,"SibSp"-兄弟姐妹数,"Parch"-老人小孩数,"Fare"-船票,"Embarked"-登船地点
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# print(test_data[predictors].info())

# print(train_data["Embarked"].value_counts())
# print(train_data["Embarked"].unique())
# print(test_data[predictors].describe())
print(test_data[predictors].head())

#特征重要性衡量
from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt

# selector=SelectKBest(f_classif,k=5)
# selector.fit(train_data[predictors],train_data["Survived"])
# scores=-np.log10(selector.pvalues_)
# print(scores)
# plt.bar(range(len(predictors)),scores)
# plt.xticks(range(len(predictors)),predictors,rotation='vertical')
# plt.show()

xgb=XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=3, min_child_weight=1, gamma=0, subsample=0.8,
              colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
param_test1 = {
 'learning_rate':[0.0001,0.001,0.01,0.1,1]
 # 'n_estimators':[i for i in range(40,60,2)]
}
gsearch1 = GridSearchCV(estimator =xgb,param_grid = param_test1,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch1.fit(train[predictors],train["Survived"])
# print("------------------------")
# print(gsearch1.grid_scores_)
# print("------------------------")
# print(gsearch1.best_params_)
# print("------------------------")
# print(gsearch1.best_score_)
xgb.fit(train[predictors],train["Survived"])

predictions=xgb.predict(test_data[predictors])

# print(test_data.head())

gender_submission=pd.read_csv("data/gender_submission.csv")


# result = test_data.DataFrame({'Survived':predictions.astype(np.int32)})
test_data["Survived"]=predictions
result=test_data[["PassengerId","Survived"]]
# print(result.head(10))
# print(predictions)
result.to_csv("data/submission.csv",index=False)

# ----------------------------------------
#http://blog.csdn.net/han_xiaoyang/article/details/50469334
#

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

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()



train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")

def clean_data(titanic):
    # Sex 预处理
    titanic.loc[titanic["Sex"] == 'male', "Sex"] = 0
    titanic.loc[titanic["Sex"] == 'female', "Sex"] = 1
    titanic["Sex"] = titanic["Sex"].fillna(titanic["Sex"].median())
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 2
    titanic["Embarked"] = titanic["Embarked"].fillna(0)
    return titanic

train_data=clean_data(train)
test_data=clean_data(test)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#少样本的情况情况下绘出学习曲线

X=train_data[predictors]
y=train_data["Survived"]

xgb=XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=3, min_child_weight=1, gamma=0, subsample=0.8,
              colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
plot_learning_curve(xgb, "LinearSVC(C=10.0)",X, y, ylim=(0, 1.01),
                    train_sizes=np.linspace(.05, 0.2, 5))
