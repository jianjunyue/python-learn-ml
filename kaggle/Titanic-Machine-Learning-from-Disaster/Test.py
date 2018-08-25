import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlens.visualization import exp_var_plot
from sklearn.decomposition import PCA
from mlens.visualization import pca_plot
from sklearn.ensemble import RandomForestClassifier

path_test="/Users/jianjun.yue/PycharmGItHub/data/titanic/test_pre.csv"
path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data=pd.read_csv(path)
data_test=pd.read_csv(path_test)
print("--------------RandomForestClassifier---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare_scaler","Embarked","NameLength"]
train=data[predictors]
X=train
y=data["Survived"]
X_submission=data_test[predictors]
print(X_submission.head())
print(X_submission.describe())

from mlens.visualization import pca_plot
from sklearn.decomposition import PCA
pca_plot(X, RandomForestClassifier(), y=y)

# from mlens.visualization import corr_X_y
# Z = pd.DataFrame(X, columns=['feature_%i' % i for i in range(1, 5)])
# corr_X_y(Z, y, 2, no_ticks=False)
plt.show()