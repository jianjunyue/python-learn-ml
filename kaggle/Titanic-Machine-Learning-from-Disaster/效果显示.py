import pandas as pd
import numpy as np
from mlens.ensemble import SuperLearner
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator

from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import  SelectKBest,f_classif
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import re
from scipy.stats import randint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train.csv"
path="/Users/jianjun.yue/PycharmGItHub/data/titanic/train_pre.csv"
data=pd.read_csv(path)
print("--------------RandomForestClassifier---------------")
predictors=["Pclass","Sex","Age","SibSp","Parch","Fare_scaler","Embarked","NameLength","Title"]
train=data[predictors]
y=data["Survived"]
seed = 2017

def getRange(start,stop,step):
    listTemp = [start]
    for i in range(start+step, stop, step):
        listTemp.append(i)
    return listTemp

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=1)

from mlens.visualization import corrmat
# Generate som different predictions to correlate
params = [0.1, 0.3, 1.0, 3.0, 10, 30]
preds = y_test
for i, c in enumerate(params):
    preds[:, i] = LogisticRegression(C=c).fit(X_train, y_train).predict(X_test)

corr = pd.DataFrame(preds, columns=['C=%.1f' % i for i in params]).corr()
corrmat(corr)
plt.show()