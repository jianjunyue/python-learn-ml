from sklearn import model_selection
from tpot import TPOTRegressor
import numpy as np
import pandas as pd

train_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练处理get_dummies.csv',header=0,encoding='utf-8')

y_train=np.log1p(train_df["Y"])
train_df=train_df.drop(["Y"], axis=1)

X_trian, X_test, Y_train, Y_test = model_selection.train_test_split(train_df, y_train, test_size=0.2)

tpot = TPOTRegressor(generations=6, verbosity=2)
tpot.fit(X_trian, Y_train)
tpot.score(X_test, Y_test)
# 导出
tpot.export('pipeline.py')