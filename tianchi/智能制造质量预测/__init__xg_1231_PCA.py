import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,StratifiedKFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

from sklearn.learning_curve import learning_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
# train_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练.xlsx',header=0,encoding='utf-8')
train_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练处理get_dummies.csv',header=0,encoding='utf-8')
y_train=np.log1p(train_df["Y"])
train_df=train_df.drop(["Y"], axis=1)
train_df = train_df.fillna(0)
# predict_df = pd.read_excel('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/测试A.xlsx',header=0,encoding='utf-8')
# 特征选定
pca = PCA(n_components=20)
fit = pca.fit(train_df)
print("解释方差：%s" % fit.explained_variance_ratio_)
print(fit.components_)
pca_df=pd.DataFrame(fit.components_)
pca_df.to_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练处理get_dummies_PCA.csv',header=False, index=False)
pca_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/智能制造质量预测/训练处理get_dummies_PCA.csv',header=0,encoding='utf-8')
print(pca_df.shape)

print(pca_df.head(3))