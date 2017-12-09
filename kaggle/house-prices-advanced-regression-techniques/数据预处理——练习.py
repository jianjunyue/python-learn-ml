import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
from scipy import stats
from sklearn import preprocessing


train_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/house_price_train.csv",index_col = 0)
test_df = pd.read_csv("/Users/jianjun.yue/PycharmGItHub/data/house_price/house_price_test.csv",index_col = 0)
# print(train_df.info())
train_df = train_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'], axis=1)
train_df["GarageType"] = train_df["GarageType"].fillna('None')
train_df["GarageFinish"] = train_df["GarageFinish"].fillna('None')
train_df["GarageQual"] = train_df["GarageQual"].fillna('None')
train_df["GarageYrBlt"] = train_df["GarageYrBlt"].fillna("0").astype('int')  #.apply(lambda x: x*2)
train_df["GarageCond"] = train_df["GarageCond"].fillna('None')
train_df["BsmtFinType2"] = train_df["BsmtFinType2"].fillna('None')
train_df["BsmtExposure"] = train_df["BsmtExposure"].fillna('None')
train_df["BsmtQual"] = train_df["BsmtQual"].fillna('None')
train_df["BsmtFinType1"] = train_df["BsmtFinType1"].fillna('None')
train_df["BsmtCond"] = train_df["BsmtCond"].fillna('None')
train_df["MasVnrType"] = train_df["MasVnrType"].fillna('None')
train_df["Electrical"] = train_df["Electrical"].fillna('None')
train_df["MasVnrArea"] = train_df["MasVnrArea"].fillna(0.)
# print(train_df["GarageType"].describe())
# print(train_df["GarageType"].value_counts())
# columns=["GarageType","GarageFinish","GarageQual","GarageYrBlt","GarageCond","BsmtFinType2","BsmtExposure","BsmtQual","BsmtFinType1","BsmtCond","MasVnrType","MasVnrArea","Electrical"]
# for c in columns:
    # print(c,train_df.dtypes[c])
na_count = train_df.isnull().sum().sort_values(ascending=False)
# print(na_count.head(20))
na_rate = na_count / len(train_df)
# print(na_count.type)
na_data = pd.concat([na_count,na_rate ],axis=1,keys=[ 'count','ratio' ])
# print(na_data.head(20))
# df_train = train_df.drop(na_data[na_data['count']>1].index, axis=1)  # 删除上述前18个特征
# df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)  # 删除 Electrical 取值丢失的样本
train_df.shape  # 缺失值处理后的数据大小：1459个样本，63个特征
# print(df_train.head(5))
df_y = train_df['SalePrice']
# print(train_df.columns)
# train_df = train_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'], axis=1)
# print(df_train.head(5))
quantity = [attr for attr in train_df.columns if train_df.dtypes[attr] != 'object']  # 数值变量集合
quality = [attr for attr in train_df.columns if train_df.dtypes[attr] == 'object']  # 类型变量集合
na_count = train_df.isnull().sum().sort_values(ascending=False)
# print(train_df[quality].head(5))
le = preprocessing.LabelEncoder()
for c in quality:
    train_df[c] = le.fit_transform(train_df[c])

print(train_df.head())

#各特征与房价的相关性分析
def spearman(frame, features):
    '''
    采用“斯皮尔曼等级相关”来计算变量与房价的相关性(可查阅百科)
    此相关系数简单来说，可以对上述encoder()处理后的等级变量及其它与房价的相关性进行更好的评价（特别是对于非线性关系）
    '''
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(8, 0.15*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')
    plt.show()
# features = quantity + quality_encoded
spearman(train_df, train_df.columns)

