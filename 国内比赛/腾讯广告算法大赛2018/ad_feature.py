import numpy as np
import pandas as pd



ad_df = pd.read_csv('/Users/jianjun.yue/PycharmGItHub/data/国内比赛/腾讯广告算法大赛2018/adFeature.csv',header=0,encoding='utf-8')
# print(ad_df.head())
# print(ad_df.query("advertiserId==79"))
print("----------------")
print(ad_df.groupby('productType').size())