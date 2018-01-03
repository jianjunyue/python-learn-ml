import numpy as np
import pandas as pd
import xlrd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

#https://github.com/demonicCode/Intelligent-manufacturing/blob/master/Imanufactur.py

#### calculate miss values
def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col','missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df

#### obtain cols of XX type
def obtain_x(train_df,xtype):
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ['col','type']
    return dtype_df[dtype_df.type==xtype].col.values

def date_cols(train_df,float_col):
    float_date_col = []
    for col in float_col:
        if train_df[col].min() > 1e13:
            float_date_col.append(col)
    return float_date_col

def float_uniq(float_df,float_col):
    float_uniq_col = []
    for col in tqdm(float_col):
        uniq = float_df[col].unique()
        if len(uniq) == 1:
            float_uniq_col.append(col)
    return float_uniq_col

def cal_corrcoef(float_df,y_train,float_col):
    corr_values = []
    for col in float_col:
        corr_values.append(abs(np.corrcoef(float_df[col].values,y_train)\
                [0,1]))
    corr_df = pd.DataFrame({'col':float_col,'corr_value':corr_values})
    corr_df = corr_df.sort_values(by='corr_value',ascending=False)
    return corr_df

def build_model(x_train,y_train):
    reg_model = LinearRegression()
    reg_model.fit(x_train,y_train)
    return reg_model
