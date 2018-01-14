import pandas as pd
import numpy as np
from sklearn import preprocessing

#使用sklearn做单机特征工程
#https://www.cnblogs.com/jasonfreak/p/5448385.html

def rename_columns(pre_name,columns_name):
    name_dict={}
    for name in columns_name:
        name_dict[name]=pre_name+name
    return name_dict

df=pd.DataFrame()
value1=["1","2","3"]
value2=["mall","room","group"]
df["id"]=value1
df["gender"]=value2
print(df)
print("------")
lb = preprocessing.LabelBinarizer()
# gender 改为 0-1 数值
# df["gender"]= lb.fit_transform(df['gender'])
pclass_dummies_titanic  = pd.get_dummies(df['gender'])
occ_cols = ['gender_' +  columns_name for columns_name in pclass_dummies_titanic.columns]
pclass_dummies_titanic.rename(columns=rename_columns('gender_',pclass_dummies_titanic.columns), inplace = True)
# print(type(pclass_dummies_titanic.columns) )

print(pclass_dummies_titanic)
# print(rename_columns('gender_',pclass_dummies_titanic.columns))
# print(type({'A':'a', 'B':'b', 'C':'c'}))
print("------------------------")
df = df.join(pclass_dummies_titanic)
print(df)
#


