import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer


df=pd.DataFrame()
value=["mall","room","group","data"]
df["gender"]=value
print(df)
print("------")
# lb = preprocessing.LabelBinarizer()
# gender 改为 0-1 数值
# df["gender"]= lb.fit_transform(df['gender'])

#标签编码（Label encoding）
le = preprocessing.LabelEncoder()
df["gender"]= le.fit_transform(df['gender'])

print(df)




 
 