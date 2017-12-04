import numpy as np
import pandas as pd

df = pd.DataFrame([
            ['green', 1, "10.1", 'class1'],
            ['red', 2, "13.5", ],
            ['blue', 3, "15.3", ]])
# print(df)
df.columns = ['color', 'size', 'prize', 'class']
print("-------------------")
# print(df)

size_mapping = {
           'XL': 3,
           'L': 2,
           'M': 1}
# df['size'] = df['size'].map(size_mapping)
print("-------------------")
# print(df)
df['class'] = df['class'].astype(object)
# class_mapping = {label:idx for idx,label in enumerate(set(df['class label']))}
# df['class label'] = df['class label'].map(class_mapping)
print("-------------------")
print(df.dtypes)
# pd.get_dummies(df , prefix=['color'])
# print("-------------------")
# print(df)

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['1', '2', '3'],  'C': [1, 2, 3]})

print("-------------------")
print(df)
df=pd.get_dummies(df, columns=["A","B"])
print("-------------------")
print(df)