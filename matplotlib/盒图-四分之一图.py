import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

x=["a1","a2","a3","a4","a5","a6","a7","a8"]
x1=[1,3,5,6,8,9,5,22]
x2=[1,3,5,6,8,9,5,22]
y1=[21,34,5,6,78,9,5,22]
y2=[21,34,5,6,78,9,5,22]
df = pd.DataFrame(columns=['key1', 'key2', 'count1', 'count2'])
df["key1"]=x1
df["key2"]=x2
df["count1"]=y1
df["count2"]= y2

plt.figure(figsize=(8,6))
cols=["count1","count2"]
plt.boxplot( df[cols].values )
plt.xticks(range(len(cols)),cols)
plt.legend(loc='upper right')
plt.title("test title")
plt.xlabel('test X');
plt.ylabel('test Y');
plt.show()