import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# x=["a1","a2","a3","a4","a5","a6","a7","a8"]
x=[1,3,5,6,8,9,5,22]
y=[21,34,5,6,78,9,5,22]
df = pd.DataFrame(columns=['key', 'count'])
df["key"]=x
df["count"]=y

# fig,ax=plt.subplots()
# ax = df["count"].plot.bar()
# ax.set_xticklabels(df["key"],rotation='1')
# ax.set_xlabel('test X');
# ax.set_ylabel('test Y');
# plt.show()

# plt.bar(df["key"].values,df["count"].values)
plt.bar(left=df["key"].values, height=df["count"].values, width=0.35, align="center", yerr=0.0001)
plt.xticks(df["key"].values)# x轴刻度，优先级比bar left高
# plt.set_xticklabels(df["key"],rotation='1')

plt.title("test title")
plt.xlabel('test X');
plt.ylabel('test Y');
plt.show()