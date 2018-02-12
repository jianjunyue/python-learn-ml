from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
filename = '/Users/jianjun.yue/PycharmGItHub/data/tianchi/乘用车零售量预测/yancheng_train_20171226_0211_处理.csv'
data = read_csv(filename,header=0)
data=data.sort_values("sale_date")
class_id=data.get("class_id")
dict_class_id={}
for c_id in class_id:
    dict_class_id[c_id]=1

dict_data={}
for c_id in dict_class_id.keys():
    data_temp=data[data.class_id==c_id]
    dict_data[c_id]=data_temp
    print(data_temp["sale_date"].head(1))

# df=dict_data[289403]
df_1=dict_data[289403]
print(df_1.head(10))

plt.figure(figsize=(26,6))
# plt.plot(range(len(df["sale_date"].values)),  df["sale_quantity"].values)
plt.plot(range(len(df_1["sale_date"].values)),  df_1["sale_quantity"].values)
# plt.scatter(range(len(df["key"].values)),  df["count1"].values,s=30,c='blue',marker='x',alpha=0.5,label='C2')
plt.xticks(range(len(df_1["sale_date"].values)),df_1["sale_date"].values,rotation='90') #给X轴赋值名称
# plt.xticks(range(len(df["key"].values)),df["key_name"].values)
plt.legend(loc='upper right')
plt.title("test title")
plt.xlabel('test X');
plt.ylabel('test Y');
plt.show()
