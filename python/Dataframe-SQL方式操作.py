import pandas as pd
import numpy as np

# 【Python实战】Pandas：让你像写SQL一样做数据分析（一）
# https://www.cnblogs.com/en-heng/p/5630849.html
# 【Python实战】Pandas：让你像写SQL一样做数据分析（二）
# http://www.cnblogs.com/en-heng/p/5686062.html

df = pd.DataFrame({'row_id': [1, 2, 3, 4, 5],
                   'total_bill': [16.99, 10.34, 23.68, 23.68, 24.59],
                   'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
                   'sex': ['Female', 'Male', 'Male', 'Male', 'Female']})
print(df)
print("------------------------------")
# 1. SQL操作
# select
print("--------------select----------------")
# print(df.loc[1:3, ['tip','total_bill']])
# print(df[['tip','total_bill']])
# where 过滤
print("--------------where 过滤----------------")
# print(df.query("total_bill < 121 and tip==3.50 and sex=='Male'").query("sex=='Male'"))
# distinct 去重
print("--------------distinct 去重----------------")
# print(df.drop_duplicates(subset=['sex'], keep='first', inplace=False)) #distinct 去重方法
# group
print("--------------group ----------------")
# print(df.groupby('sex').agg({'tip': np.max, 'total_bill': np.sum}))
# print(df.groupby('sex').count())
# print(df.groupby('sex')['tip'].count())
# as修改列的别名
print("--------------as ----------------")
# df.columns = ['total', 'pit', 'xes']
# df.rename(columns={'total_bill': 'total', 'tip': 'pit', 'sex': 'xes'}, inplace=True)
# join
print("--------------join ----------------")
df1=df
df2=df
# 2.方式
# df2.rename(columns={'total_bill': 'total_bill2', 'tip': 'tip2', 'sex': 'sex2'}, inplace=True)
# print(pd.merge(df1, df2, how='left', left_on='row_id', right_on='row_id')) # left join

#order 支持多列order
print("--------------order ----------------")
# print(df.sort_values(['sex', 'tip'], ascending=[False, True])) # ascending 控制是否倒排

#top
print("--------------top ----------------")
# print(df.nlargest(3, columns=['tip']))
# apply 自定义
print("--------------apply 自定义  ----------------")
# print(df['tip'].map(lambda x: x - 1))
# print(df[['total_bill', 'tip']].apply(sum))
# print(df["sex"].apply(lambda x: x.upper() if type(x) is str else x)) #单个字段
# print(df.applymap(lambda x: x.upper() if type(x) is str else x)) #全部字段
#Add一行或一列数据
print("--------------Add ----------------")
#增加一行
row_df = pd.DataFrame(np.array([['6', 'ios', '4', 32]]), columns=['row_id', 'sex', 'tip', 'total_bill'])
df = df.append(row_df, ignore_index=True)
# print(df)
#增加一列
df['time'] = '2016-07-19'
# print(df)
#To Dict
print("--------------To Dict ----------------")
# print(df.set_index('row_id')[['sex', 'tip']].to_dict())
#排序编号
print("-------------- 排序编号 ----------------")
df['rank']=df['tip'].rank(method='first', ascending=False).apply(lambda x: int(x))
# print(df['rank'])

# 写MySQL
print("-------------- 写MySQL ----------------")
# import MySQLdb
# from sshtunnel import SSHTunnelForwarder
# with SSHTunnelForwarder(('porxy host', port),
#                         ssh_password='os passwd',
#                         ssh_username='os user name',
#                         remote_bind_address=('mysql host', 3306)) as server:
#     conn = MySQLdb.connect(host="127.0.0.1", user="mysql user name", passwd="mysql passwd",
#                            db="db name", port=server.local_bind_port, charset='utf8')
#     df.to_sql(name='tb name', con=conn, flavor='mysql', if_exists='append', index=False)