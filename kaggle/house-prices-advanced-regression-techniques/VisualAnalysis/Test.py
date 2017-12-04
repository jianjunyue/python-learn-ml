import numpy as np
import pandas as pd

df = pd.DataFrame({'code': ['foo', 'bar', 'baz'] * 2,
                 'data': [0.16, -0.21, 0.33, 0.45, -0.59, 0.62],
                     'flag': [False, True] * 3})
print(df)
code_groups = df.groupby('code')
print("-----------------")
print(code_groups[['data']].transform(sum))
agg_n_sort_order = code_groups[['data']].transform(sum).sort_values(by='data')
print("-----------------")
print(agg_n_sort_order)
sorted_df = df.loc[agg_n_sort_order.index]
print("-----------------")
print(sorted_df)