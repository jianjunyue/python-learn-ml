import pandas as pd

# 2017-11-26
# 把分类统计的数据转换成 DataFrame
# sort_values =train_df["MSSubClass"].value_counts()
def group_values(value_counts):
    dict = value_counts.to_dict()
    df = pd.DataFrame(columns=['keyid', 'count'])
    listkey = []
    listcount = []
    for key in dict:
        listkey.append(key)
        listcount.append(dict[key])
        # df.loc[df.shape[0] + 1] = {'keyid': key, 'count': dict[key]}
    df['keyid']=listkey
    df['count']=listcount
    return df
