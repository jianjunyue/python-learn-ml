import pandas as pd


dict_item_category={}
dict_item_category["asda"]=1
dict_item_category["asasdasda"]=1
dict_item_category["asssssssda"]=1
dict_item_category["sdssssss"]=0

category_df = pd.DataFrame()

category_df=category_df.append(pd.DataFrame.from_dict(dict_item_category, orient='index').T)
category_df1=category_df.append(pd.DataFrame.from_dict(dict_item_category, orient='index').T)
print(category_df1)