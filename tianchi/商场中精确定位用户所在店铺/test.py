# import pandas as pd
#
# s="s;b;b"
# print("s：", s.split(";").__len__())
#
# table1=pd.DataFrame()
# row1={"user_id":"1","shop_id":"26"}
# print(type(row1))
# row2={"user_id":"2","shop_id":"12"}
# row3={"user_id":"3","shop_id":"12"}
# table1=table1.append(row1,ignore_index=True)
# table1=table1.append(row2,ignore_index=True)
# table1=table1.append(row3,ignore_index=True)
# table1.to_csv("table1.csv",index=False)
# table1=pd.read_csv("table1.csv")
# table1=pd.read_csv("table1.csv")
# # table1=pd.DataFrame()
# # for index in list(table1_1.shape(0)):
# # print(table1_1.columns)
# # for rowValue in table1_1.values:
# #     print(type(rowValue))
# #     # print(rowValue[0])
# #     # print(rowValue[1])
# #     row = {"user_id": rowValue[1], "shop_id": rowValue[0]}
# #     table1 = table1.append(row, ignore_index=True)
#
# table1['shop_id'] = table1['shop_id'].astype(str)
#
# # print(table1)
# table2=pd.DataFrame()
# row1={"shop_id":"26","category_id":"11_name","mall_id":"11_type"}
# row2={"shop_id":"133","category_id":"12_name","mall_id":"12_type"}
# table2=table2.append(row1,ignore_index=True)
# table2=table2.append(row2,ignore_index=True)
#
# table2_1=pd.DataFrame(index=table2["shop_id"])
# # table2_1 = pd.DataFrame({'shop_name': table2["shop_name"].values},
# #                    index=table2["shop_id"].values)
# table2_1["category_id"]=table2["category_id"].values
# table2_1["mall_id"]=table2["mall_id"].values
# # table2_1["shop_id"]=table2["shop_id"].values
#
# # print(table2)
# print("------table1--------")
# print(table1)
#
# # print("------table2_1--------")
# # print(table2_1)
#
# # print("--------------")
# # 左联表（注意是3个等号！）
# table3 = table1.join(table2_1, on='shop_id')
# # table3 = table1.join(table2_1, how='left', on='shop_id')
# # table3 = pd.join(table1, table2_1, how='left', on='shop_id')
# # table3 =pd.merge(table1, table2_1, on="shop_id", how='left')
# print("-------table3-------")
# print(table3)
#
# # left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
# #     'B': ['B0', 'B1', 'B2', 'B3'],
# #     'key': ['K0', 'K1', 'K0', 'K1']})
# #
# # right = pd.DataFrame({'C': ['C0', 'C1'],
# #                      'D': ['D0', 'D1']},
# #                    index=['K0', 'K1'])
# #
# # result = left.join(right, on='key')
# # print(left)
# # print(right)
# # print(result)
#
# train_user_shop=pd.read_csv("user-ccf_first_round_shop_info.csv")
# train_shop=pd.read_csv("训练数据-ccf_first_round_shop_info.csv")
#
# train_shop['shop_id'] = train_shop["shop_id"].apply(lambda x: x.replace("s_","")).astype(int)
# train_shop['category_id'] = train_shop["category_id"].apply(lambda x: x.replace("c_",""))
# train_shop['mall_id'] = train_shop["mall_id"].apply(lambda x: x.replace("m_",""))
# train_shop['shop_longitude'] =train_shop['longitude']
# train_shop['shop_latitude'] =train_shop['latitude']
#
# train_shop_temp=pd.DataFrame(index=train_shop["shop_id"])
# # print(train_user_shop.columns)
# # train_user_shop=train_user_shop.drop(["user_id","time_stamp"], axis = 1)
#
# train_shop_temp["category_id"]=train_shop["category_id"].values
# train_shop_temp["shop_longitude"]=train_shop["shop_longitude"].values
# train_shop_temp["shop_latitude"]=train_shop["shop_latitude"].values
# train_shop_temp["price"]=train_shop["price"].values
# train_shop_temp["mall_id"]=train_shop["mall_id"].values
# # print(table2_1.head())
# print("----------------------")
# # train_user_shop_info = train_user_shop.join(train_shop_temp, on='shop_id')
# # train_user_shop_info = train_user_shop.join(table2_1, on='shop_id')
# # print(train_user_shop_info.head())
#
# #
# # train_user_shop_info_temp=pd.DataFrame()
# # train_user_shop_info_temp["shop_id"]=train_user_shop["shop_id"]
# # train_user_shop_info_temp["user_id"]=train_user_shop["user_id"]
# # # train_user_shop_info_temp["longitude"]=train_user_shop["longitude"]
# #
# # print("----------------------")
# # print(train_user_shop_info_temp.head())
# # print("----------------------")
# # train_user_shop['shop_id'] = train_user_shop['shop_id'].astype(str)
# train_shop['shop_id'] = train_shop['shop_id'].astype(int)
# train_user_shop_info = train_user_shop.join(train_shop_temp, on='shop_id')
# print(train_user_shop_info.head())
#
# # train_user_shop_info.to_csv("训练数据-ccf_first_round_user_shop_behavior_new_datetime_shopinfo.csv",index=False)
#
# keys_counts=train_user_shop["user_shop_distance_int"].value_counts()
# # keys_counts.index
# keys_counts_temp=pd.DataFrame()
# keys_counts_temp["index"]=keys_counts.index
# keys_counts_temp["values"]=keys_counts.values
# keys_counts_temp
train_user_shop_row=""
wifi_info_list = train_user_shop_row.split(";")
for wifi_info in wifi_info_list:
    # wifi_info=""
    if wifi_info == "":
        wifi_info=" | | "
    bssid, signal, used = wifi_info.split("|")
    bssid = bssid.replace("b_", "")
    used = 1 if used == "true" else 0
    print(bssid,"---")