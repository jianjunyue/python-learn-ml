import pickle
import pandas as pd

def process_date(data):
    # year month day
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)
    data = data.drop(['Date'], axis=1)
    # data.loc[data["StateHoliday"] == '0', "StateHoliday"] = 0
    # data.loc[data["StateHoliday"] == 'a', "StateHoliday"] = 1
    # data.loc[data["StateHoliday"] == 'b', "StateHoliday"] = 2
    # data.loc[data["StateHoliday"] == 'c', "StateHoliday"] = 3

    stateHoliday = pd.get_dummies(data['StateHoliday'], prefix='StateHoliday')
    data = data.join(stateHoliday)
    data = data.drop(['StateHoliday'], axis=1)
    # data['StateHoliday'] = data['StateHoliday'].astype(int)
    data = data.fillna(0)

    return data


def process_Interval(data):
    # promo interval "Jan,Apr,Jul,Oct"
    data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
    data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)
    data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
    data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Apr" in x else 0)
    data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "May" in x else 0)
    data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jun" in x else 0)
    data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jul" in x else 0)
    data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Aug" in x else 0)
    data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Sep" in x else 0)
    data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Oct" in x else 0)
    data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Nov" in x else 0)
    data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Dec" in x else 0)

    data = data.fillna(0)

    # data["CompetitionDistance"]=np.log1p(data.CompetitionDistance)

    # data.loc[data["StoreType"] == 'a', "StoreType"] = 0
    # data.loc[data["StoreType"] == 'b', "StoreType"] = 1
    # data.loc[data["StoreType"] == 'c', "StoreType"] = 2
    # data.loc[data["StoreType"] == 'd', "StoreType"] = 3
    #
    # data.loc[data["Assortment"] == 'a', "Assortment"] = 0
    # data.loc[data["Assortment"] == 'b', "Assortment"] = 1
    # data.loc[data["Assortment"] == 'c', "Assortment"] = 2

    storeType = pd.get_dummies(data["StoreType"], prefix='StoreType')
    assortment = pd.get_dummies(data["Assortment"], prefix='Assortment')
    data = data.join(assortment)
    data = data.join(storeType)

    data = data.drop(['PromoInterval', 'StoreType', 'Assortment'], axis=1)

    return data


store = pd.read_csv('../../data/rossmann-store-sales/store.csv')
train_df = pd.read_csv('../../data/rossmann-store-sales/train.csv', dtype={'StateHoliday': pd.np.string_})
test_df = pd.read_csv('../../data/rossmann-store-sales/test.csv', dtype={'StateHoliday': pd.np.string_})
# print(store.head())


print(store["CompetitionDistance"].isnull().sum())
store['CompetitionDistance'] = store['CompetitionDistance'].fillna(0)
print(store["CompetitionDistance"].isnull().sum())
# store['LotFrontage'].corr(store['CompetitionDistance'])         #计算两个列的相关度
# print(store["CompetitionDistance"].fillna(0))
# print(store["CompetitionDistance"].info())
# plt.figure()
# plt.scatter( store["Store"]  ,  np.log1p(store["CompetitionDistance"]) )
# # plt.plot( store["Store"]  , np.log1p(store["CompetitionDistance"]) )
# plt.title('CompetitionDistance log1p Feature Importance')
# plt.xlabel('CompetitionDistance  scatter importance')
# plt.show()
# time.sleep(3)
# plt.close('all')
store = process_Interval(store)
train_df = pd.merge(train_df, store, on='Store', how='left')
test_df = pd.merge(test_df, store, on='Store', how='left')

train_df = process_date(train_df)
train_df = train_df.drop(['Customers', 'StateHoliday_b', 'StateHoliday_c'], axis=1)
test_df = process_date(test_df)

print("------------------------")
print(train_df['Sales'].corr(train_df['CompetitionDistance']))  # 计算两个列的相关度
print("------------------------")
