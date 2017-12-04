import pandas as pd
import datetime
import csv
import numpy as np
import os
import scipy as sp
import time
import xgboost as xgb
import operator
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation
from matplotlib import pylab as plt
plot = True

goal = 'Sales'
myid = 'Id'

#实体嵌入
#https://www.kaggle.com/thie1e/exploratory-analysis-rossmann/notebook

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

def process_date(data):
    # year month day
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)
    data=data.drop(['Date'],axis = 1)
    # data.loc[data["StateHoliday"] == '0', "StateHoliday"] = 0
    # data.loc[data["StateHoliday"] == 'a', "StateHoliday"] = 1
    # data.loc[data["StateHoliday"] == 'b', "StateHoliday"] = 2
    # data.loc[data["StateHoliday"] == 'c', "StateHoliday"] = 3

    stateHoliday = pd.get_dummies(data['StateHoliday'], prefix='StateHoliday')
    data = data.join(stateHoliday)
    data=data.drop(['StateHoliday'],axis = 1)
    # data['StateHoliday'] = data['StateHoliday'].astype(int)
    data=data.fillna(0)

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

    data=data.fillna(0)

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

    data=data.drop(['PromoInterval','StoreType','Assortment'],axis = 1)

    return data

store = pd.read_csv('../../data/rossmann-store-sales/store.csv')
train_df = pd.read_csv('../../data/rossmann-store-sales/train.csv',dtype={'StateHoliday':pd.np.string_})
test_df = pd.read_csv('../../data/rossmann-store-sales/test.csv',dtype={'StateHoliday':pd.np.string_})
# print(store.head())


print(store["CompetitionDistance"].isnull().sum())
store['CompetitionDistance']=store['CompetitionDistance'].fillna(0)
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
store=process_Interval(store)
train_df=pd.merge(train_df, store, on='Store', how='left')
test_df=pd.merge(test_df, store, on='Store', how='left')

train_df=process_date(train_df)
train_df=train_df.drop(['Customers','StateHoliday_b','StateHoliday_c'],axis = 1)
test_df=process_date(test_df)

print("------------------------")
print(train_df['Sales'].corr(train_df['CompetitionDistance'])) #计算两个列的相关度
print("------------------------")


# print(store["StoreType"].value_counts())
# print(store["Assortment"].value_counts())
print(train_df.info())
print("--------------------")
print(test_df.info())

print(train_df.head())
print("--------------------")
print(test_df.head())

# print(train_df["StateHoliday"].value_counts())

def XGB_native(train,test,features,features_non_numeric):
    depth = 13
    eta = 0.01
    ntrees = 8000
    mcw = 3
    params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "min_child_weight": mcw,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1
              }
    print("Running with params: " + str(params))
    print("Running with ntrees: " + str(ntrees))
    print("Running with features: " + str(features))

    # Train model with local split
    tsize = 0.05
    X_train, X_test = cross_validation.train_test_split(train, test_size=tsize)
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train[goal] + 1))
    dvalid = xgb.DMatrix(X_test[features], np.log(X_test[goal] + 1))
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[goal].values)
    print(error)

    # Predict and Export
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({myid: test[myid], goal: np.exp(test_probs) - 1})
    if not os.path.exists('../../result/'):
        os.makedirs('../../result/')

    s_time=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    submission.to_csv("../../result/dat-xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s_time%s.csv" % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize),str(s_time)) , index=False)
    # Feature importance
    if plot:
      outfile = open(str(s_time)+'xgb.fmap', 'w')
      i = 0
      for feat in features:
          outfile.write('{0}\t{1}\tq\n'.format(i, feat))
          i = i + 1
      outfile.close()
      importance = gbm.get_fscore(fmap=str(s_time)+'xgb.fmap')
      importance = sorted(importance.items(), key=operator.itemgetter(1))
      df = pd.DataFrame(importance, columns=['feature', 'fscore'])
      df['fscore'] = df['fscore'] / df['fscore'].sum()
      # Plotitup
      plt.figure()
      df.plot()
      df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
      plt.title('XGBoost Feature Importance')
      plt.xlabel('relative importance')
      plt.show()
      plt.gcf().savefig('Feature_Importance_xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s_time%s.png' % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize),str(s_time)))

print("=> 载入数据中...")
# train,test,features,features_non_numeric = load_data()
print("=> 处理数据与特征工程...")
# train,test,features,features_non_numeric = process_data(train,test,features,features_non_numeric)
train=train_df
test=test_df
fcolumns=train.columns.tolist()
features=[f for f in fcolumns if f not in ["Id","Sales"]]
features_non_numeric=[]

# print(features)
# print(train[features])
# print(test[goal])
print("=> 使用XGBoost建模...")
XGB_native(train,test,features,features_non_numeric)

