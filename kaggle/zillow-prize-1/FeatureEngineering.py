import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

class FeatureEngineering(object):
    def __init__(self):
        return

    def train_test_all_data(self):
        X_train_all = pd.read_csv('../../../data/zillow-prize/X_train_all', low_memory=False)
        y_train_all = pd.read_csv('../../../data/zillow-prize/y_train_all', low_memory=False)
        return  X_train_all, y_train_all

    def train_test_split(self,test_size=0.33,random_state=9):
        X_train = pd.read_csv('../../../data/zillow-prize/X_train', low_memory=False)
        X_test = pd.read_csv('../../../data/zillow-prize/X_test', low_memory=False)
        y_train = pd.read_csv('../../../data/zillow-prize/y_train', low_memory=False)
        y_test = pd.read_csv('../../../data/zillow-prize/y_test', low_memory=False)
        return  X_train, X_test, y_train, y_test

    def train_test_split_temp(self,test_size=0.33,random_state=9):
        X_train_all, y_train_all, x_test = self.load_data()
        # X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33,random_state=9)
        pd.DataFrame(X_train_all).to_csv("../../../data/zillow-prize/X_train_all", index=False)
        pd.DataFrame(y_train_all).to_csv("../../../data/zillow-prize/y_train_all", index=False)
        # pd.DataFrame(X_train).to_csv("../../../data/zillow-prize/X_train", index=False)
        # pd.DataFrame(X_test).to_csv("../../../data/zillow-prize/X_test", index=False)
        # pd.DataFrame(y_train).to_csv("../../../data/zillow-prize/y_train", index=False)
        # pd.DataFrame(y_test).to_csv("../../../data/zillow-prize/y_test", index=False)
        X_train = pd.read_csv('../../../data/zillow-prize/X_train', low_memory=False)
        X_test = pd.read_csv('../../../data/zillow-prize/X_test', low_memory=False)
        y_train = pd.read_csv('../../../data/zillow-prize/y_train', low_memory=False)
        y_test = pd.read_csv('../../../data/zillow-prize/y_test', low_memory=False)
        return  X_train, X_test, y_train, y_test

    def load_data(self):
        train = pd.read_csv('../../../data/zillow-prize/train_2016.csv', low_memory=False)
        # train = pd.read_csv('../input/train_2016_v2.csv')
        properties = pd.read_csv('../../../data/zillow-prize/properties_2016.csv', low_memory=False)
        sample = pd.read_csv('../../../data/zillow-prize/sample_submission.csv', low_memory=False)

        print("Preprocessing...")
        for c, dtype in zip(properties.columns, properties.dtypes):
            if dtype == np.float64:
                properties[c] = properties[c].astype(np.float32)

        print("Set train/test data...")
        id_feature = ['heatingorsystemtypeid', 'propertylandusetypeid', 'storytypeid', 'airconditioningtypeid',
                      'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid',
                      'typeconstructiontypeid']
        for c in properties.columns:
            properties[c] = properties[c].fillna(-1)
            if properties[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(properties[c].values))
                properties[c] = lbl.transform(list(properties[c].values))
            if c in id_feature:
                lbl = LabelEncoder()
                lbl.fit(list(properties[c].values))
                properties[c] = lbl.transform(list(properties[c].values))
                dum_df = pd.get_dummies(properties[c])
                dum_df = dum_df.rename(columns=lambda x: c + str(x))
                properties = pd.concat([properties, dum_df], axis=1)
                properties = properties.drop([c], axis=1)
                # print np.get_dummies(properties[c])

        #
        # Add Feature
        #
        # error in calculation of the finished living area of home
        properties['N-LivingAreaError'] = properties['calculatedfinishedsquarefeet'] / properties[
            'finishedsquarefeet12']

        #
        # Make train and test dataframe
        #
        train = train.merge(properties, on='parcelid', how='left')
        sample['parcelid'] = sample['ParcelId']
        test = sample.merge(properties, on='parcelid', how='left')

        # drop out ouliers
        train = train[train.logerror > -0.4]
        train = train[train.logerror < 0.419]

        train["transactiondate"] = pd.to_datetime(train["transactiondate"])
        train["Month"] = train["transactiondate"].dt.month
        train["quarter"] = train["transactiondate"].dt.quarter

        test["Month"] = 10
        test['quarter'] = 4

        x_train = train.drop(
            ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'],
            axis=1)
        # y_train = train["logerror"].values
        y_train = train["logerror"]

        x_test = test[x_train.columns]
        del test, train
        print(x_train.shape, y_train.shape, x_test.shape)
        # print("--------------------------")
        # print(y_train)
        # print("--------------------------")
        # print(x_train)
        return x_train, y_train, x_test

# stack = FeatureEngineering()
# x_train, y_train, x_test =stack.load_data()
# print(x_train)