import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import  SelectKBest,f_classif
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

path="/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/train.csv"
data=pd.read_csv(path)
path_test="/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/test.csv"
data_test=pd.read_csv(path_test)
print(data.head())
print(data.describe())

print(data["MSSubClass"].describe())
print(data["MSSubClass"].unique())
print(data["MSSubClass"].value_counts())
print("--------------------------------------")
print(data_test["MSSubClass"].describe())
print(data_test["MSSubClass"].unique())
print(data_test["MSSubClass"].value_counts())

def getMSZoning(value):
    MSZoning = value["MSZoning"]
    title_mapping={"RL":1,"RM":2,"C (all)":3,"FV":4,"RH":5}
    for k,v in title_mapping.items():
        MSZoning[MSZoning==k]=v
    value["MSZoning"] = MSZoning
    value["MSZoning"] = value["MSZoning"].fillna(6)
getMSZoning(data)
getMSZoning(data_test)
print(data["MSZoning"].describe())
print(data["MSZoning"].unique())
print(data["MSZoning"].value_counts())
print("--------------------------------------")
print(data_test["MSZoning"].describe())
print(data_test["MSZoning"].unique())
print(data_test["MSZoning"].value_counts())

data["LotFrontage"]=data["LotFrontage"].fillna(data["LotFrontage"].mean())
data_test["LotFrontage"]=data_test["LotFrontage"].fillna(data_test["LotFrontage"].mean())
print(data["LotFrontage"].describe())
print(data["LotFrontage"].unique())
print(data["LotFrontage"].value_counts())
print("--------------------------------------")
print(data_test["LotFrontage"].describe())
print(data_test["LotFrontage"].unique())
print(data_test["LotFrontage"].value_counts())

data["LotArea"]=data["LotArea"].fillna(data["LotArea"].mean())
data_test["LotArea"]=data_test["LotArea"].fillna(data_test["LotArea"].mean())
print(data["LotArea"].describe())
print(data["LotArea"].unique())
print(data["LotArea"].value_counts())
print("--------------------------------------")
print(data_test["LotArea"].describe())
print(data_test["LotArea"].unique())
print(data_test["LotArea"].value_counts())

keyName="Street"
def getStreet(value):
    Street = value["Street"]
    title_mapping={"Pave":1,"Grvl":2}
    for k,v in title_mapping.items():
        Street[Street==k]=v
    value["Street"] = Street
getStreet(data)
getStreet(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Alley"
def getAlley(value):
    temp = value[keyName]
    title_mapping={"Pave":1,"Grvl":2}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(3)
getAlley(data)
getAlley(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="LotShape"
def getLotShape(value):
    temp = value[keyName]
    title_mapping={"Reg":1,"IR1":2,"IR2":3,"IR3":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getLotShape(data)
getLotShape(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="LandContour"
def getLandContour(value):
    temp = value[keyName]
    title_mapping={"Lvl":1,"Bnk":2,"Low":3,"HLS":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getLandContour(data)
getLandContour(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Utilities"
def getUtilities(value):
    temp = value[keyName]
    title_mapping={"AllPub":1,"NoSeWa":2}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(3)
getUtilities(data)
getUtilities(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="LotConfig"
def getLotConfig(value):
    temp = value[keyName]
    title_mapping={"Inside":1,"FR2":2,"Corner":3,"CulDSac":4,"FR3":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getLotConfig(data)
getLotConfig(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="LandSlope"
def getLandSlope(value):
    temp = value[keyName]
    title_mapping={"Gtl":1,"Mod":2,"Sev":3}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getLandSlope(data)
getLandSlope(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Neighborhood"
def getNeighborhood(value):
    temp = value[keyName]
    title_mapping={"CollgCr":1,"Veenker":2,"Crawfor":3,"NoRidge":4,"Mitchel":5,"Somerst":6,"NWAmes":7,"OldTown":8,"BrkSide":9,"Sawyer":10,"NridgHt":11,"NAmes":12,"SawyerW":13,"IDOTRR":14,"MeadowV":15,"Edwards":16,"Timber":17,"Gilbert":18,"StoneBr":19,"NPkVill":20,"Blmngtn":21,"BrDale":22,"SWISU":23,"Blueste":24,"ClearCr":25}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getNeighborhood(data)
getNeighborhood(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Condition1"
def getCondition1(value):
    temp = value[keyName]
    title_mapping={"Norm":1,"Feedr":2,"PosN":3,"Artery":4,"RRAe":5,"RRNn":6,"RRAn":7,"PosA":8,"RRNe":9}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getCondition1(data)
getCondition1(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Condition2"
def getCondition2(value):
    temp = value[keyName]
    title_mapping={"Norm":1,"Artery":2,"RRNn":3,"Feedr":4,"PosN":5,"PosA":6,"RRAn":7,"RRAe":8}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getCondition2(data)
getCondition2(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BldgType"
def getBldgType(value):
    temp = value[keyName]
    title_mapping={"1Fam":1,"2fmCon":2,"Duplex":3,"TwnhsE":4,"Twnhs":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getBldgType(data)
getBldgType(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="HouseStyle"
def getHouseStyle(value):
    temp = value[keyName]
    title_mapping={"2Story":1,"1Story":2,"1.5Fin":3,"1.5Unf":4,"SFoyer":5,"SLvl":5,"2.5Unf":5,"2.5Fin":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getHouseStyle(data)
getHouseStyle(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="OverallQual"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="OverallCond"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="YearBuilt"
def getYearBuilt(value):
    temp = 2018-value
    return temp
data[keyName] = data[keyName].apply(getYearBuilt)
data_test[keyName] = data_test[keyName].apply(getYearBuilt)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="YearRemodAdd"
def getYearRemodAdd(value):
    temp = 2018-value
    return temp
data[keyName] = data[keyName].apply(getYearRemodAdd)
data_test[keyName] = data_test[keyName].apply(getYearRemodAdd)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="RoofStyle"
def getRoofStyle(value):
    temp = value[keyName]
    title_mapping={"Gable":1,"Hip":2,"Gambrel":3,"Mansard":4,"Flat":5,"Shed":6}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getRoofStyle(data)
getRoofStyle(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="RoofMatl"
def getRoofMatl(value):
    temp = value[keyName]
    title_mapping={"CompShg":1,"WdShngl":2,"Metal":3,"WdShake":4,"Membran":5,"Tar&Grv":7,"Roll":7,"ClyTile":8}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(3)
getRoofMatl(data)
getRoofMatl(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Exterior1st"
def getExterior1st(value):
    temp = value[keyName]
    title_mapping={"VinylSd":1,"MetalSd":2,"Wd Sdng":3,"HdBoard":4,"BrkFace":5,"WdShing":7,"CemntBd":7,"Plywood":8,"AsbShng":9,"Stucco":10,"BrkComm":11,"AsphShn":12,"Stone":13,"ImStucc":14,"CBlock":15}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(16)
getExterior1st(data)
getExterior1st(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Exterior2nd"
def getExterior2nd(value):
    temp = value[keyName]
    title_mapping={"VinylSd":1,"MetalSd":2,"Wd Shng":3,"HdBoard":4,"Plywood":5,"Wd Sdng":7,"CemntBd":7,"BrkFace":8,"Stucco":9,"AsbShng":10,"Brk Cmn":11,"ImStucc":12,"AsphShn":13,"Stone":14,"CmentBd":15,"CBlock":16,"Other":17}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(18)
getExterior2nd(data)
getExterior2nd(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="MasVnrType"
def getMasVnrType(value):
    temp = value[keyName]
    title_mapping={"BrkFace":1,"None":2,"Stone":3,"BrkCmn":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(5)
getMasVnrType(data)
getMasVnrType(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="MasVnrArea"
data[keyName]=data[keyName].fillna(data[keyName].mean())
data_test[keyName]=data_test[keyName].fillna(data_test[keyName].mean())
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="ExterQual"
def getExterQual(value):
    temp = value[keyName]
    title_mapping={"Gd":1,"TA":2,"Ex":3,"Fa":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(18)
getExterQual(data)
getExterQual(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="ExterCond"
def getExterCond(value):
    temp = value[keyName]
    title_mapping={"Gd":1,"TA":2,"Ex":3,"Fa":4,"Po":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(18)
getExterCond(data)
getExterCond(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Foundation"
def getFoundation(value):
    temp = value[keyName]
    title_mapping={"PConc":1,"CBlock":2,"BrkTil":3,"Wood":4,"Slab":5,"Stone":6}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(18)
getFoundation(data)
getFoundation(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtQual"
def getBsmtQual(value):
    temp = value[keyName]
    title_mapping={"Gd":1,"TA":2,"Ex":3,"Fa":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(5)
getBsmtQual(data)
getBsmtQual(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtCond"
def getBsmtCond(value):
    temp = value[keyName]
    title_mapping={"TA":1,"Gd":2,"Fa":3,"Po":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(5)
getBsmtCond(data)
getBsmtCond(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtExposure"
def getBsmtExposure(value):
    temp = value[keyName]
    title_mapping={"No":1,"Gd":2,"Mn":3,"Av":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(5)
getBsmtExposure(data)
getBsmtExposure(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtFinType1"
def getBsmtFinType1(value):
    temp = value[keyName]
    title_mapping={"GLQ":1,"ALQ":2,"Unf":3,"Rec":4,"BLQ":5,"LwQ":6}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(7)
getBsmtFinType1(data)
getBsmtFinType1(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtFinSF1"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtFinType2"
def getBsmtFinType2(value):
    temp = value[keyName]
    title_mapping={"GLQ":1,"ALQ":2,"Unf":3,"Rec":4,"BLQ":5,"LwQ":6}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(7)
getBsmtFinType2(data)
getBsmtFinType2(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtFinSF2"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtUnfSF"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="TotalBsmtSF"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Heating"
def getHeating(value):
    temp = value[keyName]
    title_mapping={"GasA":1,"GasW":2,"Grav":3,"Wall":4,"OthW":5,"Floor":6}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(7)
getHeating(data)
getHeating(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="HeatingQC"
def getHeatingQC(value):
    temp = value[keyName]
    title_mapping={"Ex":1,"Gd":2,"TA":3,"Fa":4,"Po":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(7)
getHeatingQC(data)
getHeatingQC(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="CentralAir"
def getCentralAir(value):
    temp = value[keyName]
    title_mapping={"Y":1,"N":2}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(7)
getCentralAir(data)
getCentralAir(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Electrical"
def getElectrical(value):
    temp = value[keyName]
    title_mapping={"SBrkr":1,"FuseF":2,"FuseA":3,"FuseP":4,"Mix":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(6)
getElectrical(data)
getElectrical(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="1stFlrSF"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="2ndFlrSF"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="LowQualFinSF"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="GrLivArea"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtFullBath"
data_test[keyName] = data_test[keyName].fillna(4)
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BsmtHalfBath"
data_test[keyName] = data_test[keyName].fillna(3)
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="FullBath"
# data_test[keyName] = data_test[keyName].fillna(3)
# data_test[keyName] = data_test[keyName].apply(lambda x:int(x))
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="HalfBath"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="BedroomAbvGr"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="KitchenAbvGr"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="KitchenQual"
def getKitchenQual(value):
    temp = value[keyName]
    title_mapping={"Gd":1,"TA":2,"Ex":3,"Fa":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(5)
getKitchenQual(data)
getKitchenQual(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="TotRmsAbvGrd"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Functional"
def getFunctional(value):
    temp = value[keyName]
    title_mapping={"Typ":1,"Min1":2,"Maj1":3,"Min2":4,"Mod":5,"Maj2":6,"Sev":7}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(8)
getFunctional(data)
getFunctional(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Fireplaces"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="FireplaceQu"
def getFireplaceQu(value):
    temp = value[keyName]
    title_mapping={"TA":1,"Gd":2,"Fa":3,"Ex":4,"Po":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(6)
getFireplaceQu(data)
getFireplaceQu(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="GarageType"
def getFireplaceQu(value):
    temp = value[keyName]
    title_mapping={"Attchd":1,"Detchd":2,"BuiltIn":3,"CarPort":4,"Basment":5,"2Types":6}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(7)
getFireplaceQu(data)
getFireplaceQu(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="GarageYrBlt"
def getGarageYrBlt(value):
    # print(value)
    temp =2018 - value
    temp=int(temp)
    if temp<0:
        temp=0
    return int(temp)
data[keyName] = data[keyName].fillna(data[keyName].mean())
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
mean=data_test[keyName].mean()
data_test[keyName]=data_test[keyName].apply(lambda x: mean if x>2018 else x )

data[keyName]=data[keyName].apply(lambda x:getGarageYrBlt(x))
data_test[keyName]=data_test[keyName].apply(lambda x:getGarageYrBlt(x))
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="GarageFinish"
def getGarageFinish(value):
    temp = value[keyName]
    title_mapping={"RFn":1,"Unf":2,"Fin":3}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(4)
getGarageFinish(data)
getGarageFinish(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="GarageCars"
data_test[keyName] = data_test[keyName].fillna(2)
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="GarageArea"
data_test[keyName] = data_test[keyName].fillna(data_test[keyName].mean())
data_test[keyName] = data_test[keyName].apply(lambda x:int(x))
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="GarageQual"
def getGarageQual(value):
    temp = value[keyName]
    title_mapping={"TA":1,"Fa":2,"Gd":3,"Ex":4,"Po":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(6)
getGarageQual(data)
getGarageQual(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="GarageCond"
def getGarageQual(value):
    temp = value[keyName]
    title_mapping={"TA":1,"Fa":2,"Gd":3,"Ex":4,"Po":5}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(6)
getGarageQual(data)
getGarageQual(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="PavedDrive"
def getPavedDrive(value):
    temp = value[keyName]
    title_mapping={"Y":1,"N":2,"P":3}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(6)
getPavedDrive(data)
getPavedDrive(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="WoodDeckSF"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="OpenPorchSF"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="EnclosedPorch"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="3SsnPorch"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="ScreenPorch"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="PoolArea"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="PoolQC"
def getPoolQC(value):
    temp = value[keyName]
    title_mapping={"Ex":1,"Fa":2,"Gd":3}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(4)
getPoolQC(data)
getPoolQC(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="Fence"
def getFence(value):
    temp = value[keyName]
    title_mapping={"MnPrv":1,"GdWo":2,"GdPrv":3,"MnWw":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(5)
getFence(data)
getFence(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="MiscFeature"
def getMiscFeature(value):
    temp = value[keyName]
    title_mapping={"Shed":1,"Gar2":2,"TenC":3,"Othr":4}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(5)
getMiscFeature(data)
getMiscFeature(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="MiscVal"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="MoSold"
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="YrSold"
def getYrSold(value):
    temp =2010 - value
    return int(temp)
data[keyName]=data[keyName].apply(lambda x:getYrSold(x))
data_test[keyName]=data_test[keyName].apply(lambda x:getYrSold(x))
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="SaleType"
def getSaleType(value):
    temp = value[keyName]
    title_mapping={"WD":1,"New":2,"COD":3,"ConLD":4,"ConLI":5,"CWD":6,"ConLw":7,"Con":8,"Oth":9}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    value[keyName] = value[keyName].fillna(10)
getSaleType(data)
getSaleType(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

keyName="SaleCondition"
def getSaleCondition(value):
    temp = value[keyName]
    title_mapping={"Normal":1,"Abnorml":2,"Partial":3,"AdjLand":4,"Alloca":5,"Family":6}
    for k,v in title_mapping.items():
        temp[temp==k]=v
    value[keyName] = temp
    # value[keyName] = value[keyName].fillna(10)
getSaleCondition(data)
getSaleCondition(data_test)
print(data[keyName].describe())
print(data[keyName].unique())
print(data[keyName].value_counts())
print("--------------------------------------")
print(data_test[keyName].describe())
print(data_test[keyName].unique())
print(data_test[keyName].value_counts())

data.to_csv('/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/train_预处理.csv',index=None)
data_test.to_csv('/Users/jianjun.yue/PycharmGItHub/data/kaggle/House-Prices/test_预处理.csv',index=None)