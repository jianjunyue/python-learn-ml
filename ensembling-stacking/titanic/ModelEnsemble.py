from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from FeatureEngineering import *

class ModelEnsemble(object):
    def __init__(self):
        return

    def predict(self,clfs,X_train, y_train,X_test):
        y_predictions = pd.DataFrame();
        for clf_key in clfs.keys():
            clf = clfs[clf_key]
            clf_name=clf.__class__.__name__
            print('the clf name is :',clf_name)
            clf.fit(X_train, y_train)
            y_pre= clf.predict(X_test)
            y_predictions[clf_name]=y_pre
            # print(len(y_pre))
            oof_train,oof_test=self.get_oof(clf,X_train, y_train,X_test)
            # y_predictions["oof_train"]=oof_train
            # print(len(oof_test))
            # y_predictions["oof_test"]=oof_test
        return y_predictions

    def get_oof(self,clf, x_train, y_train):
        ntrain = x_train.shape[0]
        ntest = y_train.shape[0]
        NFOLDS = 5
        SEED = 0
        kf = KFold(n_splits=5,shuffle=False)
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        y_predictions = pd.DataFrame();
        for train_index, test_index in kf.split(x_train):
            print("TRAIN:", train_index, "TEST:", test_index)
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]
            y_te = y_train[test_index]

            clf.fit(x_tr, y_tr)
            clf_name=clf.__class__.__name__
            oof_train[test_index] = clf.predict(x_te)
            # oof_test_skf[oofname, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)