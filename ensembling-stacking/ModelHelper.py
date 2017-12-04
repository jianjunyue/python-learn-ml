import numpy as np
from sklearn.cross_validation import KFold

# Class to extend the Sklearn classifier
class ModelHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

    def get_oof(self,x_train, y_train, x_test):
        # Some useful parameters which will come in handy later on
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        SEED = 0  # for reproducibility
        NFOLDS = 5  # set folds for out-of-fold prediction
        kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            self.clf.train(x_tr, y_tr)

            oof_train[test_index] = self.clf.predict(x_te)
            oof_test_skf[i, :] = self.clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)