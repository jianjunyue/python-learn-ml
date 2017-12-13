import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from hyperopt import tpe
from hpsklearn import HyperoptEstimator, any_classifier
from hpsklearn import svc

digits = load_digits()
X = digits.data
y = digits.target
test_size = int(0.2*len(y))
np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]

estim = HyperoptEstimator(classifier=any_classifier('clf'),algo=tpe.suggest, seed=0)
estim.fit(X_train,y_train)
print(estim.score(X_test,y_test))
print(estim.best_model())