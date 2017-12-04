from common.avazu_ctr_prediction_utils import *
import common.avazu_ctr_prediction_utils

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=32, max_depth=40, min_samples_split=100, min_samples_leaf=10, random_state=0, criterion='entropy',max_features=8, verbose = 1, n_jobs=-1, bootstrap=False)