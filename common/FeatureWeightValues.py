from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame

#特征重要性衡量
class FeatureValues(object):

    def sort_weight_values(self,X, y,predictors):
        selector = SelectKBest(f_classif, k=5)
        selector.fit(X, y)
        scores =-np.log10(selector.pvalues_)
        dt = DataFrame()
        dt["predictors"] = predictors
        dt["scores"] = scores
        dt = dt.sort_values(by='scores', axis=0, ascending=False)
        print("------------------------------------------------")
        print("特征重要性衡量:")
        print(dt)
        print("------------------------------------------------")
        plt.bar(range(len(predictors)), dt["scores"])
        plt.xticks(range(len(predictors)), dt["predictors"], rotation='vertical')
        plt.show()