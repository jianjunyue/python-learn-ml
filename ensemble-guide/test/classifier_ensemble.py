from sklearn.datasets import load_iris

from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
#scikit-learn的基本分类方法（决策树、SVM、KNN）和集成方法（随机森林，Adaboost和GBRT）
#http://blog.csdn.net/u010900574/article/details/52669072

iris = load_iris()

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    import numpy
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def load_data():
    iris.data, iris.target = shuffle_in_unison(iris.data, iris.target)
    x_train ,x_test = iris.data[:100],iris.data[100:]
    y_train, y_test = iris.target[:100].reshape(-1,1),iris.target[100:].reshape(-1,1)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()

clfs = {'svm': svm.SVC(),
        'decision_tree':tree.DecisionTreeClassifier(),
        'naive_gaussian': naive_bayes.GaussianNB(),
        'naive_mul':naive_bayes.MultinomialNB(),
        'K_neighbor' : neighbors.KNeighborsClassifier(),
        'bagging_knn' : BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5),
        'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),
        'random_forest' : RandomForestClassifier(n_estimators=50),
        'adaboost':AdaBoostClassifier(n_estimators=50),
        'gradient_boost' : GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
        }

def try_different_method(clf):
    clf.fit(x_train,y_train.ravel())
    score = clf.score(x_test,y_test.ravel())
    print('the score is :', score)

def loss_score(y_pre, y_test):
    loss_error = 0;
    z = zip(y_pre, y_test)
    for py, ty in z:
        if py!=ty:
            loss_error+=1
    return loss_error/len(y_test)


stacker_pred_ensemble_df=pd.DataFrame()
def predict_ls(clf_key,clf):
    clf.fit(x_train,y_train.ravel())
    y_pre = clf.predict(x_test)
    stacker_pred_ensemble_df[clf_key]=y_pre
    loss_error=loss_score(y_pre, y_test.ravel())
    print("loss_error:",loss_error)



for clf_key in clfs.keys():
    print('the classifier is :',clf_key)
    clf = clfs[clf_key]
    try_different_method(clf)
    predict_ls(clf_key,clf)

def y_pre_max_count(y_pre):
    try:
        rdict = {}
        for y in y_pre:
            count = rdict.get(y)
            if count == None:
                count=0
            count+=1
            rdict[y]=count
    except Exception as err:
        print(err)
    max_count=0
    y_pre_value=-1
    for key in rdict:
        if max_count<rdict[key]:
            max_count=rdict[key]
            y_pre_value=key
    return y_pre_value

def get_y_pre(y_pre_list):
    list=[]
    for y_pre in y_pre_list.values:
        list.append(y_pre_max_count(y_pre))
    return list

y_max_count=get_y_pre(stacker_pred_ensemble_df )
stacker_pred_ensemble_df["y_max_count"] =y_max_count
stacker_pred_ensemble_df["y_test"] =y_test.ravel()
stacker_pred_ensemble_df.to_csv("ensemble_sub"  ,index=False)
print("MAX_loss_error:", loss_score(y_max_count, y_test.ravel())  )
