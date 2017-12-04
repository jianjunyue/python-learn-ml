from sklearn.datasets import load_iris

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


from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


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

for clf_key in clfs.keys():
    print('the classifier is :',clf_key)
    clf = clfs[clf_key]
    try_different_method(clf)