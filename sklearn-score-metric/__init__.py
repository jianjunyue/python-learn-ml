from sklearn.metrics import fbeta_score, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.dummy import DummyClassifier

from sklearn.linear_model import SGDClassifier
#sklearn中的模型评估-构建评估函数
#http://www.cnblogs.com/harvey888/p/6964741.html


# ftwo_scorer = make_scorer(fbeta_score, beta=2)
# grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer)

def my_custom_loss_func(train, predictions):
    diff = np.abs(train - predictions).max()
    print("---------------")
    print(train)
    print(predictions)
    print(np.log(1 + diff))
    return np.log(1 + diff)

loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
score = make_scorer(my_custom_loss_func, greater_is_better=True)
train = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
train1 = np.array([[-1, -1], [-2, -1], [1, 1], [2, 2]])
predictions  = np.array([1, 1, 1, 11])
predictions1  = np.array([1, 1, 1, 1])
predictions2  = np.array([1, 1, 1, 0])
clf = SGDClassifier()
clf = clf.fit(train, predictions)
predict=clf.predict(train)
print(predict)
ls=loss(clf,train1, predictions1)
print("loss:",ls)
sc=score(clf,train, predictions)
print("score:",sc)
print(my_custom_loss_func(predictions1,predictions))