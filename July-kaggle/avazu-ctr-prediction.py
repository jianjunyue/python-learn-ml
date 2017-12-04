
#https://github.com/owenzhang

#FM 模型：libFM, libMF, svdfeature

import pandas as pd
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# from utils import load_df

# Initial setup
train_filename = "../../data/avazu-ctr-prediction/train_small.csv"
test_filename = "../../data/avazu-ctr-prediction/test"
submission_filename = "../../data/avazu-ctr-prediction/sampleSubmission"

train_df = pd.read_csv(train_filename)
test_df = pd.read_csv(test_filename)
tcolumns="year,month,day,hours,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21".split(",")
def get_data(data):
    for c in tcolumns:
        if data[c].dtype=="object":
            lbl = LabelEncoder()
            lbl.fit(list(data[c].values))
            data[c] = lbl.transform(list(data[c].values))

    hour =data["hour"]
    data["hours"]=(hour%100).astype(np.uint32)
    hour=hour//100
    data["day"]=(hour%100).astype(np.uint32)
    hour = hour // 100
    data["month"]=(hour%100).astype(np.uint32)
    hour = hour // 100
    data["year"]=(hour%100).astype(np.uint32)
    return data

train_df= get_data(train_df)
test_df= get_data(test_df)
print(train_df.head())
# print(train_df.info())
# print(train_df.describe())


# for c in train_df.columns:
#     print("------------ "+c+" ----------")
#     print(train_df[c].value_counts())

# 结果衡量
def print_metrics(true_values, predicted_values):
    print("Accuracy: ", metrics.accuracy_score(true_values, predicted_values))
    print("AUC: ", metrics.roc_auc_score(true_values, predicted_values))
    print("Confusion Matrix: ", + metrics.confusion_matrix(true_values, predicted_values))
    print(metrics.classification_report(true_values, predicted_values))

# 拟合分类器
def classify(classifier_class, train_input, train_targets):
    classifier_object = classifier_class()
    classifier_object.fit(train_input, train_targets)
    return classifier_object

# 模型存储
def save_model(clf):
    joblib.dump(clf, 'classifier.pkl')

# 训练和存储模型
X_train, X_test, y_train, y_test = train_test_split(train_df[tcolumns], train_df[["click"]],test_size=0.3, random_state=0)
print("start")
classifier = classify(LogisticRegression, X_train, y_train)
predictions = classifier.predict(X_test)
print_metrics(y_test, predictions)
save_model(classifier)

# 按照指定的格式生成结果
def create_submission(ids, predictions, filename=submission_filename):
    # submission_df = pd.DataFrame({"id": ids, "click": predictions})
    submission_df = pd.DataFrame(data={'aid' : ids, 'click': predictions})
    print(submission_df.head())
    # submission_df.to_csv(submission_filename+"_sub", header=['id', 'click'], index=False)
    submission_df.to_csv(submission_filename + "_sub",index=False)


import numpy as np
from pandas import DataFrame

classifier = joblib.load('classifier.pkl')
# test_data_df = pd.read_csv(test_filename) #load_df('test.csv', training=False)
# print(test_df.head())
test_df['id'] = test_df['id'].astype(np.uint64)
ids = test_df["id"]
predictions = classifier.predict(test_df[tcolumns])
print(predictions)
create_submission(ids, predictions)
print("end")