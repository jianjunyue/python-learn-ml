# from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#调参
class ParameterTuning(object):
    def __init__(self):
        return

    # 分类模型 Classification scoring
    # ‘accuracy’    metrics.accuracy_score
    # ‘average_precision’    metrics.average_precision_score
    # ‘f1’    metrics.f1_score
    # ‘f1_micro’    metrics.f1_score
    # ‘f1_macro’    metrics.f1_score
    # ‘f1_weighted’    metrics.f1_score
    # ‘f1_samples’    metrics.f1_score
    # ‘neg_log_loss’    metrics.log_loss
    # ‘precision’ etc.metrics.precision_score
    # ‘recall’ etc.metrics.recall_score
    # ‘roc_auc’    metrics.roc_auc_score
    def fit_Classifier(self,alg,param_grid,X_train,y_train):
        grid_search = GridSearchCV(estimator=alg, param_grid=param_grid, scoring='roc_auc', n_jobs=4,iid=False, cv=5)
        grid_result = grid_search.fit( X_train, y_train)
        print("-----------grid_scores_-------------")
        print(grid_result.grid_scores_)
        print("-------------best_params_-----------")
        print(grid_result.best_params_)
        print("------------best_score_------------")
        print(grid_result.best_score_)
        print("Best: %f using %s " % (grid_result.best_score_, grid_result.best_params_))
        # print(grid_result.cv_results_)
        train_means = grid_result.cv_results_["mean_train_score"]
        test_means = grid_result.cv_results_["mean_test_score"]
        params = grid_result.cv_results_["params"]

        print("------------alg.feature_importances_------------")
        print(alg.feature_importances_)

        for train_mean,test_mean, param in zip(train_means,test_means, params):
            print("%f , %f with %r" % (train_mean,test_mean, param))

        # print(type(train_means))
        # print(type(test_means))
        # print(params)
        # self.pltshow(train_means, test_means, params)

        return train_means,test_means, params


    # 回归模型 Regression scoring
    # ‘explained_variance’    metrics.explained_variance_score
    # ‘neg_mean_absolute_error’    metrics.mean_absolute_error
    # ‘neg_mean_squared_error’    metrics.mean_squared_error
    # ‘neg_mean_squared_log_error’    metrics.mean_squared_log_error
    # ‘neg_median_absolute_error’    metrics.median_absolute_error
    # ‘r2’    metrics.r2_score
    def fit_Regression(self,alg,param_grid,X_train,y_train):
        grid_search = GridSearchCV(estimator=alg, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=4,iid=False, cv=5)
        grid_result = grid_search.fit( X_train, y_train)
        print("-----------grid_scores_-------------")
        print(grid_result.grid_scores_)
        print("-------------best_params_-----------")
        print(grid_result.best_params_)
        print("------------best_score_------------")
        print(grid_result.best_score_)
        print("Best: %f using %s " % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_["mean_test_score"]
        params = grid_result.cv_results_["params"]

        for mean, param in zip(means, params):
            print("%f with %r" % (mean, param))

    # 聚类模型 Regression scoring
    # ‘adjusted_mutual_info_score’    metrics.adjusted_mutual_info_score
    # ‘adjusted_rand_score’    metrics.adjusted_rand_score
    # ‘completeness_score’    metrics.completeness_score
    # ‘fowlkes_mallows_score’    metrics.fowlkes_mallows_score
    # ‘homogeneity_score’    metrics.homogeneity_score
    # ‘mutual_info_score’    metrics.mutual_info_score
    # ‘normalized_mutual_info_score’    metrics.normalized_mutual_info_score
    # ‘v_measure_score’    metrics.v_measure_score
    def fit_Clustering(self,alg,param_grid,X_train,y_train):
        grid_search = GridSearchCV(estimator=alg, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=4,iid=False, cv=5)
        grid_result = grid_search.fit( X_train, y_train)
        print("-----------grid_scores_-------------")
        print(grid_result.grid_scores_)
        print("-------------best_params_-----------")
        print(grid_result.best_params_)
        print("------------best_score_------------")
        print(grid_result.best_score_)
        print("Best: %f using %s " % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_["mean_test_score"]
        params = grid_result.cv_results_["params"]

        for mean, param in zip(means, params):
            print("%f with %r" % (mean, param))

    def pltshow(self,train_means,test_means, params):
        params_value=[]

        for para in params:
            params_value.append(para["n_estimators"])

        print("------------pltshow-------------")
        print(params_value)
        print(train_means)
        print(test_means)
        plt.plot(params_value, train_means, 'r-')
        plt.plot(params_value, test_means, 'b-')
        plt.show()




    def run_model(self,clf_key,clf):
        loss_error=self.predict_ls(clf_key, clf)
        return loss_error

    def predict_ls(self,clf_key, clf):
        clf.fit(self.X_train, self.y_train)
        y_pre = clf.predict(self.X_train)
        # stacker_pred_ensemble_df[clf_key] = y_pre
        loss_error = self.def_loss_score(y_pre, self.y_train.values.ravel())
        print("loss_error:", loss_error)
        return loss_error

    def def_loss_score(self,y_pre,y):
        loss_error = 0;
        z = zip(y_pre, y)
        for py, ty in z:
            loss_error += abs(py-ty)
        return loss_error / len(y)