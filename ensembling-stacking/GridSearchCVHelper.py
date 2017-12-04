from sklearn.model_selection import GridSearchCV

#模型参数选择
class GridSearchCVHelper(object):
    def __init__(self):
        return

    def fit(self,alg,param_grid,X_train,y_train):
        grid_search = GridSearchCV(estimator=alg, param_grid=param_grid, scoring='roc_auc', n_jobs=4,iid=False, cv=5)
        grid_result = grid_search.fit( X_train, y_train)
        print(grid_result.grid_scores_)
        print(grid_result.best_params_)
        print(grid_result.best_score_)
        print("Best: %f using %s " % (grid_result.best_score_, grid_result.best_params_))
        train_means = grid_result.cv_results_["mean_train_score"]
        test_means = grid_result.cv_results_["mean_test_score"]
        params = grid_result.cv_results_["params"]

        print("------------alg.feature_importances_------------")
        print(alg.feature_importances_)

        for train_mean,test_mean, param in zip(train_means,test_means, params):
            print("%f , %f with %r" % (train_mean,test_mean, param))

        return train_means,test_means, params

