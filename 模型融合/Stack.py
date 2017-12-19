from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


#http://blog.csdn.net/hhy518518/article/details/54915900

# Stacking
def stackModel():
    input_df = pd.read_csv('train.csv', index_col=0)
    submit_df = pd.read_csv('test.csv', index_col=0)
    train_y = np.log1p(input_df.pop('SalePrice')).as_matrix()  # 训练标签
    df = pd.concat([input_df, submit_df])
    df = dataprocess.processData(df)
    input_df = df.loc[input_df.index]
    submit_df = df.loc[submit_df.index]

    train_X = input_df.values
    test_X = submit_df.values

    clfs = [RandomForestRegressor(n_estimators=500, max_features=.3),
            XGBRegressor(max_depth=6, n_estimators=500),
            Ridge(15)]
    # 训练过程
    dataset_stack_train = np.zeros((train_X.shape[0], len(clfs)))
    dataset_stack_test = np.zeros((test_X.shape[0], len(clfs)))
    for j, clf in enumerate(clfs):
        clf.fit(train_X, train_y)
        y_submission = clf.predict(test_X)
        y_train = clf.predict(train_X)
        dataset_stack_train[:, j] = y_train
        dataset_stack_test[:, j] = y_submission
    print("开始Stacking....")
    clf = RandomForestRegressor(n_estimators=1000, max_depth=8)
    clf.fit(dataset_stack_train, train_y)
    y_submission = clf.predict(dataset_stack_test)
    predictions = np.expm1(y_submission)
    result = pd.DataFrame({"Id": submit_df.index, "SalePrice": predictions})
    result.to_csv('stack_result.csv', index=False)