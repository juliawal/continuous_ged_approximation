'''
Module pour inférer les propriétés des graphes à partir des distances
'''


def evaluate_D(D_app, y_app, D_test, y_test, mode='reg'):
    '''
    '''
    # mode en Enum
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import accuracy_score as accuracy
    from sklearn.model_selection import GridSearchCV

    if (mode == 'reg'):
        knn = KNeighborsRegressor(metric='precomputed')
        scoring = 'neg_root_mean_squared_error'
        perf_eval = mse
    else:
        knn = KNeighborsClassifier(metric='precomputed')
        scoring = 'accuracy'
        perf_eval = accuracy
    grid_params = {
        'n_neighbors': [3, 5, 7, 9, 11]
    }

    clf = GridSearchCV(knn, param_grid=grid_params,
                       scoring=scoring,
                       cv=5, return_train_score=True, refit=True)
    clf.fit(D_app, y_app)
    y_pred_app = clf.predict(D_app)
    y_pred_test = clf.predict(D_test)
    return perf_eval(y_pred_app, y_app), perf_eval(y_pred_test, y_test), clf
