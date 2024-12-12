from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

"""""
def tune_xgboost(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    search = RandomizedSearchCV(xgb, param_dist, scoring='accuracy', cv=3, n_iter=10, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_


def tune_catboost(X_train, y_train):
    param_dist = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    }
    catboost = CatBoostClassifier(verbose=0, random_state=42)
    search = RandomizedSearchCV(catboost, param_dist, scoring='accuracy', cv=3, n_iter=10, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

"""""
def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [100],
        #'max_depth': [None, 10, 20, 30],
        #'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_dist, scoring='accuracy', cv=3, n_iter=10, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_