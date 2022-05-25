import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from preprocessing import preprocess
from sklearn.dummy import DummyClassifier
import skopt
from sklearn.pipeline import Pipeline

cfg = {
    LogisticRegression: {
        'penalty' : skopt.space.Categorical(['elasticnet']),
        'C' : skopt.space.Real(0.01, 4),
        'solver' : skopt.space.Categorical(['saga']),
        'l1_ratio': skopt.space.Real(0.01, 0.99),
        'max_iter': skopt.space.Categorical([10000])
    },
    RandomForestClassifier: {
        'n_estimators': skopt.space.Integer(5, 1000), 
        'max_features': skopt.space.Categorical(['auto', 'sqrt']), 
        'max_depth': skopt.space.Integer(3, 20), 
        'min_samples_split': skopt.space.Integer(2, 10),
        'min_samples_leaf': skopt.space.Integer(1,5), 
        'bootstrap': skopt.space.Categorical([True, False])
    },
    AdaBoostClassifier : {
        'n_estimators' : skopt.space.Integer(5, 100),
        'learning_rate' : skopt.space.Real(1e-6, 1)
    },
    SVC : {
        'C' : skopt.space.Real(1e-6, 100.0, 'log-uniform'),
        'kernel' : skopt.space.Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
        'degree' : skopt.space.Integer(1, 5),
        'gamma' : skopt.space.Real(1e-6, 100.0, 'log-uniform')
    },
    GaussianNB : {
        'var_smoothing': skopt.space.Real(1e-9,1, prior='log-uniform')
    },
    KNeighborsClassifier : {
        'n_neighbors' : skopt.space.Integer(3, 10)
    }
}

    
ensembles = [
        StackingClassifier([
        ('svc', SVC(probability=True)),
        ('abc', AdaBoostClassifier()),
        ('rfc', RandomForestClassifier())],
        final_estimator=LogisticRegression()
    ),
    StackingClassifier([
        ('rfc1', RandomForestClassifier()),
        ('en1', AdaBoostClassifier()),
        ('en2', AdaBoostClassifier()),
        ('rfc2', RandomForestClassifier())],
        final_estimator=LogisticRegression()
    ),
    StackingClassifier([
        ('svc', SVC(probability=True)),
        ('gnb', GaussianNB()),
        ('knn', KNeighborsClassifier())],
        final_estimator=LogisticRegression()
    ),
    StackingClassifier([
        ('knn1', KNeighborsClassifier()),
        ('abc', AdaBoostClassifier()),
        ('knn2', KNeighborsClassifier())],
        final_estimator=LogisticRegression()
    ),
    StackingClassifier([
        ('rfc', RandomForestClassifier()),
        ('abc', AdaBoostClassifier())],
        final_estimator=LogisticRegression()
    ),
    StackingClassifier([
        ('abc1', AdaBoostClassifier()),
        ('abc2', AdaBoostClassifier())],
        final_estimator=LogisticRegression()
    ),
    StackingClassifier([
        ('abc1', AdaBoostClassifier()),
        ('gnb1', GaussianNB()),
        ('rfc1', RandomForestClassifier()),
        ('svc1', SVC(probability=True))
        ],
        final_estimator=LogisticRegression()
    )
]

def set_model(prefix, model_settings):
    local_set = {}
    for key, value in model_settings.items():
        local_set["ensemble__"+ prefix + "__" + key] = value
    return local_set

def create_folds(X, y, spliter):
    folds = []
    for train, test in spliter.split(X, y): 
        folds.append((train, test))
    return folds

def select_model(X, y, metric, spliter):
    column_transformer = preprocess(X, y)
    folds = create_folds(X, y, spliter)
    scores = []
    
    for ensemble in ensembles:
        pipe = Pipeline(steps=[
            ("proc", column_transformer),
            ("ensemble", ensemble)
        ])
        ensemble_config = {}
        for prefix, model in ensemble.get_params()['estimators']:
            model_settings = cfg[type(model)]
            ensemble_config.update(set_model(prefix, model_settings))

        final_model = ensemble.get_params()["final_estimator"]
        model_settings = cfg[type(final_model)]
        ensemble_config.update(set_model("final_estimator", model_settings))
        bs = BayesSearchCV(
            estimator=pipe,
            search_spaces=ensemble_config,
            n_iter=30,
            scoring=metric,
            n_jobs=-1,
            cv=folds,
            refit=True,
            verbose=1
        )
        print(f"Bayes search of: {ensemble}")
        bs.fit(X, y)
        print(f"Best score: {bs.best_score_}")
        scores.append((bs.best_estimator_, bs.best_score_))
        
    max_val = -np.inf
    best_model = DummyClassifier()
    for model, score in scores:
        if score > max_val:
            max_val = score
            best_model = model
                
    return scores, best_model
    



