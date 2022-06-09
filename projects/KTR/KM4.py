import logging
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from team5.preprocessing.preprocessor import get_pre


class OurAutoML:
    def __init__(self, scoring='roc_auc', n_iter=5):
        self.scoring = scoring
        self.n_iter = n_iter
        self._logger = logging.getLogger(__name__)
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params = None
        self.model_list = [
            LogisticRegression(),
            #KNeighborsClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            #DecisionTreeClassifier(),
        ]
        param_grids = []
        #regression
        param_grids.append({"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"], "solver":["saga"]})
        #kneighbors
        '''param_grids.append({'n_neighbors': (1, 10, 1),
                            'leaf_size': (20, 40, 1),
                            'p': (1, 2),
                            'weights': ('uniform', 'distance'),
                            'metric': ('minkowski', 'chebyshev')})
        '''
        #RandomForest
        param_grids.append({'bootstrap': [True, False],
                            'max_depth': [10, 25, 50, 75, 100, None],
                            'max_features': ['log2', 'sqrt'],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10],
                            'n_estimators': [250, 500, 750, 1000, 1250, 1500, 1750, 2000]})
        #GBC
        param_grids.append({"learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "min_samples_split": np.linspace(0.1, 0.5, 12),
                            "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                            "max_depth": [3, 5, 8],
                            "subsample": [0.5, 0.8, 0.9, 1.0]})
        #DTC
        '''param_grids.append({'max_depth': [1, 2, 5, 10, 15, 20],
                            'min_samples_leaf': [5, 10, 20, 50, 75, 100],
                            'criterion': ["gini", "entropy"],
                            'splitter': ['best', 'random'],
                            'max_features': [ 'sqrt', 'log2', None]})
        '''
#        param_grids.append({'C': [0.1, 0.5, 1, 50, 100],
#                            'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
#                            'kernel': ['rbf', 'poly', 'sigmoid'],
#                            'degree': [2, 3, 5]})
#
        self.param_grids = param_grids

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, cv=5):
        """
        The fit method searches for the best model and its hyperparameter config.
        For preprocessing, we're using the preprocessor class designed by us for the previous milestone
        Parameters:
        :param x: pd.DataFrame of features
        :param y: pd.DataFrame of labels
        :param cv: Cross-validation
        :return:
        """
        __start = perf_counter()
        preprocessing = get_pre()
        x_transformed = preprocessing.fit_transform(x, y)
        self._logger.info(f"preprocessing finished in {perf_counter() - __start}")
        self.best_estimator_ = None
        self.best_score_ = 0
    
        for i in trange(len(self.model_list)):
            
            random_search = RandomizedSearchCV(estimator=self.model_list[i], param_distributions=self.param_grids[i],
                                            n_iter=self.n_iter, scoring=self.scoring, cv=cv, n_jobs=-1, verbose=2)
            random_search.fit(x_transformed, y)
            
            if random_search.best_score_ > self.best_score_:
                self.best_score_ = random_search.best_score_
                self.best_estimator_ = random_search.best_estimator_
                self.best_params = random_search.best_params_

    
    def predict(self, x, y=None):
        return self.best_estimator_.predict(x)

    def predict_proba(self, x, y=None):
        return self.best_estimator_.predict_proba(x)
