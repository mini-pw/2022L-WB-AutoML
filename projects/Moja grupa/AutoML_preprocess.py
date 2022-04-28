from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from autosklearn.metrics import roc_auc
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

def fit_automl(X:pd.DataFrame, y:pd.DataFrame):
    #daty na wejsciu musza byc typu datetime
    automl = AutoSklearn2Classifier(
        time_left_for_this_task = 60,
        per_run_time_limit = 27, # mniej niż połowa time_left_for_this_task
        disable_evaluator_output=False,
        n_jobs = -1,
        memory_limit = None,
        metric=roc_auc
        )
    automl.fit(X, y)
    return automl

def preprocess(X, y):
    for col in X.select_dtypes(['datetime']).columns:
        X[col + "_day"] = pd.to_datetime(X[col]).dt.day
        X[col + "_month"] = pd.to_datetime(X[col]).dt.month
        X[col + "_year"] = pd.to_datetime(X[col]).dt.year
        X.drop(col, axis=1, inplace=True)
    
    #Kolumny liczbowe w których jest mniej niż 10 unikalnych wartości zamieniamy na kategorie
    for col in X.select_dtypes(['number']).columns:  
        if (X[col].nunique() < 10):
            X[col] = X[col].astype("category")
    
    #zamieniamy kolumny typu object na categorical
    object_columns = X.select_dtypes(['object']).columns
    for column in object_columns:
        X[column] = X[column].astype('category')
        
    #usuwamy kolumny ze wszystkimi wartościami na
    X = X.dropna(axis=1, how='all')

        
    return X, y


def cross_valid(predictors, target, indexes):
    X, y = preprocess(predictors, target)
    result = {}
    result["accuracy"] = []
    result["auc"] = []
    
    for index in indexes:
        X_train = X.iloc[index.train]
        y_train = y.iloc[index.train]
        model = fit_automl(X_train, y_train)
        
        X_test = X.iloc[index.test]
        y_test = y.iloc[index.test]
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        result["accuracy"].append(accuracy_score(y_test, y_pred))
        result["auc"].append(roc_auc_score(y_test, y_proba))
    return result