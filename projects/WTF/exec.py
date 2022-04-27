import os
import pandas as pd
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import tempfile as tmp
import warnings

def funkcja(X,y, fold_time = 300, time_per_model = 75, n_folds = 10): 

    os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    
    for feature in X.columns:
         if X[feature].dtype == 'object':
            X[feature] = X[feature].astype('category')
    if y.dtype == 'object':
        y = y.astype('category')
    
    SKF = StratifiedKFold(n_splits=n_folds)
    scores = [0] * n_folds
    models = [0] * n_folds
    i = 0 
    for train_idx, test_idx in SKF.split(X,y):
        X_train= X.iloc[train_idx]
        y_train= y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
            
        api = TabularClassificationTask()
        api.search(X_train= X_train, y_train= y_train,X_test = X_test,
                   y_test = y_test, optimize_metric='roc_auc',total_walltime_limit=fold_time, 
                   func_eval_time_limit_secs=time_per_model,  memory_limit=None) 
        y_pred = api.predict_proba(X_test)
        score = roc_auc_score(y_test, y_pred[:, 1])
        scores[i] = score
        models[i] = api
        i+=1
    
    return models, scores
    