import autokeras as ak
import openml
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import json
import time
from tensorflow.keras.models import load_model
import traceback
import sys
import os
import shutil


def get_datasets():
    '''
    Funkcja pobiera binarne datasety z OpenML study ID:218 - AutoML Benchmark.
    Zwraca dictionary, z nazwą datasetu jako klucz, a w nim dataset i nazwę zmiennej celu.
    '''
    data = {}
    tasks = openml.study.get_suite(218).tasks
    for task_id in tasks:
        task = openml.tasks.get_task(task_id)
        if len(task.class_labels) == 2: #tylko binarne
            dataset = task.get_dataset()
            data[dataset.name] = {
                'df' : dataset.get_data()[0], #dostajemy krotkę z której tylko pierwsze jest ważne
                'target_name' : dataset.default_target_attribute,
                'class_labels' : dataset.retrieve_class_labels()
            }
    return data
  
   

def run(X : pd.DataFrame,y : pd.DataFrame):
    start = time.time()
    
    kfold = KFold(n_splits=5, shuffle=True)
    
    
    acc_list = []
    prc_list = []
    f1_list = []
    auc_list = []
    
    best_acc = 0
    
    for fold, (train_index, test_index) in enumerate(kfold.split(X,y)):  # crossvalidation
        
        clf = ak.StructuredDataClassifier(max_trials = 10) 
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
        
        clf.fit(train_X, train_y)
        prediction = pd.DataFrame(clf.predict(test_X))
        test_y = test_y.apply(pd.to_numeric, errors='coerce').fillna(test_y) # problems with types...
        prediction = prediction.apply(pd.to_numeric, errors='coerce').fillna(prediction)  
        
        #scores
        
        acc = accuracy_score(test_y, prediction)
        
        acc_list.append(acc)
        prc_list.append(precision_score(test_y, prediction))
        f1_list.append(f1_score(test_y, prediction))
        auc_list.append(roc_auc_score(test_y,prediction))
        
        if acc > best_acc:
            best_acc = acc
            best_model = clf
            
        # shutil.rmtree(r'structured_data_classifier')

    scores_dict = {
        'accuracy' : acc_list, 
        'precision' : prc_list,
        'f1' : f1_list,
        'auc' : auc_list
    }
    
    end = time.time()
    execution_time = end - start
    
    return clf, scores_dict, execution_time
    

    
def save_json(key,content):
    '''
    Zapisywanie wyników do jsona, żeby potem ich użyć w ipynb/prezentacji.
    '''
    with open("results.json","r") as f:
        loaded = json.load(f)
    
    loaded[key] = content
    
    with open("results.json", "w") as output:
        json.dump(loaded,output, indent=4)
    
    
def save_model(clf,dataset_name):
    model = clf.export_model()
    try:
        model.save("models/model_" + dataset_name , save_format="tf")
    except Exception:
        model.save("models/model_" + dataset_name + ".h5")
        
def get_model(dataset_name):
    return load_model("models/model_" + dataset_name, custom_objects=ak.CUSTOM_OBJECTS)
        

    
    
    
def main():
    
    data = get_datasets()
    
    with open("results.json","r") as f:
        loaded = json.load(f)
    
    for ds in list(data.keys()):
        if ds in (loaded.keys()) or ds in ('KDDCup09_appetency','guillermo',"riccardo",'christine','albert'): 
            continue
        
        print(ds,"in progress...") 

        target_name = data[ds]["target_name"]
        df = data[ds]["df"]

        y = pd.DataFrame(df[target_name])
        X = df.drop(columns = target_name)
        
        if ds == 'kr-vs-kp':
            y[target_name] = y[target_name].map({'nowin':0, 'won' : 1})
        elif ds == 'credit-g':
            y[target_name] = y[target_name].map({'bad':0, 'good':1})
        elif ds == 'adult':
            y[target_name] = y[target_name].map({'<=50K':0, '>50K':1})
        elif ds == 'APSFailure':
            y[target_name] = y[target_name].map({'neg':0, 'pos':1})
            

        
        try:
            clf, scores_dict, execution_time = run(X,y)
            save_model(clf = clf,dataset_name = ds)
            save_json(ds,{"scores" : scores_dict, "time" : execution_time})
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            message =  ''.join(lines)  
            save_json(ds,message)
        print(ds + " done")
        
if __name__ == '__main__':
    main()