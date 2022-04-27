import autokeras as ak
from openml import datasets
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import time
from tensorflow.keras.models import load_model

def get_datasets(names):
    '''
    Funkcji podaje się listę nazw datasetów, ona szuka je w openml i zwraca je. Zwraca też listę nazw, których nie znalazło.
    '''
  
    ds = pd.DataFrame(datasets.list_datasets()).transpose().reset_index(drop = True)
    # zbieranie datasetów z api openmlowego
    matches = ds[np.in1d(ds.name,names)]  # to co się udalo znalezc po nazwie
    matches = matches[matches.status == "active"]   # wywalam nieaktywne
    matches.version = matches.version.astype(int)   # żeby mogło wybrać największą liczbę
    matches = matches.groupby("name").apply(lambda d: d.nlargest(1,columns = "version")) # wybieram te z najnowszą wersją
    matches.reset_index(drop = True, inplace = True)
    # jak wyszukiwałem po nazwie to nie znajdowało mi 9 datasetów
    unmatched = names[np.where(np.logical_not(np.in1d(names,ds.name)))[0]]    # to są te nieznalezione
    # zbieram datasety z api openmlowego uzywając ich id 
    ds_list = datasets.get_datasets(dataset_ids = matches.did)
    return ds_list, unmatched
    

def run(X : pd.DataFrame,y : pd.DataFrame):
    
    start = time.time()
    
    train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=420)

    clf = ak.StructuredDataClassifier() 
    clf.fit(train_X, train_y)
    
    # następne trzy linijki bo były problemy z typami
    prediction = pd.DataFrame(clf.predict(test_X))
    test_y = test_y.apply(pd.to_numeric, errors='coerce').fillna(test_y)
    prediction = prediction.apply(pd.to_numeric, errors='coerce').fillna(prediction)

     
    accuracy = accuracy_score(test_y, prediction)
    recall = recall_score(test_y, prediction,average='micro')
    precision = precision_score(test_y, prediction,average='micro')
    f1 = f1_score(test_y, prediction, average='micro')
    # auc = roc_auc_score(test_y,prediction) # trzeba ogarnac pewnie dla multilabelow
    
    
    scores_dict = {
        'accuracy' : accuracy, 
        'recall' : recall,
        'precision' : precision,
        'f1_micro' : f1
        # ,
        # 'auc' : auc
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
    dataset_names = np.array([       # nazwy wzięte ręcznie z dokumentu, bo nie działają linki do openml
        "adult", "airlines",
        # "albert",  #możliwe, że działa, mi zacina
        "amazon employee...", "apsfailure" ,"australian" ,
        "bank-marketing",
        "blood-transfusion" ,
        # "christine" , #wywala blad
        "credit-g" ,
        "guiellermo" ,
        "higgs" ,"jasmine" ,"kc1" ,"kddcup09 appetency" ,"kr-vs-kp" ,"miniboone" ,
        "nomao" ,"numerai28.6" ,"phoneme" ,
        # "riccardo" , #wywala blad
        "sylvine",
        "car" ,"cnae-9" ,"connect-4" ,"covertype",
        # "dilbert" ,"dionis" ,"fabert" , # wywalają błąd
        "fashion-mnist" ,
        # "helena" , # wywala kernela
        #"jannis" , # wywala kernala
        "jungle chess...","mfeat-factors" ,
        # "robert" , #wywala blad
        "segment" ,
        "shuttle" ,"vehicle" ,
        #"volkert" # wywala blad
        ])
    datas, unmatched = get_datasets(dataset_names)

    save_json("unmatched", list(unmatched.astype(str)))
    
    for data in datas:
        print(data.name + " in progress...")
        # ogólnie to y zawsze jest None, a targetowa zmienna wydaje się być zawsze na końcu X (niepotwierdzone info)
        X, y, categorical_indicator, attribute_names = data.get_data()
        y = pd.DataFrame(X.iloc[:,X.shape[1]-1])
        X = X.iloc[:,0:(X.shape[1]-1)]
        try:
            clf, scores_dict, execution_time = run(X,y)
            save_model(clf = clf,dataset_name = data.name)
            save_json(data.name,{"scores" : scores_dict, "time" : execution_time})
        except:
            save_json(data.name,"failed")
        print(data.name + " done")
        
if __name__ == '__main__':
    main()