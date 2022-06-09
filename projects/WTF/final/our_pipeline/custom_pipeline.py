import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from scikeras.wrappers import KerasClassifier

def custom_function(X, y, test):
    """
    Funkcja tworzy model, trenuje go na zbiorze treningowym oraz wyznacza prawdopodobieństwa dla zbioru testowego.
    Do stworzenia modelu wykorzystuje RFC, SVC oraz XGB, a także modele deep learningowe.
    Hiperparametry wyznaczane są za pomocą optymalizacji bayesowskiej.
    Ostateczny model wyznaczany jest przy pomocy ensemblingu wytrenowanych modeli.
    Wykonywany jest także preprocessing.


    Parametry:
        X: zbiór treningowy - zmienne objaśniające
        y: zbiór treningowy - zmienna objaśniana
        test: zbiór testowy - zmienne objaśniające
    Zwraca:
        model: Wytrenowany model
        prob: estymatory prawdopodobieństwa zbioru testowego
    """
    
    name = y.name
    df = pd.concat([X, y], axis=1)
    
    # preprocessing:
    df.drop_duplicates(inplace = True)
    df = df[df[name].notna()]
    
    y = df[name]
    X = df.drop(name, axis = 1)
    
    #imputacja
    im = KNNImputer(n_neighbors=1)
    X = pd.DataFrame(im.fit_transform(X.values), columns = X.columns, index = X.index)
    test = pd.DataFrame(im.transform(test.values), columns = test.columns, index = test.index)
    
    #encoding
    toOneHot = []
    for feature in X.columns:
        if X[feature].dtype == 'object':
            toOneHot.append(feature)
    ohe = OneHotEncoder() 
    index = X.index
    drop = X.drop(toOneHot, axis=1)
    X = ohe.fit_transform(X[toOneHot]).toarray()
    col = ohe.get_feature_names_out(toOneHot)
    X = pd.DataFrame(X, index = index, columns = col)
    X = X.join(drop)
    index = test.index
    drop = test.drop(toOneHot, axis = 1)
    test = ohe.transform(test[toOneHot]).toarray()
    col = ohe.get_feature_names_out(toOneHot)
    test = pd.DataFrame(test, index = index, columns = col)
    test = test.join(drop)
    
    #usunięcie kolumn o niskiej wariancji
    toDrop = X.std()[X.std() < 0.2].index.values
    X.drop(toDrop, axis=1, inplace = True)
    test.drop(toDrop, axis=1, inplace = True)
    
    #skalowanie do przedziału [0,1]
    sc = MinMaxScaler()
    X = pd.DataFrame(sc.fit_transform(X.values), columns = X.columns, index = X.index)
    test = pd.DataFrame(sc.transform(test.values), columns = test.columns, index = test.index)
    
    # przygotowanie i trenowanie modeli:
    #wyznaczenie przestrzeni do szukanie hiperparametrów 
    svc_params =  {
        'C': [0.1, 0.5, 1, 1.5, 2, 3],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }
    svc = SVC(probability=True, max_iter=100000)
    
    xgb_params = {"eta": [0.1, 0.2, 0.3, 0.4],
                  "gamma": (0, 20),
                  "max_depth": (6, 200)
                 }
    xgb = XGBClassifier()
    
    rfc_params = {"n_estimators": (500, 1000),
                  "max_depth": (5, 200),
                  "max_features": (1, 50),
                  'min_samples_split': (2, 30),
                  'min_samples_leaf': (1, 30),
                 }
    rfc = RandomForestClassifier()

    estimators=[('XGB', xgb, xgb_params), ('RF', rfc,rfc_params), ('SVC', svc, svc_params)]
    
    #wyznaczanie hiperparametrów za pomocą optymalizacji bayesowskiej
    models = []
    for model in estimators:
        clf = BayesSearchCV(
                estimator=model[1],
                search_spaces=model[2],
                scoring='roc_auc',
                random_state=42,
                n_jobs = 10,
                n_iter = 50,
                verbose = 1,
            )
        models.append((model[0], clf))
    
    #modele wykorzystujące deep learning
    model = Sequential()
    model.add(Dense(40, input_dim=X.shape[1], activation='sigmoid'))
    for i in range(5):
        model.add(Dropout(0.4))
        model.add(Dense(80, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    keras_clf = KerasClassifier(model, epochs=300, batch_size=10, verbose=0)
    models.append(('keras_sigomid', keras_clf))
    model = Sequential()
    model.add(Dense(40, input_dim=X.shape[1], activation='relu'))
    for i in range(5):
        model.add(Dropout(0.4))
        model.add(Dense(80, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    keras_clf = KerasClassifier(model, epochs=300, batch_size=10, verbose=0)
    models.append(('keras_relu', keras_clf))
    
    #ensembling modeli
    model = VotingClassifier(estimators=models, voting='soft', verbose=True)
    
    model.fit(X, y)
    prob = model.predict_proba(test)
    
    return model, prob