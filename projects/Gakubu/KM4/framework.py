import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from preprocessing.preprocessing import preprocessing_pipeline
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class GakubuFramework:
    """
    Class representing our framework.

    ...

    Attributes
    ----------
    pipeline : Pipeline
        pipeline used for preprocessing
    leaderboard : DataFrame

    best_model : model type from scikit-learn
        best model out from leaderboard with highest ROC AUC score
    final_models : list with trained models
        list with trained models

    Methods
    -------
    fit(X, y)
        Fit the framework, train the models

    setBestModel()
        Sets bests model after fitting

    showBestModel()
        Print best model (highest ROC AUC score)

    predict()
        Predicts labels

    predict_proba()
        Predicts probability of the positive class
    """

    __slots__ = "pipeline", "leaderboard", "best_model", "final_models"

    def __init__(self):
        self.leaderboard = pd.DataFrame(
            columns=[
                "Model",
                "ROC_AUC",
                "Accuracy Score",
                "F1 score",
                "Recall score",
                "Precision",
            ]
        )
        self.final_models = []
        self.best_model = None

    def fit(self, X, y):
        y["Y"] = np.where(y["Y"] == "Good", "1", "0")
        y["Y"] = y["Y"].astype("int")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y
        )

        self.pipeline = preprocessing_pipeline(X)

        self.pipeline.fit(X_train)
        X_train = self.pipeline.transform(X_train)
        X_test = self.pipeline.transform(X_test)

        models = [
            LogisticRegression(),
            RandomForestClassifier(),
            XGBClassifier(),
            SVC(probability=True),
        ]
        models.append(
            VotingClassifier(
                estimators=[
                    ("logreg", models[0]),
                    ("ranfor", models[1]),
                    ("xgb", models[2]),
                    ("svc", models[3]),
                ],
                voting="soft",
            )
        )

        # i = 0
        for model in models:
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            roc_score = roc_auc_score(y_test, y_proba)

            y_proba = np.where(y_proba >= 0.5, 1, 0)

            accuracy = accuracy_score(y_test, y_proba)
            f1 = f1_score(y_test, y_proba)
            precision = precision_score(y_test, y_proba)
            recall = recall_score(y_test, y_proba)

            self.final_models.append(model)

            temp_df = pd.DataFrame(
                {
                    "Model": [model],
                    "ROC_AUC": [roc_score],
                    "Accuracy Score": [accuracy],
                    "F1 score": [f1],
                    "Recall score": [recall],
                    "Precision": [precision],
                }
            )

            self.leaderboard = pd.concat(
                [self.leaderboard, temp_df], ignore_index=True, axis=0
            )

    def setBestModel(self):
        idx = self.leaderboard[
            self.leaderboard["ROC_AUC"] == max(self.leaderboard["ROC_AUC"])
        ].index
        self.best_model = self.final_models[idx[0]]

    def showBestModel(self):
        if self.best_model is None:
            self.setBestModel()
        idx = self.leaderboard[
            self.leaderboard["ROC_AUC"] == max(self.leaderboard["ROC_AUC"])
        ].index
        self.best_model = self.final_models[idx[0]]
        print(self.best_model)

    def predict(self, X):
        if self.best_model is None:
            self.setBestModel()
        X = self.pipeline.transform(X)
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            self.setBestModel()
        X = self.pipeline.transform(X)
        return self.best_model.predict_proba(X)
