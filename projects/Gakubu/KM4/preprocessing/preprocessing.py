import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessing_pipeline(X) -> ColumnTransformer:
    """Function which creates Pipeline for given X"""

    num_col_all = X.select_dtypes(include="number").columns.to_list()
    cat_col_all = X.select_dtypes(include="object").columns.to_list()

    numeric_pipe = Pipeline(
        steps=[
            ("outlier", OutlierRemover(factor=1.2)),
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("impute", KNNImputer(n_neighbors=5)),
            ("encode", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, num_col_all),
            ("categoric", categorical_pipe, cat_col_all),
            ("drop", IdentifierColumnsRemover(), num_col_all + cat_col_all),
        ],
        remainder="passthrough",
    )


class IdentifierColumnsRemover(BaseEstimator, TransformerMixin):
    """Class responsible for dropping columns."""

    __slots__ = "col_to_drop"

    def __init__(self, factor=0.6):
        self.factor = factor

    def fit(self, X, y=None):
        temp = set(
            X.loc[:, X.nunique() == X.shape[0]].columns.to_list()
            + X.loc[
                :, X.isin([" ", "NULL", np.nan]).mean() < self.factor
            ].columns.to_list()
        )  # to o czym pisal kacper
        self.col_to_drop = list(temp)
        return self

    def transform(self, X, y=None):
        X_ = X.drop(self.col_to_drop, axis=1)
        return X_


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Class responsible for removing outliers."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i])] = self.lower_bound[i]
            x[(x > self.upper_bound[i])] = self.upper_bound[i]
            X.iloc[:, i] = x
        return X
